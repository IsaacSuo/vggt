from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.adapters.nero import (  # noqa: E402
    _build_colmap_intrinsic_matrix,
    _extract_zip_member,
    _frame_sort_key,
    _load_trimesh_mesh,
    _read_colmap_cameras,
    _read_colmap_images,
    _select_frame_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render NeRO GlossyReal mesh masks by triangle rasterization.")
    parser.add_argument("--scene", default="bear")
    parser.add_argument(
        "--archive",
        default="/home/fangsuo/py/vggt/.local_benchmarks/nero/GlossyReal.tar.gz",
    )
    parser.add_argument(
        "--mesh-zip",
        default="/home/fangsuo/py/vggt/.local_benchmarks/nero/glossy-real-meshes-gt.zip",
    )
    parser.add_argument(
        "--transform-json",
        default="/home/fangsuo/py/vggt/.local_benchmarks/nero/bear_blender_transform.json",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/nero_glossyreal_raster_mask",
    )
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument(
        "--cluster-ratio",
        type=float,
        default=1.0 / 250.0,
        help="Vertex clustering size as a fraction of mesh bbox diagonal before rasterization.",
    )
    return parser.parse_args()


def simplify_mesh_vertex_clustering(
    vertices: np.ndarray,
    faces: np.ndarray,
    cluster_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    if cluster_size <= 0:
        return vertices, faces

    grid = np.floor(vertices / cluster_size).astype(np.int64)
    unique_keys, inverse = np.unique(grid, axis=0, return_inverse=True)
    del unique_keys

    simplified_vertices = np.zeros((inverse.max() + 1, 3), dtype=np.float64)
    counts = np.bincount(inverse)
    np.add.at(simplified_vertices, inverse, vertices)
    simplified_vertices /= counts[:, None]

    simplified_faces = inverse[faces]
    valid = (
        (simplified_faces[:, 0] != simplified_faces[:, 1])
        & (simplified_faces[:, 1] != simplified_faces[:, 2])
        & (simplified_faces[:, 0] != simplified_faces[:, 2])
    )
    simplified_faces = simplified_faces[valid]
    simplified_faces = np.sort(simplified_faces, axis=1)
    simplified_faces = np.unique(simplified_faces, axis=0)
    return simplified_vertices, simplified_faces


def project_vertices(vertices_world: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cam = vertices_world @ extrinsic[:, :3].T + extrinsic[:, 3]
    z = cam[:, 2]
    u = intrinsic[0, 0] * (cam[:, 0] / np.maximum(z, 1e-8)) + intrinsic[0, 2]
    v = intrinsic[1, 1] * (cam[:, 1] / np.maximum(z, 1e-8)) + intrinsic[1, 2]
    return np.stack([u, v], axis=-1), cam


def rasterize_mesh_mask(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_hw: tuple[int, int],
) -> np.ndarray:
    h, w = image_hw
    uv, cam = project_vertices(vertices_world, extrinsic, intrinsic)
    tri_cam = cam[faces]

    valid_depth = np.all(tri_cam[:, :, 2] > 1e-6, axis=1)
    tri_uv = uv[faces]
    tri_cam = tri_cam[valid_depth]
    tri_uv = tri_uv[valid_depth]
    if tri_uv.shape[0] == 0:
        return np.zeros((h, w), dtype=bool)

    edge1 = tri_cam[:, 1] - tri_cam[:, 0]
    edge2 = tri_cam[:, 2] - tri_cam[:, 0]
    normals = np.cross(edge1, edge2)
    facing = np.einsum("ij,ij->i", normals, tri_cam[:, 0]) < 0
    tri_uv = tri_uv[facing]
    if tri_uv.shape[0] == 0:
        return np.zeros((h, w), dtype=bool)

    min_xy = tri_uv.min(axis=1)
    max_xy = tri_uv.max(axis=1)
    in_view = (
        (max_xy[:, 0] >= 0)
        & (max_xy[:, 1] >= 0)
        & (min_xy[:, 0] < w)
        & (min_xy[:, 1] < h)
    )
    tri_uv = tri_uv[in_view]
    if tri_uv.shape[0] == 0:
        return np.zeros((h, w), dtype=bool)

    polygons = np.rint(tri_uv).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    chunk_size = 5000
    for start in range(0, polygons.shape[0], chunk_size):
        chunk = polygons[start : start + chunk_size]
        cv2.fillPoly(mask, chunk, color=255, lineType=cv2.LINE_8)
    return mask > 0


def load_transform(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    transform = payload["transform"] if "transform" in payload else payload["scenes"]["bear"]["transform"]
    return np.asarray(transform, dtype=np.float64)


def save_overlay(image: np.ndarray, mask: np.ndarray, output_base: Path) -> None:
    mask_u8 = (mask.astype(np.uint8) * 255)
    overlay = image.copy()
    red = np.zeros_like(image)
    red[..., 0] = 255
    alpha = 0.3
    overlay[mask] = np.clip((1.0 - alpha) * overlay[mask] + alpha * red[mask], 0, 255).astype(np.uint8)
    edge = np.zeros_like(mask, dtype=bool)
    edge[1:, :] |= mask[1:, :] != mask[:-1, :]
    edge[:-1, :] |= mask[:-1, :] != mask[1:, :]
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    edge &= mask
    overlay[edge] = np.array([0, 255, 0], dtype=np.uint8)
    Image.fromarray(image).save(output_base.with_name(output_base.name + "_image.png"))
    Image.fromarray(mask_u8, mode="L").save(output_base.with_name(output_base.name + "_mask.png"))
    Image.fromarray(overlay).save(output_base.with_name(output_base.name + "_overlay.png"))


def main() -> None:
    args = parse_args()
    scene = args.scene
    archive_path = Path(args.archive)
    mesh_zip_path = Path(args.mesh_zip)
    transform_path = Path(args.transform_json)
    output_dir = Path(args.output_dir) / scene
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = load_transform(transform_path)
    raw_mesh_path = _extract_zip_member(
        mesh_zip_path,
        f"{scene}-align.ply",
        Path("/tmp/nero_glossyreal_raster_mask_raw") / f"{scene}-align.ply",
    )
    mesh = _load_trimesh_mesh(raw_mesh_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    cluster_size = bbox_diag * args.cluster_ratio
    simp_vertices, simp_faces = simplify_mesh_vertex_clustering(vertices, faces, cluster_size)
    transformed_vertices = simp_vertices @ transform[:3, :3].T + transform[:3, 3]

    metadata = {
        "scene": scene,
        "cluster_size": cluster_size,
        "raw_vertices": int(vertices.shape[0]),
        "raw_faces": int(faces.shape[0]),
        "simplified_vertices": int(simp_vertices.shape[0]),
        "simplified_faces": int(simp_faces.shape[0]),
        "transform_json": str(transform_path),
    }

    with tarfile.open(archive_path, "r:*") as tf:
        cameras = _read_colmap_cameras(tf.extractfile(f"GlossyReal/{scene}/colmap/sparse/0/cameras.bin").read())
        image_records = _read_colmap_images(tf.extractfile(f"GlossyReal/{scene}/colmap/sparse/0/images.bin").read())
        image_members = {
            Path(m.name).name: m.name
            for m in tf.getmembers()
            if m.isfile() and m.name.startswith(f"GlossyReal/{scene}/images/")
        }
        frames = []
        for rec in sorted(image_records, key=lambda item: _frame_sort_key(str(item["name"]))):
            image_name = str(rec["name"])
            if image_name not in image_members:
                continue
            camera_id = int(rec["camera_id"])
            frames.append(
                {
                    "name": image_name,
                    "image_member": image_members[image_name],
                    "extrinsic": np.asarray(rec["extrinsic"], dtype=np.float32),
                    "intrinsic": _build_colmap_intrinsic_matrix(cameras[camera_id]).astype(np.float32),
                }
            )

        selected = _select_frame_ids(len(frames), args.num_frames, "evenly_spaced")
        for idx in selected.tolist():
            frame = frames[int(idx)]
            image = np.asarray(Image.open(io.BytesIO(tf.extractfile(frame["image_member"]).read())).convert("RGB"))
            h, w = image.shape[:2]
            mask = rasterize_mesh_mask(
                vertices_world=transformed_vertices,
                faces=simp_faces,
                extrinsic=frame["extrinsic"],
                intrinsic=frame["intrinsic"],
                image_hw=(h, w),
            )
            stem = Path(frame["name"]).stem
            save_overlay(image, mask, output_dir / stem)

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(output_dir)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
