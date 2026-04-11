from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

from huggingface_hub import snapshot_download


TRANSLAB_HF_REPO = "Longxiang-ai/TransLab"
NERO_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1arqYrMxfPc7ZOCSwgZxgLjBRDZ_8leGV?usp=sharing"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {dst}")
    shutil.copytree(src, dst)


def _find_nero_scene_dir(root: Path, subset: str, scene: str) -> Optional[Path]:
    subset_lower = subset.lower()
    scene_lower = scene.lower()
    matches: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        parts_lower = [part.lower() for part in path.parts]
        if scene_lower not in parts_lower:
            continue
        if subset_lower not in parts_lower:
            continue
        matches.append(path)
    if not matches:
        return None
    matches.sort(key=lambda item: (len(item.parts), str(item)))
    return matches[0]


def _extract_matching_archives(root: Path, subset: str, scene: str, temp_extract_root: Path) -> Optional[Path]:
    subset_lower = subset.lower()
    scene_lower = scene.lower()
    archives = sorted(root.rglob("*.zip"))
    for archive in archives:
        name_lower = archive.name.lower()
        if subset_lower not in name_lower and scene_lower not in name_lower:
            continue
        extract_dir = temp_extract_root / archive.stem
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(extract_dir)
        match = _find_nero_scene_dir(extract_dir, subset=subset, scene=scene)
        if match is not None:
            return match
    return None


def _require_gdown():
    try:
        import gdown  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "gdown is required for NeRO downloads. Install it with `pip install gdown`."
        ) from exc
    return gdown


def _list_nero_remote_files(output_root: Path) -> List[object]:
    gdown_mod = _require_gdown()
    files = gdown_mod.download_folder(
        url=NERO_DRIVE_FOLDER_URL,
        output=str(output_root),
        quiet=True,
        remaining_ok=True,
        skip_download=True,
    )
    if files is None:
        raise RuntimeError("Failed to enumerate the official NeRO Drive folder.")
    return files


def _filter_nero_subset_files(files: List[object], subset: str) -> List[object]:
    subset_lower = subset.lower()
    matches: List[object] = []
    for item in files:
        path_lower = str(item.path).lower()
        base_lower = Path(str(item.path)).name.lower()
        if subset_lower == "glossysynthetic":
            if base_lower == "glossysynthetic.tar.gz":
                matches.append(item)
        elif subset_lower == "glossyreal":
            if base_lower in {"glossyreal.tar.gz", "glossy-real-meshes-gt.zip"}:
                matches.append(item)
    return matches


def _download_exact_nero_files(
    files: List[object],
) -> List[str]:
    gdown_mod = _require_gdown()
    outputs: List[str] = []
    for item in files:
        local_path = Path(item.local_path)
        _ensure_dir(local_path.parent)
        result = gdown_mod.download(
            url="https://drive.google.com/uc?id=" + item.id,
            output=str(local_path),
            quiet=False,
            resume=True,
        )
        if result is None:
            raise RuntimeError(f"Failed to download NeRO file for {item.path}")
        outputs.append(result)
    return outputs


def download_nero_subset(subset: str, output_root: Path) -> Path:
    target_root = output_root / "nero"
    _ensure_dir(target_root)
    remote_files = _list_nero_remote_files(target_root)
    matched_files = _filter_nero_subset_files(remote_files, subset=subset)
    if not matched_files:
        exposed = sorted(str(item.path) for item in remote_files)
        raise FileNotFoundError(
            f"Could not locate an official NeRO subset package for `{subset}`. "
            f"Exposed remote entries include: {exposed}"
        )
    _download_exact_nero_files(matched_files)
    return target_root


def download_translab_scene(scene: str, output_root: Path) -> Path:
    if not scene.startswith("scene_"):
        raise ValueError(f"Unexpected TransLab scene name: {scene}")

    target_root = output_root / "translab"
    _ensure_dir(target_root)
    snapshot_download(
        repo_id=TRANSLAB_HF_REPO,
        repo_type="dataset",
        local_dir=str(target_root),
        allow_patterns=[f"{scene}/**", ".gitattributes"],
        max_workers=2,
    )
    scene_dir = target_root / scene
    if not scene_dir.exists():
        raise FileNotFoundError(
            f"TransLab scene `{scene}` was not materialized under {scene_dir}. "
            "The official dataset layout may have changed."
        )
    return scene_dir


def download_translab_all(output_root: Path) -> Path:
    target_root = output_root / "translab"
    _ensure_dir(target_root)
    snapshot_download(
        repo_id=TRANSLAB_HF_REPO,
        repo_type="dataset",
        local_dir=str(target_root),
        max_workers=4,
    )
    return target_root


def download_nero_all(output_root: Path) -> Path:
    gdown = _require_gdown()
    target_root = output_root / "nero"
    _ensure_dir(target_root)
    gdown.download_folder(
        url=NERO_DRIVE_FOLDER_URL,
        output=str(target_root),
        quiet=False,
        remaining_ok=True,
        resume=True,
    )
    return target_root


def download_nero_scene(subset: str, scene: str, output_root: Path) -> Path:
    _ = scene
    return download_nero_subset(subset=subset, output_root=output_root)


def build_single_scene_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download one benchmark scene for local adapter development."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["nero", "translab"],
        help="Which dataset to download from.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root directory where the dataset subtree will be written.",
    )
    parser.add_argument(
        "--scene",
        required=True,
        help=(
            "Scene name. Example: `scene_01` for TransLab. "
            "For NeRO the official release is subset-granular, so this argument is accepted for CLI compatibility but ignored."
        ),
    )
    parser.add_argument(
        "--nero-subset",
        default="GlossySynthetic",
        help="NeRO subset name, e.g. `GlossySynthetic` or `GlossyReal`.",
    )
    return parser


def build_all_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download full benchmark datasets for server-side evaluation."
    )
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["nero", "translab", "all"],
        help="Which dataset(s) to download.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root directory where the dataset subtree will be written.",
    )
    return parser


def run_single_scene_download(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).expanduser().resolve()
    if args.dataset == "translab":
        return download_translab_scene(scene=args.scene, output_root=output_root)
    return download_nero_scene(
        subset=args.nero_subset,
        scene=args.scene,
        output_root=output_root,
    )


def run_all_download(args: argparse.Namespace) -> List[Path]:
    output_root = Path(args.output_root).expanduser().resolve()
    results: List[Path] = []
    if args.dataset in {"translab", "all"}:
        results.append(download_translab_all(output_root=output_root))
    if args.dataset in {"nero", "all"}:
        results.append(download_nero_all(output_root=output_root))
    return results
