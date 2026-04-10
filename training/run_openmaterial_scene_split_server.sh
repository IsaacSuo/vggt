#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
python training/data/preprocess/openmaterial_scene_split.py \
  --data_dir /opt/data/private/dataset/OpenMaterial_ablation \
  --output_dir /opt/data/private/fyp/vggt_runs/splits/openmaterial_scene_split_seed42 \
  --test_ratio 0.2 \
  --seed 42 \
  "$@"
