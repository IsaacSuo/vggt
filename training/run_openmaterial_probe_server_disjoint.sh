#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${SCRIPT_DIR}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
  launch.py \
  --config openmaterial_probe_server_disjoint \
  "$@"
