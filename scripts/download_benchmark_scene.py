#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_data_download import build_single_scene_parser, run_single_scene_download


def main() -> None:
    parser = build_single_scene_parser()
    args = parser.parse_args()
    result = run_single_scene_download(args)
    print(f"Downloaded scene to: {result}")


if __name__ == "__main__":
    main()
