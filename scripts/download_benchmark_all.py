#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_data_download import build_all_parser, run_all_download


def main() -> None:
    parser = build_all_parser()
    args = parser.parse_args()
    results = run_all_download(args)
    for path in results:
        print(f"Downloaded dataset to: {path}")


if __name__ == "__main__":
    main()
