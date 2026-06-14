#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ModernTSF visual report generator.")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", dest="python_bin", default=sys.executable, help="Python executable with required packages.")
    parser.add_argument("--allow-gpu", action="store_true", help="Do not clear CUDA_VISIBLE_DEVICES.")
    args, passthrough = parser.parse_known_args()
    if passthrough[:1] == ["--"]:
        passthrough = passthrough[1:]
    args.passthrough = passthrough
    return args


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    if not args.allow_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [args.python_bin, "-m", "modern_tsf_visualizer.cli", *args.passthrough]
    print("REPO", args.repo)
    print("CMD", " ".join(cmd))
    subprocess.run(cmd, cwd=args.repo, env=env, check=True)


if __name__ == "__main__":
    main()
