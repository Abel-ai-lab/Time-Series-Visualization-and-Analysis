#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/data/jm/tsorchestra_repro_20260610/envs/tso_fast/bin/python}"
CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" -m modern_tsf_visualizer.cli \
  --config configs/finance1000.cloud.json \
  --ticker-book all
