#!/usr/bin/env bash
# Tables 1 & 2 -- "HoloGene (Ours)" on MOSTA E16.5 (default 8:2 slice split).
set -euo pipefail
cd "$(dirname "$0")/.."
C=experiments/configs/split_8to2

python experiments/run_stage.py gae --config "$C/gae.yaml"
python experiments/run_stage.py train --config "$C/hologene_full.yaml"
python experiments/run_stage.py eval --prediction outputs/mosta/split_8to2/hologene_full/prediction.h5ad \
  --out-dir outputs/mosta/split_8to2/hologene_full/eval
