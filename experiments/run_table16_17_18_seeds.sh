#!/usr/bin/env bash
# Tables 16-18 -- five training seeds (fixed split) and five section splits (fixed seed).
set -euo pipefail
cd "$(dirname "$0")/.."
for c in experiments/configs/table16a_training_seeds/*/ experiments/configs/table16b_split_seeds/*/; do
  c=${c%/}; out=outputs/mosta/$(basename $(dirname $c))/$(basename $c)
  python experiments/run_stage.py gae --config "$c/gae.yaml"
  python experiments/run_stage.py train --config "$c/hologene_full.yaml"
  python experiments/run_stage.py eval --prediction "$out/hologene_full/prediction.h5ad" --out-dir "$out/hologene_full/eval"
done
