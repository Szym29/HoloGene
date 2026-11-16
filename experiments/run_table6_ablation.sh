#!/usr/bin/env bash
# Table 6 (and MOSTA columns of Table 3) -- component ablations on MOSTA E16.5, 8:2 split.
# All five variants share the same GAE, data, split, seed and initial weights.
set -euo pipefail
cd "$(dirname "$0")/.."
C=experiments/configs/split_8to2

python experiments/run_stage.py gae --config "$C/gae.yaml"
for v in xyz film film_psn film_coral full; do
  out=outputs/mosta/split_8to2/hologene_$v
  python experiments/run_stage.py train --config "$C/hologene_$v.yaml"
  python experiments/run_stage.py eval --prediction "$out/prediction.h5ad" --out-dir "$out/eval"
done
