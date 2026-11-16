#!/usr/bin/env bash
# Table 9 -- HoloGene on MOSTA E16.5 under different train:test slice ratios.
# Every ratio pretrains its own GAE on its own training slices.
set -euo pipefail
cd "$(dirname "$0")/.."
for s in 8to2 6to4 5to5 4to6 2to8; do
  C=experiments/configs/split_$s
  out=outputs/mosta/split_$s/hologene_full
  python experiments/run_stage.py gae --config "$C/gae.yaml"
  python experiments/run_stage.py train --config "$C/hologene_full.yaml"
  python experiments/run_stage.py eval --prediction "$out/prediction.h5ad" --out-dir "$out/eval"
done
