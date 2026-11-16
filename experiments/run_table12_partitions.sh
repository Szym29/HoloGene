#!/usr/bin/env bash
# Table 12 -- three random section partitions per held-out ratio (20-60%).
# The split-seed-8848 runs at 20/40/50/60% are the Table 9 models.
set -euo pipefail
cd "$(dirname "$0")/.."
run() {  # run <config dir> <output dir>
  python experiments/run_stage.py gae --config "$1/gae.yaml"
  python experiments/run_stage.py train --config "$1/hologene_full.yaml"
  python experiments/run_stage.py eval --prediction "$2/hologene_full/prediction.h5ad" --out-dir "$2/hologene_full/eval"
}
for s in 8to2 6to4 5to5 4to6; do run "experiments/configs/split_$s" "outputs/mosta/split_$s"; done
for c in experiments/configs/table12_partitions/*/; do
  c=${c%/}; run "$c" "outputs/mosta/table12_partitions/$(basename "$c")"
done
