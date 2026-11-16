#!/usr/bin/env bash
# Tables 10 & 11 -- controlled axial stride (k = 2, 3, 4; three training seeds).
set -euo pipefail
cd "$(dirname "$0")/.."
for k in 2 3 4; do
  for s in 8848 7848 7849; do
    C=experiments/configs/table10_stride/k${k}_seed$s
    out=outputs/mosta/table10_stride/k${k}_seed$s
    python experiments/run_stage.py gae --config "$C/gae.yaml"
    python experiments/run_stage.py train --config "$C/hologene_full.yaml"
    python experiments/run_stage.py eval --prediction "$out/hologene_full/prediction.h5ad" --out-dir "$out/hologene_full/eval"
    if [ "$k" = 4 ]; then
      python experiments/split_by_distance.py --run-dir "$out/hologene_full"
      for d in 1 2; do
        python experiments/run_stage.py eval --prediction "$out/hologene_full/distance$d/prediction.h5ad" --out-dir "$out/hologene_full/distance$d/eval"
      done
    fi
  done
done
