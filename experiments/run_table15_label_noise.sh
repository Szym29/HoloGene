#!/usr/bin/env bash
# Table 15 -- corrupted training pseudo-labels (3/5/10%) and the uncorrupted reference.
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/run_stage.py gae --config experiments/configs/split_8to2/gae.yaml
python experiments/run_stage.py train --config experiments/configs/split_8to2/hologene_full.yaml
python experiments/run_stage.py eval --prediction outputs/mosta/split_8to2/hologene_full/prediction.h5ad \
  --out-dir outputs/mosta/split_8to2/hologene_full/eval
for c in experiments/configs/table15_label_noise/*/; do
  c=${c%/}; out=outputs/mosta/table15_label_noise/$(basename $c)
  python experiments/run_stage.py train --config "$c/hologene_full.yaml"
  python experiments/run_stage.py eval --prediction "$out/hologene_full/prediction.h5ad" --out-dir "$out/hologene_full/eval" --skip-biology
done
