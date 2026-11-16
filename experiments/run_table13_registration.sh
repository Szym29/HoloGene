#!/usr/bin/env bash
# Table 13 -- registration perturbations (three perturbation seeds per condition).
# The unperturbed row uses the Table 16a models with training seeds 7848, 12 and 8888.
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/run_stage.py gae --config experiments/configs/split_8to2/gae.yaml
run() {  # run <config dir> <output dir>
  python experiments/run_stage.py train --config "$1/hologene_full.yaml"
  python experiments/run_stage.py eval --prediction "$2/hologene_full/prediction.h5ad" --out-dir "$2/hologene_full/eval" --skip-biology
}
for s in 7848 12 8888; do
  c=experiments/configs/table16a_training_seeds/seed$s; out=outputs/mosta/table16a_training_seeds/seed$s
  python experiments/run_stage.py gae --config "$c/gae.yaml"
  python experiments/run_stage.py train --config "$c/hologene_full.yaml"
  python experiments/run_stage.py eval --prediction "$out/hologene_full/prediction.h5ad" --out-dir "$out/hologene_full/eval"
done
for c in experiments/configs/table13_registration/*/; do
  c=${c%/}; run "$c" "outputs/mosta/table13_registration/$(basename "$c")"
done
