#!/usr/bin/env bash
# Table 14 -- Leiden resolution of the pseudo-labels (resolution 1.0 is the main model).
set -euo pipefail
cd "$(dirname "$0")/.."
H=data/mosta/E16.5_10percent_noCavity_coor_leiden_resolutions_top3000.h5ad
SOURCE=data/mosta/E16.5_10percent_noCavity_coor_leiden_hologene_full_top3000.h5ad
if [ ! -f "$H" ] || [ "$SOURCE" -nt "$H" ]; then
  python experiments/add_leiden_resolutions.py --input "$SOURCE" --output "$H"
fi
python experiments/run_stage.py gae --config experiments/configs/split_8to2/gae.yaml
python experiments/run_stage.py train --config experiments/configs/split_8to2/hologene_full.yaml
python experiments/run_stage.py eval --prediction outputs/mosta/split_8to2/hologene_full/prediction.h5ad \
  --out-dir outputs/mosta/split_8to2/hologene_full/eval
for c in experiments/configs/table14_leiden_resolution/*/; do
  c=${c%/}; out=outputs/mosta/table14_leiden_resolution/$(basename $c)
  python experiments/run_stage.py train --config "$c/hologene_full.yaml"
  python experiments/run_stage.py eval --prediction "$out/hologene_full/prediction.h5ad" --out-dir "$out/hologene_full/eval" --skip-biology
done
