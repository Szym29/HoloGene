# Reproducing the MOSTA E16.5 results

| paper table | configs (`experiments/configs/`) | script |
|---|---|---|
| Tables 1 & 2 | `split_8to2/{gae,hologene_full}.yaml` | `run_table1_table2.sh` |
| Table 6 (ablations) | `split_8to2/hologene_{xyz,film,film_psn,film_coral,full}.yaml` | `run_table6_ablation.sh` |
| Table 9 (train:test ratios) | `split_{8to2,6to4,5to5,4to6,2to8}/` | `run_table9_split_ratio.sh` |
| Tables 10 & 11 (axial stride) | `table10_stride/` | `run_table10_11_stride.sh` |
| Table 12 (random partitions) | `table12_partitions/` (+ `split_*`) | `run_table12_partitions.sh` |
| Table 13 (registration perturbations) | `table13_registration/` | `run_table13_registration.sh` |
| Table 14 (Leiden resolution) | `table14_leiden_resolution/` | `run_table14_leiden_resolution.sh` |
| Table 15 (label corruption) | `table15_label_noise/` | `run_table15_label_noise.sh` |
| Tables 16–18 (repeated runs) | `table16a_training_seeds/`, `table16b_split_seeds/` | `run_table16_17_18_seeds.sh` |

Run any script from the repository root or by its path. Each script trains missing or outdated
models and evaluates them; metrics are written to `outputs/mosta/<run>/eval/metrics.json`.
