"""Split held-out predictions by distance to the nearest observed section (Table 11).

    python experiments/split_by_distance.py --run-dir outputs/mosta/table10_stride/k4_seed8848/hologene_full

Writes ``<run-dir>/distance<d>/prediction.h5ad`` for every distance d (in section steps).
"""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    split = json.loads((run / "split.json").read_text())
    train_z = np.asarray(split["train_z"], dtype=np.float32)
    all_z = np.sort(np.concatenate([train_z, np.asarray(split["test_z"], dtype=np.float32)]))
    observed = np.flatnonzero(np.isin(all_z, train_z))

    pred = ad.read_h5ad(run / "prediction.h5ad")
    z = pred.obsm["spatial"][:, 2]
    rank = np.searchsorted(all_z, z)
    distance = np.abs(rank[:, None] - observed[None, :]).min(axis=1)
    source_mtime = max((run / "prediction.h5ad").stat().st_mtime_ns,
                       (run / "split.json").stat().st_mtime_ns)
    for d in np.unique(distance):
        out = run / f"distance{d}"
        destination = out / "prediction.h5ad"
        if destination.is_file() and destination.stat().st_mtime_ns >= source_mtime:
            print(f"distance {d}: current -> {destination}")
            continue
        out.mkdir(exist_ok=True)
        pred[distance == d].copy().write_h5ad(destination)
        print(f"distance {d}: {(distance == d).sum()} spots -> {destination}")


if __name__ == "__main__":
    main()
