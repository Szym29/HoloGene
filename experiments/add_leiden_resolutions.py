"""Add Leiden pseudo-labels at several resolutions (Table 14).

    python experiments/add_leiden_resolutions.py \
        --input  data/mosta/E16.5_10percent_noCavity_coor_leiden_hologene_full_top3000.h5ad \
        --output data/mosta/E16.5_10percent_noCavity_coor_leiden_resolutions_top3000.h5ad

Adds obs columns ``leiden_res0p5``, ``leiden_res0p8``, ``leiden_res1p2``, ``leiden_res1p5``
computed on the neighbour graph stored in the file.
"""
import argparse

import scanpy as sc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolutions", type=float, nargs="+", default=[0.5, 0.8, 1.2, 1.5])
    args = parser.parse_args()
    adata = sc.read_h5ad(args.input)
    for r in args.resolutions:
        key = "leiden_res" + format(r, "g").replace(".", "p")
        sc.tl.leiden(adata, resolution=r, random_state=0, flavor="leidenalg", key_added=key)
        print(f"{key}: {adata.obs[key].nunique()} clusters")
    adata.write_h5ad(args.output)


if __name__ == "__main__":
    main()
