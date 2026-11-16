"""Pretrain the graph autoencoder on the training slices.

    python pretrain_gae.py --config configs/gae.yaml

Writes ``gae.pt`` (plus config, split and gene list) to ``output_dir``. Use the
same ``data`` block as the HoloGene config.
"""
import argparse

from hologene.gae_trainer import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None, help="e.g. cuda:0 (default) or cpu")
    args = parser.parse_args()
    run(args.config, device=args.device)
