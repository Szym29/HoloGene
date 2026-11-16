"""Train HoloGene (latent synthesis stage) and predict the held-out slices.

    python train.py --config configs/hologene.yaml

Outputs (in ``output_dir`` of the config): ``prediction.h5ad`` (held-out
slices; X = measured expression, obsm['prediction'] = HoloGene prediction,
obsm['inr_embeddings'] = predicted latent), ``hologene.pt``, ``split.json``,
``config.yaml`` and ``training_history.json``.
"""
import argparse

from hologene.trainer import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None, help="e.g. cuda:0 (default) or cpu")
    args = parser.parse_args()
    run(args.config, device=args.device)
