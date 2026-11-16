#!/usr/bin/env python
"""Evaluate a HoloGene prediction file.

Example::

    python evaluate.py --prediction outputs/mosta/split_8to2/hologene_full/prediction.h5ad \\
                       --out-dir outputs/mosta/split_8to2/hologene_full/eval

Writes ``<out-dir>/metrics.json`` (plus DE tables and g:Profiler request/response
files when biological metrics are computed). From Python::

    from evaluate import evaluate_prediction
    metrics = evaluate_prediction("prediction.h5ad", "eval_out")

Pathway F1 queries the g:Profiler web API unless a matching cached response is
found in ``--gprofiler-cache``; ``--offline`` forbids web queries.
"""

from __future__ import annotations

import argparse
import json

from hologene.evaluation import evaluate_prediction


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prediction", required=True, help="Prediction h5ad (X = ground truth, obsm['prediction']).")
    parser.add_argument("--out-dir", required=True, help="Output folder for metrics.json and intermediate files.")
    parser.add_argument("--skip-biology", action="store_true", help="Only compute reconstruction metrics.")
    parser.add_argument("--gprofiler-cache", default=None,
                        help="Folder with cached g:Profiler request.json/response.json pairs.")
    parser.add_argument("--offline", action="store_true", help="Never query g:Profiler online.")
    parser.add_argument("--label-key", default="annotation", help="obs column with evaluation annotations.")
    parser.add_argument("--prediction-key", default="prediction", help="obsm key of the predicted expression.")
    args = parser.parse_args(argv)
    metrics = evaluate_prediction(args.prediction, args.out_dir, skip_biology=args.skip_biology,
                                  gprofiler_cache=args.gprofiler_cache, offline=args.offline,
                                  label_key=args.label_key, prediction_key=args.prediction_key)
    print(json.dumps(metrics, indent=2, default=str))
    return metrics


if __name__ == "__main__":
    main()
