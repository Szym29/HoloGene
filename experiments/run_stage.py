"""Run an experiment stage when its saved result is missing or out of date."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RECONSTRUCTION_METRICS = {"masked_MSE", "masked_Cosine", "Pearson", "masked_Spearman"}
BIOLOGY_METRICS = {"Pathway_F1", "CCC_Pearson", "CCC_Top_F1", "DEG_F1"}


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def newer_than_result(result: Path, inputs: list[Path]) -> bool:
    return any(path.exists() and path.stat().st_mtime_ns > result.stat().st_mtime_ns for path in inputs)


def config_is_current(config: dict, output: Path) -> bool:
    saved = output / "config.yaml"
    if not saved.is_file():
        return False
    try:
        return yaml.safe_load(saved.read_text()) == config
    except (OSError, yaml.YAMLError):
        return False


def run_training(stage: str, config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    output = project_path(config["output_dir"])
    if stage == "gae":
        result = output / "gae.pt"
        artifacts = [result, output / "split.json", output / "gene_names.txt",
                     output / "training_losses.npy"]
        inputs = [project_path(config["data"]["h5ad"])]
        script = "pretrain_gae.py"
    else:
        result = output / "prediction.h5ad"
        artifacts = [result, output / "hologene.pt", output / "split.json",
                     output / "training_history.json"]
        inputs = [project_path(config["data"]["h5ad"]),
                  project_path(config["gae"]["checkpoint"])]
        script = "train.py"
    missing_inputs = [str(path) for path in inputs if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing {stage} input: {', '.join(missing_inputs)}")
    current = (all(path.is_file() for path in artifacts)
               and config_is_current(config, output)
               and not newer_than_result(result, inputs)
               and all(path.stat().st_mtime_ns >= result.stat().st_mtime_ns
                       for path in artifacts))
    if current:
        print(f"[skip] {stage}: {output}", flush=True)
        return
    print(f"[run] {stage}: {output}", flush=True)
    subprocess.run([sys.executable, str(ROOT / script), "--config", str(config_path)],
                   cwd=ROOT, check=True)


def run_evaluation(prediction: Path, out_dir: Path, skip_biology: bool) -> None:
    if not prediction.is_file():
        raise FileNotFoundError(f"Missing prediction: {prediction}")
    metrics_path = out_dir / "metrics.json"
    required = RECONSTRUCTION_METRICS | (set() if skip_biology else BIOLOGY_METRICS)
    current = False
    partial_path = out_dir / "metrics.partial.json"
    if (metrics_path.is_file()
            and not newer_than_result(metrics_path, [prediction, partial_path])):
        try:
            metrics = json.loads(metrics_path.read_text())
            current = all(isinstance(metrics.get(key), (int, float))
                          and math.isfinite(metrics[key]) for key in required)
        except (OSError, json.JSONDecodeError):
            pass
    if current:
        print(f"[skip] eval: {out_dir}", flush=True)
        return
    print(f"[run] eval: {out_dir}", flush=True)
    command = [sys.executable, str(ROOT / "evaluate.py"), "--prediction", str(prediction),
               "--out-dir", str(out_dir)]
    if skip_biology:
        command.append("--skip-biology")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in ("gae", "train"):
        subparsers.add_parser(stage).add_argument("--config", required=True)
    evaluation = subparsers.add_parser("eval")
    evaluation.add_argument("--prediction", required=True)
    evaluation.add_argument("--out-dir", required=True)
    evaluation.add_argument("--skip-biology", action="store_true")
    args = parser.parse_args()
    if args.stage == "eval":
        run_evaluation(project_path(args.prediction), project_path(args.out_dir), args.skip_biology)
    else:
        run_training(args.stage, project_path(args.config))


if __name__ == "__main__":
    main()
