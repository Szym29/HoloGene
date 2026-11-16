"""End-to-end evaluation of one prediction file."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from anndata.io import read_elem

from . import pathway
from .ccc import ccc_metrics
from .de import one_vs_rest_de, select_up_degs
from .deg import deg_f1
from .reconstruction import reconstruction_metrics


@dataclass
class PredictionData:
    """Arrays needed for evaluation; all rows refer to the same held-out spots."""

    truth: np.ndarray
    prediction: np.ndarray
    coords: np.ndarray | None = None
    labels: np.ndarray | None = None
    gene_symbols: np.ndarray | None = None


def resolve_gene_symbols(var: pd.DataFrame, gene_symbol_key: str | None = None) -> np.ndarray:
    """Gene symbol per column.

    Uses ``var[gene_symbol_key]`` (or ``var['gene_symbol']``) when available,
    otherwise ``var_names``; if the var_names are mostly Ensembl IDs and a
    ``gene_name`` column exists, that column is used. Missing symbols fall back
    to the var_name.
    """
    names = np.asarray(var.index.astype(str))
    key = gene_symbol_key or ("gene_symbol" if "gene_symbol" in var.columns else None)
    if key is not None:
        values = var[key].astype(str).to_numpy()
    else:
        values = names
        if "gene_name" in var.columns and pd.Series(names).str.startswith("ENS").mean() > 0.5:
            values = var["gene_name"].astype(str).to_numpy()
    return np.asarray([v if v and v.lower() != "nan" else n for v, n in zip(values, names)], dtype=str)


def _dense(x: Any) -> np.ndarray:
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x)


def load_prediction(path: str | Path, *, truth_key: str = "X", prediction_key: str = "prediction",
                    spatial_key: str = "spatial", label_key: str = "annotation",
                    gene_symbol_key: str | None = None) -> PredictionData:
    """Read only the elements needed for evaluation from a prediction h5ad."""
    with h5py.File(path, "r") as f:
        truth = _dense(read_elem(f[truth_key]))
        prediction = _dense(read_elem(f["obsm"][prediction_key]))
        coords = np.asarray(read_elem(f["obsm"][spatial_key])) if spatial_key in f["obsm"] else None
        labels = None
        if label_key in f["obs"]:
            labels = pd.Series(read_elem(f["obs"][label_key])).astype(str).to_numpy()
        gene_symbols = resolve_gene_symbols(read_elem(f["var"]), gene_symbol_key)
    return PredictionData(truth, prediction, coords, labels, gene_symbols)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def evaluate_arrays(data: PredictionData, out_dir: str | Path | None = None, *, skip_biology: bool = False,
                    gprofiler_cache: str | Path | None = None, offline: bool = False,
                    chunk_size: int = 256, min_label_cells: int = 100,
                    metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute all metrics for in-memory arrays.

    Parameters
    ----------
    out_dir
        If given, ``metrics.json`` and the intermediate DE tables and
        g:Profiler request/response files are written here.
    skip_biology
        Only compute the four reconstruction metrics.
    gprofiler_cache
        Folder with cached ``request.json``/``response.json`` pairs; a cached
        response is used when its request equals the one built here.
    offline
        Never contact g:Profiler; fail if no matching cached response exists.
    metadata
        Extra entries copied to the front of the returned/written metrics.
    """
    out = Path(out_dir) if out_dir is not None else None
    metrics: dict[str, Any] = dict(metadata or {})
    metrics.update(reconstruction_metrics(data.truth, data.prediction, chunk_size=chunk_size))
    if skip_biology:
        if out is not None:
            _write_json(out / "metrics.json", metrics)
            (out / "metrics.partial.json").unlink(missing_ok=True)
        return metrics
    if data.labels is None or data.coords is None or data.gene_symbols is None:
        raise ValueError("Biological metrics need labels, coordinates and gene symbols")

    de = {}
    for name, matrix in (("GT", data.truth), ("prediction", data.prediction)):
        de[name] = one_vs_rest_de(matrix, data.labels, data.gene_symbols, min_label_cells=min_label_cells)
        if out is not None:
            (out / "de" / name).mkdir(parents=True, exist_ok=True)
            de[name].to_csv(out / "de" / name / "de_per_label.csv.gz", index=False)
    metrics.update(deg_f1(de["GT"], de["prediction"]))
    metrics.update(ccc_metrics(data.truth, data.prediction, data.labels, data.coords, data.gene_symbols,
                               min_label_cells=min_label_cells))
    if out is not None:  # retain partial results if the pathway request fails
        _write_json(out / "metrics.partial.json", metrics)

    terms, versions, origins = {}, {}, {}
    for name in ("GT", "prediction"):
        selected = select_up_degs(de[name])
        queries, mapping = pathway.build_queries(selected)
        payload = pathway.build_payload(queries, data.gene_symbols)
        response, origins[name] = pathway.run_gprofiler(payload, cache_dir=gprofiler_cache, offline=offline)
        table = pathway.enrichment_table(response, queries, mapping)
        terms[name] = pathway.top_terms_per_label(table)
        versions[name] = (response.get("meta") or {}).get("version")
        if out is not None:
            folder = out / "gprofiler" / name
            selected.to_csv(out / "de" / name / "selected_up_degs.csv", index=False)
            _write_json(folder / "request.json", payload)
            _write_json(folder / "response.json", response)
            table.to_csv(folder / "enrichment.csv", index=False)
    if versions["GT"] and versions["prediction"] and versions["GT"] != versions["prediction"]:
        raise RuntimeError(f"g:Profiler versions differ between GT ({versions['GT']}) and prediction "
                           f"({versions['prediction']}); pathway terms are not comparable")
    version = versions["GT"] or versions["prediction"]
    metrics.update(pathway.pathway_f1(terms["GT"], terms["prediction"]))
    metrics["gprofiler_version"] = version
    metrics["gprofiler_response_origin"] = origins

    if out is not None:
        _write_json(out / "metrics.json", metrics)
        (out / "metrics.partial.json").unlink(missing_ok=True)
    return metrics


def evaluate_prediction(prediction_path: str | Path, out_dir: str | Path | None = None, *,
                        skip_biology: bool = False, gprofiler_cache: str | Path | None = None,
                        offline: bool = False, label_key: str = "annotation",
                        prediction_key: str = "prediction", gene_symbol_key: str | None = None,
                        chunk_size: int = 256) -> dict[str, Any]:
    """Evaluate one prediction h5ad (X = ground truth, obsm['prediction'] = prediction)."""
    data = load_prediction(prediction_path, prediction_key=prediction_key, label_key=label_key,
                           gene_symbol_key=gene_symbol_key)
    return evaluate_arrays(data, out_dir, skip_biology=skip_biology, gprofiler_cache=gprofiler_cache,
                           offline=offline, chunk_size=chunk_size,
                           metadata={"prediction_file": str(prediction_path)})
