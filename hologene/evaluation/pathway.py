"""Pathway micro F1 via g:Profiler g:GOSt (paper: database version e114_eg62_p19_9dc92ac8)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

GPROFILER_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
ORGANISM = "mmusculus"
SOURCES = ("GO:BP", "REAC", "WP", "KEGG")
FDR_THRESHOLD = 0.05
MIN_QUERY_GENES = 3
MAX_TERM_SIZE = 1000
MIN_INTERSECTION = 15
TOP_TERMS_PER_LABEL = 20
MIN_GT_TERMS = 3


def build_queries(selected: pd.DataFrame, min_genes: int = MIN_QUERY_GENES
                  ) -> tuple[dict[str, list[str]], dict[str, dict[str, str]]]:
    """One query (sorted unique gene symbols) per (label, direction)."""
    queries: dict[str, list[str]] = {}
    mapping: dict[str, dict[str, str]] = {}
    for (label, direction), sub in selected.groupby(["label", "direction"], sort=True):
        genes = sorted({str(g) for g in sub["gene_symbol"].dropna().astype(str) if str(g)})
        if len(genes) < min_genes:
            continue
        name = f"{label}||{direction}"
        queries[name] = genes
        mapping[name] = {"label": str(label), "direction": str(direction)}
    return queries, mapping


def build_payload(queries: dict[str, list[str]], background: Iterable[str]) -> dict[str, Any]:
    """g:GOSt request body; ``background`` is the evaluated gene panel."""
    return {
        "organism": ORGANISM,
        "query": queries,
        "sources": list(SOURCES),
        "user_threshold": FDR_THRESHOLD,
        "significance_threshold_method": "fdr",
        "domain_scope": "custom_annotated",
        "background": sorted({str(g) for g in background if str(g)}),
        "no_evidences": False,
    }


def post_gprofiler(payload: dict[str, Any], timeout: int = 120, retries: int = 3) -> dict[str, Any]:
    """Send ``payload`` to the g:Profiler web API (online mode)."""
    import requests

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.post(GPROFILER_URL, json=payload, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            last_error = RuntimeError(f"g:Profiler status={response.status_code}: {response.text[:500]}")
        except Exception as exc:  # network errors are retried
            last_error = exc
        time.sleep(1.0 + attempt)
    raise last_error if last_error is not None else RuntimeError("g:Profiler request failed")


def find_cached_response(payload: dict[str, Any], cache_dir: str | Path) -> tuple[dict[str, Any], Path] | None:
    """Return ``(response, request_path)`` from a cache whose request equals ``payload``.

    ``cache_dir`` is searched for ``request.json`` files at depth 0-2 (e.g. a
    ``gprofiler/`` folder holding one sub-folder per query set, or its parent);
    the sibling ``response.json`` of the first exactly matching request is used.
    """
    cache_dir = Path(cache_dir)
    target = json.loads(json.dumps(payload))
    candidates = sorted({*cache_dir.glob("request.json"), *cache_dir.glob("*/request.json"),
                         *cache_dir.glob("*/*/request.json")})
    for request_path in candidates:
        response_path = request_path.with_name("response.json")
        if not response_path.is_file():
            continue
        if json.loads(request_path.read_text()) == target:
            return json.loads(response_path.read_text()), request_path
    return None


def run_gprofiler(payload: dict[str, Any], *, cache_dir: str | Path | None = None,
                  offline: bool = False) -> tuple[dict[str, Any], str]:
    """Get the g:GOSt response for ``payload`` from the cache or the web API.

    Returns ``(response, origin)`` where origin is the matching cached request
    path or ``"online"``.
    """
    if not payload["query"]:
        return {"result": [], "meta": {}, "status": "no_queries"}, "no_queries"
    if cache_dir is not None:
        hit = find_cached_response(payload, cache_dir)
        if hit is not None:
            return hit[0], str(hit[1])
    if offline:
        raise FileNotFoundError(
            f"Offline mode: no cached g:Profiler request in {cache_dir} matches the request built "
            "from this prediction file (it must be identical, including the background).")
    return post_gprofiler(payload), "online"


def enrichment_table(response: dict[str, Any], queries: dict[str, list[str]],
                     mapping: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Flatten a g:GOSt response into one row per (query, term)."""
    rows = []
    for item in response.get("result", []) or []:
        query = str(item.get("query"))
        mapped = mapping.get(query, {"label": query, "direction": ""})
        genes = queries.get(query, [])
        evidences = item.get("intersections") or []
        hits = [g for g, e in zip(genes, evidences) if isinstance(e, list) and len(e) > 0]
        rows.append({
            "label": mapped["label"],
            "direction": mapped["direction"],
            "source": item.get("source"),
            "native": item.get("native"),
            "name": item.get("name"),
            "adjusted_p_value": item.get("p_value"),
            "significant": item.get("significant"),
            "term_size": item.get("term_size"),
            "query_size": item.get("query_size"),
            "intersection_size": item.get("intersection_size"),
            "effective_domain_size": item.get("effective_domain_size"),
            "hit_genes": ",".join(hits),
        })
    return pd.DataFrame(rows)


def top_terms_per_label(table: pd.DataFrame) -> dict[str, set[tuple[str, str, str]]]:
    """Filtered top-20 ``(direction, source, term id)`` sets per label."""
    if table.empty:
        return {}
    frame = table.copy()
    for col in ("adjusted_p_value", "term_size", "intersection_size"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame[(frame["adjusted_p_value"] < FDR_THRESHOLD)
                  & (frame["term_size"] <= MAX_TERM_SIZE)
                  & (frame["intersection_size"] >= MIN_INTERSECTION)]
    frame = frame.sort_values(["label", "adjusted_p_value", "intersection_size", "term_size", "native"],
                              ascending=[True, True, False, True, True])
    frame = frame.groupby(frame["label"].astype(str), sort=False).head(TOP_TERMS_PER_LABEL)
    return {str(label): set(zip(g["direction"].astype(str), g["source"].astype(str), g["native"].astype(str)))
            for label, g in frame.groupby(frame["label"].astype(str))}


def pathway_f1(truth_terms: dict[str, set], pred_terms: dict[str, set]) -> dict[str, Any]:
    """Micro F1 over labels whose ground truth has >= 3 significant terms."""
    labels = sorted(label for label, terms in truth_terms.items() if len(terms) >= MIN_GT_TERMS)
    n_truth = sum(len(truth_terms[l]) for l in labels)
    n_pred = sum(len(pred_terms.get(l, set())) for l in labels)
    overlap = sum(len(truth_terms[l] & pred_terms.get(l, set())) for l in labels)
    return {
        "Pathway_F1": 2 * overlap / (n_truth + n_pred) if n_truth else None,
        "Pathway_GT_terms": n_truth,
        "Pathway_pred_terms": n_pred,
        "Pathway_overlap": overlap,
        "Pathway_eligible_labels": len(labels),
    }
