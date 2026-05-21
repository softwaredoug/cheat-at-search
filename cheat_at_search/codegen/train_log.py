from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone

import pandas as pd


def slugify_query(query: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", query.strip().lower()).strip("-")
    if not normalized:
        normalized = "query"
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:8]
    return f"{normalized[:40]}-{digest}"


def write_training_logs(
    code: str,
    ndcg_deltas: dict[str, float],
    results_df: pd.DataFrame,
    code_dir: str,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    training_root = os.path.join(code_dir, "training", timestamp)
    os.makedirs(training_root, exist_ok=True)

    reranker_path = os.path.join(training_root, "reranker.py")
    with open(reranker_path, "w") as handle:
        handle.write(code)

    if "doc_id" not in results_df.columns and "product_id" in results_df.columns:
        results_df = results_df.copy()
        results_df["doc_id"] = results_df["product_id"]
    for col in ["query", "rank", "doc_id", "title", "description"]:
        if col not in results_df.columns:
            results_df[col] = ""

    query_paths = {}
    for query in ndcg_deltas.keys():
        query_slug = slugify_query(query)
        query_paths[query] = query_slug

        query_dir = os.path.join(training_root, query_slug)
        os.makedirs(query_dir, exist_ok=True)

        query_results = results_df[results_df["query"] == query]
        query_results = query_results[
            ["query", "rank", "doc_id", "title", "description"]
        ]
        results_path = os.path.join(query_dir, "results.csv")
        query_results.to_csv(results_path, index=False)

    queries_rows = []
    for query, delta in ndcg_deltas.items():
        queries_rows.append({
            "query": query,
            "ndcg_delta": delta,
            "query_path": query_paths[query],
        })
    queries_df = pd.DataFrame(
        queries_rows,
        columns=["query", "ndcg_delta", "query_path"],
    )
    queries_path = os.path.join(training_root, "queries.csv")
    queries_df.to_csv(queries_path, index=False)

    return os.path.relpath(training_root, code_dir)
