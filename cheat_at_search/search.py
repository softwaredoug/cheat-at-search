from typing import Literal, Optional

import pandas as pd

from cheat_at_search.eval import (
    _max_ndcg_m_by_query,
    grade_answers,
    grade_results,
    pairwise_agreement,
    pointwise_agreement,
    reciprocal_rank,
)
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.data_dir import ensure_data_subdir
from cheat_at_search.strategy import BM25Search
from cheat_at_search.wands_data import corpus as wands_corpus
from cheat_at_search.wands_data import judgments as wands_judgments


logger = log_to_stdout(logger_name="search")


def run_strategy(
    strategy,
    judgments,
    queries: Optional[list[str]] = None,
    num_queries=None,
    seed=42,
    batch_size=100,
    show_progress=True,
    cache=True,
    eval_answer=None,
    include_agreement=False,
    include_pairwise_agreement=False,
):
    available_queries = judgments[["query", "query_id"]].drop_duplicates()

    if queries:
        available_queries = available_queries[available_queries["query"].isin(queries)]
        # Check any missing queries
        missing_queries = set(queries) - set(available_queries["query"])
        if missing_queries:
            raise ValueError(
                f"You asked to search over these queries, but they are missing from judgments: {missing_queries}"
            )
        judgments = judgments[judgments["query_id"].isin(available_queries["query_id"])]

    if seed is not None:
        available_queries = available_queries.sample(frac=1, random_state=seed)

    if num_queries:
        available_queries = available_queries.head(num_queries)
        judgments = judgments[judgments["query_id"].isin(available_queries["query_id"])]

    if eval_answer is not None:
        if "answer" not in judgments.columns:
            raise ValueError(
                "eval_answer was provided but judgments has no 'answer' column."
            )
        return _run_answer_path(
            strategy,
            judgments,
            available_queries,
            batch_size,
            show_progress,
            cache,
            eval_answer,
        )

    return _run_search_path(
        strategy,
        judgments,
        available_queries,
        batch_size,
        show_progress,
        cache,
        include_agreement,
        include_pairwise_agreement,
        cache_seed=seed,
    )


def _run_search_path(
    strategy,
    judgments,
    available_queries,
    batch_size,
    show_progress,
    cache,
    include_agreement,
    include_pairwise_agreement,
    cache_seed: int | None = None,
):
    max_grade = judgments["grade"].max()
    results = strategy.search_all(
        available_queries,
        batch_size=batch_size,
        show_progress=show_progress,
        cache=cache,
        cache_seed=cache_seed,
    )
    graded = grade_results(
        judgments,
        results,
        max_grade=max_grade,
        k=10,
    )
    max_ndcg_m = _max_ndcg_m_by_query(judgments, max_grade=max_grade, k=10)
    graded = graded.merge(max_ndcg_m, on=["query", "query_id"], how="left")

    if (include_agreement or include_pairwise_agreement) and "score" not in results.columns:
        raise ValueError(
            "An agreement metric was enabled but search results have no 'score' column."
        )

    if len(graded):
        idcg_m = graded["idcg_m"].iloc[0]
        dcgs = (
            graded.groupby(["query", "query_id"])["discounted_gain"]
            .sum()
            .sort_values(ascending=False)
            .rename("dcg")
        )
        ndcg_m = (dcgs / idcg_m).rename("ndcg_m")
        idcg_g = max_ndcg_m.set_index(["query", "query_id"])["idcg_g"]
        ndcg_g = dcgs.div(idcg_g)
        ndcg_g = ndcg_g.mask(idcg_g.eq(0), 0.0).rename("ndcg_g")
        mrrs = reciprocal_rank(graded, max_grade=max_grade)

        graded = graded.merge(dcgs, on=["query", "query_id"])
        graded = graded.merge(ndcg_g, on=["query", "query_id"])
        graded = graded.merge(ndcg_m, on=["query", "query_id"])
        graded = graded.merge(mrrs, on=["query", "query_id"])
        if include_agreement:
            agreements = pointwise_agreement(graded)
            graded = graded.merge(agreements, on=["query", "query_id"])
        if include_pairwise_agreement:
            pairwise_agreements = pairwise_agreement(graded)
            graded = graded.merge(pairwise_agreements, on=["query", "query_id"])
    else:
        graded["dcg"] = 0
        graded["ndcg_g"] = 0
        graded["ndcg_m"] = 0
        graded["mrr"] = 0
        if include_agreement:
            graded["agreement"] = pd.Series(dtype=float)
        if include_pairwise_agreement:
            graded["pairwise_agreement"] = pd.Series(dtype=float)

    return graded


def _run_answer_path(
    strategy,
    judgments,
    available_queries,
    batch_size,
    show_progress,
    cache,
    eval_answer_fn,
):
    results = strategy.answer_all(
        available_queries,
        batch_size=batch_size,
        show_progress=show_progress,
        cache=cache,
    )
    expected = judgments[["query_id", "query", "answer"]].drop_duplicates()
    expected = expected.rename(columns={"answer": "answer_expected"})
    results = results.rename(columns={"answer": "answer_actual"})
    merged = expected.merge(results, on=["query_id", "query"], how="left")
    merged["is_accurate"] = grade_answers(
        merged["answer_expected"],
        merged["answer_actual"],
        eval_answer_fn,
    )
    return merged[
        [
            "query_id",
            "query",
            "answer_expected",
            "answer_actual",
            "is_accurate",
        ]
    ]


def ndcgs(graded, ideal: Literal["max", "global"] = "max"):
    """Return per-query NDCG using the max-grade or judged-document ideal."""
    if ideal == "max":
        values = graded["ndcg_m"]
    elif ideal == "global":
        values = graded["ndcg_g"]
    else:
        raise ValueError("ideal must be either 'max' or 'global'")
    return values.groupby(graded["query"]).mean().sort_values(ascending=False)


def mrrs(graded):
    return graded.groupby("query")["mrr"].mean().sort_values(ascending=False)


def ndcg_delta(
    variant_graded,
    baseline_graded,
    ideal: Literal["max", "global"] = "max",
):
    variant_ndcgs = ndcgs(variant_graded, ideal=ideal)
    baseline_ndcgs = ndcgs(baseline_graded, ideal=ideal)
    delta = variant_ndcgs - baseline_ndcgs
    delta = delta[delta != 0]
    return delta.sort_values(ascending=False)


def run_bm25(corpus, judgments):
    try:
        bm25_results_path = ensure_data_subdir("bm25_results")
        graded_bm25 = pd.read_pickle(bm25_results_path / "graded_bm25.pkl")
        if "doc_id" not in graded_bm25.columns:
            raise ValueError("Cached BM25 results missing doc_id; recomputing.")
        if "mrr" not in graded_bm25.columns:
            raise ValueError("Cached BM25 results missing mrr; recomputing.")
        required_metrics = {"idcg_g", "idcg_m", "ndcg_g", "ndcg_m", "max_ndcg_m"}
        missing_metrics = required_metrics.difference(graded_bm25.columns)
        if missing_metrics:
            raise ValueError(
                f"Cached BM25 results missing {sorted(missing_metrics)}; recomputing."
            )
        if "ndcg" in graded_bm25.columns:
            raise ValueError("Cached BM25 results contain legacy ndcg; recomputing.")
        return graded_bm25
    except Exception:
        logger.warning("BM25 results not found, running BM25 search strategy.")
        bm25_results_path = ensure_data_subdir("bm25_results")
        bm25 = BM25Search(corpus)
        graded_bm25 = run_strategy(bm25, judgments)
        graded_bm25.to_pickle(bm25_results_path / "graded_bm25.pkl")
        return graded_bm25


graded_bm25 = run_bm25(wands_corpus, wands_judgments)


def vs_ideal(
    graded_results: pd.DataFrame,
    judgments: pd.DataFrame,
    corpus: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if "doc_id" not in graded_results.columns:
        raise ValueError("graded_results must include a doc_id column for vs_ideal.")
    actual = graded_results[graded_results["rank"] <= 10].copy()
    actual = actual.rename(columns={"rank": "rank_actual", "doc_id": "doc_id_actual"})
    if "title" in actual.columns:
        actual["title_actual"] = actual["title"]
    actual["rank_pos"] = actual["rank_actual"]

    ideal = judgments.copy()
    ideal = ideal.sort_values(
        ["query_id", "grade", "doc_id"], ascending=[True, False, True]
    )
    ideal = ideal.groupby("query_id", as_index=False, group_keys=False).head(10)
    ideal = ideal.rename(
        columns={
            "doc_id": "doc_id_ideal",
            "grade": "grade_ideal",
        }
    )
    ideal["rank_ideal"] = ideal.groupby("query_id").cumcount() + 1
    ideal["rank_pos"] = ideal["rank_ideal"]

    merged = actual.merge(
        ideal[
            [
                "query_id",
                "query",
                "doc_id_ideal",
                "grade_ideal",
                "rank_ideal",
                "rank_pos",
            ]
        ],
        on=["query_id", "rank_pos"],
        how="left",
        suffixes=("", "_ideal"),
    )
    merged["query"] = merged["query"].fillna(merged.get("query_ideal"))

    if corpus is not None:
        name_lookup = corpus[["doc_id", "title"]].rename(
            columns={"doc_id": "doc_id_ideal", "title": "title_ideal"}
        )
        merged = merged.merge(name_lookup, on="doc_id_ideal", how="left")
        if "title_actual" not in merged.columns:
            actual_lookup = corpus[["doc_id", "title"]].rename(
                columns={"doc_id": "doc_id_actual", "title": "title_actual"}
            )
            merged = merged.merge(actual_lookup, on="doc_id_actual", how="left")

    cols = [
        "query_id",
        "query",
        "doc_id_ideal",
        "grade_ideal",
        "rank_ideal",
        "title_ideal",
        "title_actual",
        "rank_actual",
        "doc_id_actual",
        "grade_actual",
        "dcg",
        "ndcg_m",
    ]
    merged = merged.rename(columns={"grade": "grade_actual"})
    if "title_ideal" not in merged.columns:
        merged["title_ideal"] = None
    if "title_actual" not in merged.columns:
        merged["title_actual"] = None
    return merged[cols]
