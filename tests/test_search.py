import importlib

import pandas as pd
import pytest

from cheat_at_search.search import graded_bm25, run_bm25, run_strategy, vs_ideal
from cheat_at_search.strategy import BM25Search


@pytest.mark.parametrize("data_module", ["msmarco_data", "esci_data", "wands_data", "tmdb_data"])
def test_bm25_search(data_module):
    """
    Run BM25 search strategy on the specified dataset module.
    """
    num_queries = 10
    module = importlib.import_module(f"cheat_at_search.{data_module}")
    corpus = getattr(module, "corpus")
    judgments = getattr(module, "judgments")
    strategy = BM25Search(corpus)
    graded_results = run_strategy(strategy, judgments, num_queries=num_queries)
    return graded_results


def test_run_bm25(tmp_path, monkeypatch):
    corpus = pd.DataFrame(
        [
            {"doc_id": 1, "title": "red shoes", "description": "bright red running shoes"},
            {"doc_id": 2, "title": "blue jacket", "description": "waterproof blue jacket"},
        ]
    )
    judgments = pd.DataFrame(
        [
            {"query_id": 100, "query": "red shoes", "doc_id": 1, "grade": 2},
            {"query_id": 100, "query": "red shoes", "doc_id": 2, "grade": 0},
        ]
    )

    def _ensure_data_subdir(subdir: str):
        subdir_path = tmp_path / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
        return subdir_path

    import cheat_at_search.search as search_module
    monkeypatch.setattr(search_module, "ensure_data_subdir", _ensure_data_subdir)

    graded_bm25 = run_bm25(corpus, judgments)
    assert len(graded_bm25) > 0
    assert (tmp_path / "bm25_results" / "graded_bm25.pkl").exists()


def test_vs_ideal_mocked():
    graded_results = pd.DataFrame(
        [
            {
                "query_id": 1,
                "query": "red shoes",
                "doc_id": 101,
                "rank": 1,
                "product_name": "Red Shoes A",
                "grade": 2,
                "dcg": 1.0,
                "ndcg": 0.5,
            },
            {
                "query_id": 1,
                "query": "red shoes",
                "doc_id": 102,
                "rank": 2,
                "product_name": "Red Shoes B",
                "grade": 0,
                "dcg": 1.0,
                "ndcg": 0.5,
            },
        ]
    )
    judgments = pd.DataFrame(
        [
            {
                "query_id": 1,
                "query": "red shoes",
                "doc_id": 102,
                "grade": 3,
                "label": "E",
            },
            {
                "query_id": 1,
                "query": "red shoes",
                "doc_id": 101,
                "grade": 2,
                "label": "S",
            },
        ]
    )

    comparison = vs_ideal(graded_results, judgments)
    assert list(comparison.columns) == [
        "query_id",
        "query",
        "product_id_ideal",
        "label_ideal",
        "grade_ideal",
        "rank_ideal",
        "product_name_actual",
        "rank_actual",
        "product_id_actual",
        "product_name_actual",
        "grade_actual",
        "dcg",
        "ndcg",
    ]
    assert comparison["rank_ideal"].tolist() == [1, 2]
    assert comparison["product_id_ideal"].tolist() == [102, 101]
    assert comparison["product_id_actual"].tolist() == [101, 102]


def test_vs_ideal_wands():
    from cheat_at_search import wands_data

    corpus = wands_data.corpus
    judgments = wands_data.judgments
    strategy = BM25Search(corpus)
    graded_results = run_strategy(strategy, judgments, num_queries=2, seed=123)

    comparison = vs_ideal(graded_results, judgments)
    assert list(comparison.columns) == [
        "query_id",
        "query",
        "product_id_ideal",
        "label_ideal",
        "grade_ideal",
        "rank_ideal",
        "product_name_actual",
        "rank_actual",
        "product_id_actual",
        "product_name_actual",
        "grade_actual",
        "dcg",
        "ndcg",
    ]
    assert len(comparison) > 0
    assert comparison["rank_actual"].max() <= 10
    assert comparison["rank_ideal"].max() <= 10


def test_graded_bm25_cached():
    assert isinstance(graded_bm25, pd.DataFrame)
    assert len(graded_bm25) > 0
    assert "dcg" in graded_bm25.columns
    assert "ndcg" in graded_bm25.columns
