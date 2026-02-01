import importlib

import pandas as pd
import pytest

from cheat_at_search.search import run_bm25, run_strategy
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
