from cheat_at_search.strategy import BM25Search
import pytest
from cheat_at_search.search import run_strategy
import importlib


@pytest.mark.parametrize("data_module", ["msmarco_data", "esci_data", "wands_data"])
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
