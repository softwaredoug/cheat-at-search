from unittest.mock import patch

import pandas as pd

from cheat_at_search.strategy.strategy import SearchStrategy


def test_search_all_uses_search_batch_for_mocked_queries():
    corpus = pd.DataFrame(
        [
            {"doc_id": idx, "title": f"document {idx}"}
            for idx in range(10)
        ]
    )
    queries = pd.DataFrame(
        [
            {"query_id": idx, "query": f"query {idx}"}
            for idx in range(20)
        ]
    )

    class DummyBatchStrategy(SearchStrategy):
        def __init__(self, corpus):
            super().__init__(corpus)
            self.search_calls = 0
            self.search_batch_calls = 0

        def search(self, query, k=10):
            self.search_calls += 1
            raise AssertionError("search should not be called in batch mode")

        def _batch_results(self, k):
            top_k = list(range(k))
            scores = [1.0 / (idx + 1) for idx in range(k)]
            return top_k, scores

        def search_batch(self, queries, k=10):
            self.search_batch_calls += 1
            all_top_k = []
            all_scores = []
            for _query in queries:
                top_k, scores = self._batch_results(k)
                all_top_k.append(top_k)
                all_scores.append(scores)
            return all_top_k, all_scores

    strategy = DummyBatchStrategy(corpus)
    results = strategy.search_all(queries, k=5, batch_size=7)

    assert strategy.search_calls == 0
    assert strategy.search_batch_calls == 3
    assert len(results) == len(queries) * 5
    for query in queries["query"].tolist():
        subset = results[results["query"] == query]
        assert subset["rank"].tolist() == [1, 2, 3, 4, 5]
        assert subset["score"].tolist() == [1.0, 0.5, 0.3333333333333333, 0.25, 0.2]


def test_search_all_includes_resources_from_search_batch():
    corpus = pd.DataFrame(
        [
            {"doc_id": 10, "title": "alpha"},
            {"doc_id": 20, "title": "beta"},
        ]
    )
    queries = pd.DataFrame(
        [
            {"query_id": 1, "query": "alpha"},
            {"query_id": 2, "query": "beta"},
        ]
    )

    class ResourceBatchStrategy(SearchStrategy):
        def search(self, query, k=10):
            raise AssertionError("search should not be called in batch mode")

        def search_batch(self, queries, k=10):
            return [[0, 1], [1]], [[0.9, 0.8], [0.7]], [
                {"latency": 0.2, "tokens": 10},
                {"latency": 0.3, "tokens": 12},
            ]

    results = ResourceBatchStrategy(corpus).search_all(queries, show_progress=False)

    alpha = results[results["query"] == "alpha"]
    beta = results[results["query"] == "beta"]
    assert alpha["latency"].tolist() == [0.2, 0.2]
    assert alpha["tokens"].tolist() == [10, 10]
    assert beta["latency"].tolist() == [0.3]
    assert beta["tokens"].tolist() == [12]


@patch("cheat_at_search.strategy.strategy.ensure_data_subdir")
def test_search_all_cache_is_separated_by_cache_seed(
    mock_ensure_data_subdir, tmp_path
):
    mock_ensure_data_subdir.return_value = tmp_path
    corpus = pd.DataFrame(
        [
            {"doc_id": 10, "title": "first"},
            {"doc_id": 20, "title": "second"},
        ]
    )
    queries = pd.DataFrame([{"query_id": 1, "query": "alpha"}])

    class CachedStrategy(SearchStrategy):
        cache_key = "cache-seed-test"

        def __init__(self, corpus):
            super().__init__(corpus)
            self.search_calls = 0

        def search(self, query, k=10):
            self.search_calls += 1
            return [self.search_calls - 1], [1.0]

    strategy = CachedStrategy(corpus)

    seed_42 = strategy.search_all(queries, k=1, cache_seed=42)
    cached_seed_42 = strategy.search_all(queries, k=1, cache_seed=42)
    seed_43 = strategy.search_all(queries, k=1, cache_seed=43)

    assert strategy.search_calls == 2
    assert seed_42["doc_id"].tolist() == [10]
    assert cached_seed_42["doc_id"].tolist() == [10]
    assert seed_43["doc_id"].tolist() == [20]
