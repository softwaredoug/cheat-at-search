import pandas as pd

from cheat_at_search.strategy.strategy import SearchStrategy


def test_search_all_uses_search_batch_esci():
    from cheat_at_search import esci_data

    corpus = esci_data.corpus
    judgments = esci_data.judgments
    queries = (
        judgments[["query", "query_id"]]
        .drop_duplicates()
        .sample(n=20, random_state=123)
        .reset_index(drop=True)
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
