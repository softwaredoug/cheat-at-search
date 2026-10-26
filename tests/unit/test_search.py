from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from cheat_at_search.search import ndcg_delta, ndcgs, run_bm25, run_strategy, vs_ideal
from cheat_at_search.strategy.strategy import SearchStrategy


@patch("cheat_at_search.search.ensure_data_subdir")
def test_run_bm25(mock_ensure_data_subdir, tmp_path):
    corpus = pd.DataFrame(
        [
            {
                "doc_id": 1,
                "title": "red shoes",
                "description": "bright red running shoes",
            },
            {
                "doc_id": 2,
                "title": "blue jacket",
                "description": "waterproof blue jacket",
            },
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

    mock_ensure_data_subdir.side_effect = _ensure_data_subdir

    graded_bm25 = run_bm25(corpus, judgments)
    assert len(graded_bm25) > 0
    assert (tmp_path / "bm25_results" / "graded_bm25.pkl").exists()


def test_run_strategy_shuffles_queries_with_seed():
    judgments = pd.DataFrame(
        [
            {"query_id": 1, "query": "alpha", "answer": "A"},
            {"query_id": 2, "query": "bravo", "answer": "B"},
            {"query_id": 3, "query": "charlie", "answer": "C"},
            {"query_id": 4, "query": "delta", "answer": "D"},
            {"query_id": 5, "query": "echo", "answer": "E"},
        ]
    )

    class DummyAnswerStrategy(SearchStrategy):
        def __init__(self):
            super().__init__(pd.DataFrame())
            self.seen_queries = None

        def answer_all(self, queries, **kwargs):
            self.seen_queries = queries[["query", "query_id"]].reset_index(
                drop=True
            )
            return pd.DataFrame(
                {
                    "query_id": queries["query_id"].tolist(),
                    "query": queries["query"].tolist(),
                    "answer": ["ok"] * len(queries),
                }
            )

    seed = 123
    strategy = DummyAnswerStrategy()
    run_strategy(strategy, judgments, seed=seed, eval_answer=lambda *_: True)

    expected = (
        judgments[["query", "query_id"]]
        .drop_duplicates()
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(strategy.seen_queries, expected)


def test_run_strategy_passes_batch_size_to_search_all():
    judgments = pd.DataFrame(
        [
            {"query_id": 1, "query": "alpha", "doc_id": 10, "grade": 1},
            {"query_id": 2, "query": "bravo", "doc_id": 20, "grade": 1},
        ]
    )

    class DummySearchStrategy:
        def __init__(self):
            self.batch_size = None
            self.cache_salt = None

        def search_all(self, queries, batch_size=100, show_progress=True, cache=True, cache_salt=None):
            self.batch_size = batch_size
            self.cache_salt = cache_salt
            return pd.DataFrame(
                {
                    "query_id": queries["query_id"].tolist(),
                    "query": queries["query"].tolist(),
                    "doc_id": [10, 20],
                    "rank": [1, 1],
                }
            )

    strategy = DummySearchStrategy()
    run_strategy(strategy, judgments, batch_size=2, seed=123, cache_salt="trial-2")

    assert strategy.batch_size == 2
    assert strategy.cache_salt == "trial-2"


def test_run_strategy_includes_per_query_max_ndcg_m():
    judgments = pd.DataFrame(
        [
            {"query_id": 1, "query": "alpha", "doc_id": 10, "grade": 2},
            {"query_id": 1, "query": "alpha", "doc_id": 11, "grade": 1},
            {"query_id": 2, "query": "bravo", "doc_id": 20, "grade": 3},
        ]
    )

    class DummySearchStrategy:
        def search_all(self, queries, **_kwargs):
            doc_ids = {"alpha": [10, 11], "bravo": [20]}
            rows = []
            for _, query in queries.iterrows():
                for rank, doc_id in enumerate(doc_ids[query["query"]], start=1):
                    rows.append(
                        {
                            "query_id": query["query_id"],
                            "query": query["query"],
                            "doc_id": doc_id,
                            "rank": rank,
                        }
                    )
            return pd.DataFrame(rows)

    results = run_strategy(DummySearchStrategy(), judgments, seed=None, cache=False)
    per_query = results.groupby(["query_id", "query"])["max_ndcg_m"].first()
    idcg = 7 * sum(1 / rank for rank in range(1, 11))

    assert np.isclose(per_query.loc[(1, "alpha")], 3.5 / idcg)
    assert np.isclose(per_query.loc[(2, "bravo")], 7 / idcg)
    assert np.isclose(per_query.mean(), (3.5 + 7) / (2 * idcg))
    assert results.loc[results["query"] == "alpha", "max_ndcg_m"].nunique() == 1
    assert "max_ndcg" not in results.columns
    assert "max_ndcg_g" not in results.columns


def test_ndcgs_supports_global_judgment_ideal():
    judgments = pd.DataFrame(
        [
            {"query_id": 1, "query": "alpha", "doc_id": 10, "grade": 4},
            {"query_id": 1, "query": "alpha", "doc_id": 11, "grade": 3},
            {"query_id": 1, "query": "alpha", "doc_id": 12, "grade": 2},
            {"query_id": 1, "query": "alpha", "doc_id": 13, "grade": 1},
            {"query_id": 1, "query": "alpha", "doc_id": 14, "grade": 0},
        ]
    )

    class PartialResultsStrategy:
        def search_all(self, queries, **_kwargs):
            return pd.DataFrame(
                [
                    {"query_id": 1, "query": "alpha", "doc_id": 13, "rank": 1},
                    {"query_id": 1, "query": "alpha", "doc_id": 14, "rank": 2},
                ]
            )

    graded = run_strategy(PartialResultsStrategy(), judgments, seed=None, cache=False)
    dcg = 1.0
    idcg_g = 15 + 7 / 2 + 3 / 3 + 1 / 4
    idcg_m = 15 * sum(1 / rank for rank in range(1, 11))

    assert np.allclose(graded["idcg_g"], idcg_g)
    assert np.allclose(graded["idcg_m"], idcg_m)
    assert np.allclose(graded["max_ndcg_m"], idcg_g / idcg_m)
    assert np.allclose(graded["ndcg_g"], dcg / idcg_g)
    assert np.allclose(graded["ndcg_m"], dcg / idcg_m)
    assert "ndcg" not in graded.columns
    assert np.isclose(ndcgs(graded).loc["alpha"], dcg / idcg_m)
    assert np.isclose(
        ndcgs(graded, ideal="global").loc["alpha"], dcg / idcg_g
    )
    baseline = graded.copy()
    baseline["dcg"] = 0
    baseline["ndcg_g"] = 0
    assert np.isclose(
        ndcg_delta(graded, baseline, ideal="global").loc["alpha"], dcg / idcg_g
    )


def test_run_strategy_passes_batch_size_to_answer_all():
    judgments = pd.DataFrame(
        [
            {"query_id": 1, "query": "alpha", "answer": "A"},
            {"query_id": 2, "query": "bravo", "answer": "B"},
        ]
    )

    class DummyAnswerStrategy(SearchStrategy):
        def __init__(self):
            super().__init__(pd.DataFrame())
            self.batch_size = None

        def answer_all(self, queries, batch_size=100, **_kwargs):
            self.batch_size = batch_size
            return pd.DataFrame(
                {
                    "query_id": queries["query_id"].tolist(),
                    "query": queries["query"].tolist(),
                    "answer": ["ok"] * len(queries),
                }
            )

    strategy = DummyAnswerStrategy()
    run_strategy(strategy, judgments, batch_size=3, eval_answer=lambda *_: True)

    assert strategy.batch_size == 3


def test_vs_ideal_mocked():
    graded_results = pd.DataFrame(
        [
            {
                "query_id": 1,
                "query": "red shoes",
                "doc_id": 101,
                "rank": 1,
                "title": "Red Shoes A",
                "grade": 2,
                "dcg": 1.0,
                "ndcg_m": 0.5,
            },
            {
                "query_id": 1,
                "query": "red shoes",
                "doc_id": 102,
                "rank": 2,
                "title": "Red Shoes B",
                "grade": 0,
                "dcg": 1.0,
                "ndcg_m": 0.5,
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
            },
            {
                "query_id": 1,
                "query": "red shoes",
                "doc_id": 101,
                "grade": 2,
            },
        ]
    )

    comparison = vs_ideal(graded_results, judgments)
    assert list(comparison.columns) == [
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
    assert comparison["rank_ideal"].tolist() == [1, 2]
    assert comparison["doc_id_ideal"].tolist() == [102, 101]
    assert comparison["doc_id_actual"].tolist() == [101, 102]


def test_search_all_uses_search_batch():
    corpus = pd.DataFrame(
        [
            {"doc_id": 10, "title": "alpha"},
            {"doc_id": 20, "title": "beta"},
            {"doc_id": 30, "title": "gamma"},
            {"doc_id": 40, "title": "delta"},
        ]
    )
    queries = pd.DataFrame(
        [
            {"query_id": 1, "query": "alpha"},
            {"query_id": 2, "query": "beta"},
            {"query_id": 3, "query": "gamma"},
        ]
    )

    class DummyBatchStrategy(SearchStrategy):
        def __init__(self, corpus):
            super().__init__(corpus)
            self.search_calls = 0
            self.search_batch_calls = 0

        def _results_for_query(self, query, k):
            lookup = {
                "alpha": ([0, 1], [0.9, 0.1]),
                "beta": ([1, 2], [0.8, 0.2]),
                "gamma": ([2, 3], [0.7, 0.3]),
            }
            top_k, scores = lookup[query]
            return top_k[:k], scores[:k]

        def search(self, query, k=10):
            self.search_calls += 1
            return self._results_for_query(query, k)

        def search_batch(self, queries, k=10):
            self.search_batch_calls += 1
            all_top_k = []
            all_scores = []
            for query in queries:
                top_k, scores = self._results_for_query(query, k)
                all_top_k.append(top_k)
                all_scores.append(scores)
            return all_top_k, all_scores

    strategy = DummyBatchStrategy(corpus)
    results = strategy.search_all(queries, k=2, batch_size=2)

    assert strategy.search_calls == 0
    assert strategy.search_batch_calls == 2
    assert len(results) == len(queries) * 2

    for query, expected_scores in {
        "alpha": [0.9, 0.1],
        "beta": [0.8, 0.2],
        "gamma": [0.7, 0.3],
    }.items():
        subset = results[results["query"] == query]
        assert subset["rank"].tolist() == [1, 2]
        assert subset["score"].tolist() == expected_scores


def test_search_all_fills_empty_results():
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

    class EmptyResultStrategy(SearchStrategy):
        def __init__(self, corpus):
            super().__init__(corpus)

        def search(self, query, k=10):
            return [], []

    strategy = EmptyResultStrategy(corpus)

    results = strategy.search_all(queries, k=2)

    assert results["doc_id"].tolist() == [-1, -1]
    assert results["score"].tolist() == [0, 0]
    assert results["rank"].tolist() == [1, 1]


def test_search_all_includes_optional_resources():
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

    class ResourceStrategy(SearchStrategy):
        def search(self, query, k=10):
            if query == "alpha":
                return [0, 1], [0.9, 0.8], {"latency": 0.2, "tokens": 10}
            return [1], [0.7]

    results = ResourceStrategy(corpus).search_all(
        queries, show_progress=False
    )

    alpha = results[results["query"] == "alpha"]
    beta = results[results["query"] == "beta"]
    assert alpha["latency"].tolist() == [0.2, 0.2]
    assert alpha["tokens"].tolist() == [10, 10]
    assert beta["latency"].isna().all()
    assert beta["tokens"].isna().all()


def test_search_all_accepts_numpy_results():
    corpus = pd.DataFrame(
        [
            {"doc_id": 10, "title": "alpha"},
            {"doc_id": 20, "title": "beta"},
        ]
    )
    queries = pd.DataFrame(
        [
            {"query_id": 1, "query": "alpha"},
        ]
    )

    class NumpyResultStrategy(SearchStrategy):
        def __init__(self, corpus):
            super().__init__(corpus)

        def search(self, query, k=10):
            return np.array([0]), np.array([1.0])

    strategy = NumpyResultStrategy(corpus)

    results = strategy.search_all(queries, k=1)

    assert len(results) == 1


def test_run_strategy_raises_on_mismatched_search_lengths():
    corpus = pd.DataFrame(
        [
            {"doc_id": 10, "title": "alpha"},
            {"doc_id": 20, "title": "beta"},
        ]
    )
    judgments = pd.DataFrame(
        [
            {"query_id": 1, "query": "alpha", "doc_id": 10, "grade": 2},
            {"query_id": 1, "query": "alpha", "doc_id": 20, "grade": 0},
        ]
    )

    class BadLengthStrategy(SearchStrategy):
        def __init__(self, corpus):
            super().__init__(corpus)

        def search(self, query, k=10):
            return [0, 1], [0.9]

    strategy = BadLengthStrategy(corpus)

    with pytest.raises(ValueError, match="Length of values"):
        run_strategy(strategy, judgments, num_queries=1, seed=123)


def test_run_strategy_raises_for_strategy_missing_search():
    corpus = pd.DataFrame(
        [
            {"doc_id": 10, "title": "alpha"},
            {"doc_id": 20, "title": "beta"},
        ]
    )
    judgments = pd.DataFrame(
        [
            {"query_id": 1, "query": "alpha", "doc_id": 10, "grade": 2},
        ]
    )

    class BadStrategy(SearchStrategy):
        def __init__(self, corpus):
            super().__init__(corpus)

    strategy = BadStrategy(corpus)

    with pytest.raises(NotImplementedError, match="Subclasses should implement"):
        run_strategy(strategy, judgments, num_queries=1, seed=123)


def test_run_strategy_raises_on_string_doc_ids():
    corpus = pd.DataFrame(
        [
            {"doc_id": 10, "title": "alpha"},
            {"doc_id": 20, "title": "beta"},
        ]
    )
    judgments = pd.DataFrame(
        [
            {"query_id": 1, "query": "alpha", "doc_id": 10, "grade": 2},
            {"query_id": 1, "query": "alpha", "doc_id": 20, "grade": 0},
        ]
    )

    class StringIdStrategy(SearchStrategy):
        def __init__(self, corpus):
            super().__init__(corpus)

        def search(self, query, k=10):
            return ["0", "1"], [0.9, 0.1]

    strategy = StringIdStrategy(corpus)

    with pytest.raises(TypeError, match="non-integer key"):
        run_strategy(strategy, judgments, num_queries=1, seed=123)


def test_run_strategy_raises_on_batch_length_mismatch():
    corpus = pd.DataFrame(
        [
            {"doc_id": 10, "title": "alpha"},
            {"doc_id": 20, "title": "beta"},
        ]
    )
    judgments = pd.DataFrame(
        [
            {"query_id": 1, "query": "alpha", "doc_id": 10, "grade": 2},
            {"query_id": 2, "query": "beta", "doc_id": 20, "grade": 1},
        ]
    )

    class BatchLengthMismatchStrategy(SearchStrategy):
        def __init__(self, corpus):
            super().__init__(corpus)

        def search_batch(self, queries, k=10):
            return [[0, 1]], [[0.9, 0.1], [0.8, 0.2]]

    strategy = BatchLengthMismatchStrategy(corpus)

    with pytest.raises(ValueError, match="search_batch must return top_k"):
        run_strategy(strategy, judgments, num_queries=2, seed=123)


def test_run_strategy_raises_on_batch_item_length_mismatch():
    corpus = pd.DataFrame(
        [
            {"doc_id": 10, "title": "alpha"},
            {"doc_id": 20, "title": "beta"},
        ]
    )
    judgments = pd.DataFrame(
        [
            {"query_id": 1, "query": "alpha", "doc_id": 10, "grade": 2},
            {"query_id": 2, "query": "beta", "doc_id": 20, "grade": 1},
        ]
    )

    class BatchItemMismatchStrategy(SearchStrategy):
        def __init__(self, corpus):
            super().__init__(corpus)

        def search_batch(self, queries, k=10):
            return [[0, 1], [0]], [[0.9, 0.1], [0.8, 0.2]]

    strategy = BatchItemMismatchStrategy(corpus)

    with pytest.raises(ValueError, match="Length of values"):
        run_strategy(strategy, judgments, num_queries=2, seed=123)


def test_run_strategy_raises_on_batch_string_doc_ids():
    corpus = pd.DataFrame(
        [
            {"doc_id": 10, "title": "alpha"},
            {"doc_id": 20, "title": "beta"},
        ]
    )
    judgments = pd.DataFrame(
        [
            {"query_id": 1, "query": "alpha", "doc_id": 10, "grade": 2},
            {"query_id": 2, "query": "beta", "doc_id": 20, "grade": 1},
        ]
    )

    class BatchStringIdStrategy(SearchStrategy):
        def __init__(self, corpus):
            super().__init__(corpus)

        def search_batch(self, queries, k=10):
            return [["0", "1"], ["1"]], [[0.9, 0.1], [0.8]]

    strategy = BatchStringIdStrategy(corpus)

    with pytest.raises(TypeError, match="non-integer key"):
        run_strategy(strategy, judgments, num_queries=2, seed=123)


def test_search_all_batched_fills_empty_results():
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

    class EmptyBatchStrategy(SearchStrategy):
        def __init__(self, corpus):
            super().__init__(corpus)

        def search_batch(self, queries, k=10):
            return [[], [0]], [[], [1.0]]

    strategy = EmptyBatchStrategy(corpus)

    results = strategy.search_all(queries, k=2)

    assert results["doc_id"].tolist() == [-1, 10]
    assert results["score"].tolist() == [0, 1.0]
    assert results["rank"].tolist() == [1, 1]


def test_run_strategy_empty_results_zero_metrics():
    corpus = pd.DataFrame(
        [
            {"doc_id": 10, "title": "alpha", "description": "alpha"},
        ]
    )
    judgments = pd.DataFrame(
        [
            {"query_id": 1, "query": "alpha", "doc_id": 10, "grade": 2},
        ]
    )

    class EmptyResultStrategy(SearchStrategy):
        def __init__(self, corpus):
            super().__init__(corpus)

        def search(self, query, k=10):
            return [], []

    strategy = EmptyResultStrategy(corpus)

    graded = run_strategy(strategy, judgments, seed=None)

    assert graded["ndcg_m"].tolist() == [0]
    assert graded["mrr"].tolist() == [0]


def test_run_strategy_reports_pointwise_agreement_with_tolerance_and_ignores_empty_placeholder():
    corpus = pd.DataFrame(
        [
            {"doc_id": 10, "title": "alpha"},
            {"doc_id": 20, "title": "beta"},
            {"doc_id": 30, "title": "gamma"},
            {"doc_id": 40, "title": "delta"},
        ]
    )
    judgments = pd.DataFrame(
        [
            {"query_id": 1, "query": "alpha", "doc_id": 10, "grade": 2},
            {"query_id": 1, "query": "alpha", "doc_id": 20, "grade": 1},
            {"query_id": 1, "query": "alpha", "doc_id": 30, "grade": 0},
            {"query_id": 2, "query": "beta", "doc_id": 10, "grade": 2},
        ]
    )

    class PointwiseStrategy(SearchStrategy):
        def search(self, query, k=10):
            if query == "alpha":
                # doc_id 40 is unjudged and is treated as grade 0.
                return [0, 1, 2, 3], [2 + 1e-7, 0.8, 0.2, 0]
            return [], []

    graded = run_strategy(
        PointwiseStrategy(corpus),
        judgments,
        seed=None,
        include_agreement=True,
    )
    agreements = graded.groupby("query")["agreement"].first().to_dict()

    assert agreements["alpha"] == pytest.approx(0.5)
    assert agreements["beta"] == 0
    assert graded.loc[graded["doc_id"] == -1, "agreement"].tolist() == [0]


def test_run_strategy_does_not_report_agreement_unless_enabled():
    corpus = pd.DataFrame([{"doc_id": 10, "title": "alpha"}])
    judgments = pd.DataFrame(
        [{"query_id": 1, "query": "alpha", "doc_id": 10, "grade": 2}]
    )

    class PointwiseStrategy(SearchStrategy):
        def search(self, query, k=10):
            return [0], [2]

    graded = run_strategy(PointwiseStrategy(corpus), judgments, seed=None)

    assert "agreement" not in graded.columns


def test_run_strategy_reports_pairwise_agreement_over_top_ten():
    corpus = pd.DataFrame(
        [
            {"doc_id": doc_id, "title": f"doc {doc_id}"}
            for doc_id in range(12)
        ]
    )
    judgments = pd.DataFrame(
        [
            {"query_id": 1, "query": "ordered", "doc_id": doc_id, "grade": grade}
            for doc_id, grade in enumerate([3, 2, 0, 1] + [0] * 8 + [10, 10])
        ]
        + [
            {"query_id": 2, "query": "empty", "doc_id": 0, "grade": 1},
        ]
    )

    class PairwiseStrategy(SearchStrategy):
        def search(self, query, k=10):
            if query == "ordered":
                # The final two documents rank below the top 10 and must not count.
                return list(range(12)), list(range(12, 0, -1))
            return [], []

    graded = run_strategy(
        PairwiseStrategy(corpus),
        judgments,
        seed=None,
        include_pairwise_agreement=True,
    )
    metrics = graded.groupby("query")["pairwise_agreement"].first().to_dict()

    # Of the 45 pairs in the top 10, only doc 2 vs doc 3 is discordant.
    assert metrics["ordered"] == pytest.approx(44 / 45)
    assert metrics["empty"] == 0
