from unittest.mock import patch

import pandas as pd

from cheat_at_search import minimarco_data


@patch("cheat_at_search.minimarco_data.msmarco_data._qrels")
def test_qrels_samples_500_queries(qrels_mock):
    qrels = pd.DataFrame(
        {
            "query_id": list(range(600)),
            "doc_id": list(range(600)),
            "grade": [1] * 600,
            "query": [f"q{idx}" for idx in range(600)],
        }
    )
    qrels_mock.return_value = qrels

    sampled = minimarco_data._qrels()

    sampled_queries = (
        qrels[["query_id", "query"]]
        .drop_duplicates()
        .sample(n=500, random_state=42)
    )
    expected_ids = set(sampled_queries["query_id"].tolist())

    assert sampled["query_id"].nunique() == 500
    assert set(sampled["query_id"].tolist()) == expected_ids


@patch("cheat_at_search.minimarco_data.msmarco_data.download_msmarco")
@patch("cheat_at_search.minimarco_data.pd.read_csv")
@patch("cheat_at_search.minimarco_data._qrels")
def test_docs_builds_hard_negatives(qrels_mock, read_csv_mock, download_mock):
    passages = pd.DataFrame(
        {
            "doc_id": [1, 2, 3, 4, 5, 6],
            "description": [
                "alpha beta",
                "alpha gamma",
                "delta",
                "epsilon alpha",
                "beta",
                "zeta",
            ],
        }
    )
    qrels = pd.DataFrame(
        {
            "query_id": [10, 20],
            "doc_id": [1, 3],
            "grade": [1, 1],
            "query": ["alpha beta", "delta"],
        }
    )

    read_csv_mock.return_value = passages
    qrels_mock.return_value = qrels
    minimarco_data.__dict__.pop("judgments", None)
    minimarco_data.__dict__.pop("corpus", None)

    corpus = minimarco_data._docs()

    assert set(corpus["doc_id"].tolist()) == {1, 2, 3, 4, 5, 6}
    assert corpus["title"].tolist() == [""] * len(corpus)
    download_mock.assert_called_once()
