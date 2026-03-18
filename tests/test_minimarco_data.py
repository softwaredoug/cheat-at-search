from unittest.mock import patch

import pandas as pd

from cheat_at_search import minimarco_data


@patch("cheat_at_search.minimarco_data.msmarco_data.download_msmarco")
@patch("cheat_at_search.minimarco_data.pd.read_csv")
@patch("pandas.DataFrame.sample", autospec=True)
def test_docs_samples_one_million(sample_mock, read_csv_mock, download_mock):
    docs = pd.DataFrame(
        {
            "doc_id": [1, 2, 3],
            "description": ["a", "b", "c"],
        }
    )

    read_csv_mock.return_value = docs

    def _fake_sample(self, n=None, random_state=None):
        assert n == len(self)
        assert random_state == 42
        return self

    sample_mock.side_effect = _fake_sample

    result = minimarco_data._docs()

    assert result["title"].tolist() == ["", "", ""]
    download_mock.assert_called_once()


@patch("cheat_at_search.minimarco_data._docs")
@patch("cheat_at_search.minimarco_data.msmarco_data._qrels")
def test_qrels_filters_to_corpus_docs(qrels_mock, docs_mock):
    docs_mock.return_value = pd.DataFrame(
        {
            "doc_id": [1, 2],
            "description": ["a", "b"],
            "title": ["", ""],
        }
    )
    qrels_mock.return_value = pd.DataFrame(
        {
            "query_id": [10, 10, 11],
            "doc_id": [1, 2, 3],
            "grade": [1, 1, 1],
            "query": ["x", "x", "y"],
        }
    )

    minimarco_data.__dict__.pop("corpus", None)
    filtered = minimarco_data._qrels()

    assert filtered["doc_id"].tolist() == [1, 2]
