from unittest.mock import patch

import pandas as pd

from cheat_at_search import esci_data


@patch("cheat_at_search.esci_data.pd.read_parquet")
@patch("cheat_at_search.esci_data.fetch_esci")
def test_judgments_use_locale_qualified_doc_ids(fetch_esci, read_parquet):
    read_parquet.return_value = pd.DataFrame(
        {
            "product_id": ["B001", "B001"],
            "product_locale": ["us", "uk"],
            "query_id": [1, 2],
            "query": ["chair", "chair"],
            "esci_label": ["E", "S"],
        }
    )

    judgments = esci_data._judgments()

    assert judgments["doc_id"].tolist() == ["B001_us", "B001_uk"]


@patch("cheat_at_search.esci_data.pd.read_parquet")
@patch("cheat_at_search.esci_data.fetch_esci")
def test_corpus_uses_locale_qualified_doc_ids(fetch_esci, read_parquet):
    read_parquet.return_value = pd.DataFrame(
        {
            "product_id": ["B001", "B001"],
            "product_locale": ["us", "uk"],
            "product_title": ["Chair", "Stuhl"],
            "product_description": ["A chair", "Ein Stuhl"],
        }
    )

    corpus = esci_data._docs()

    assert corpus["doc_id"].tolist() == ["B001_us", "B001_uk"]
