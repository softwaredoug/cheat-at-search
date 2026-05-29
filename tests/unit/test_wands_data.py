import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from cheat_at_search import wands_data


def test_ideal10_computes_top_10():
    products = pd.DataFrame({
        "doc_id": ["doc1", "doc2", "doc3"],
        "title": ["A", "B", "C"],
    })
    labeled_queries = pd.DataFrame({
        "query_id": ["q1", "q1", "q1"],
        "query": ["test", "test", "test"],
        "query_class": ["class1", "class1", "class1"],
        "doc_id": ["doc1", "doc2", "doc3"],
        "grade": [2, 1, 0],
    })

    result = wands_data._ideal10(products, labeled_queries)

    assert len(result) == 3
    assert "ideal_rank" in result.columns
    assert "ideal_grade" in result.columns
    top_result = result[result["ideal_rank"] == 1]
    assert len(top_result) == 1
    assert top_result.iloc[0]["ideal_grade"] == 2


def test_ideal10_limits_to_10():
    products = pd.DataFrame({
        "doc_id": [f"doc{i}" for i in range(15)],
        "title": [f"Product {i}" for i in range(15)],
    })
    labeled_queries = pd.DataFrame({
        "query_id": ["q1"] * 15,
        "query": ["test"] * 15,
        "query_class": ["class1"] * 15,
        "doc_id": [f"doc{i}" for i in range(15)],
        "grade": list(range(15, 0, -1)),
    })

    result = wands_data._ideal10(products, labeled_queries)

    assert len(result) == 10


def test_rel_attribute_counts_categories():
    query_products = pd.DataFrame({
        "query": ["q1", "q1", "q1"],
        "grade": [2, 2, 1],
        "category": ["A", "A", "B"],
    })

    result = wands_data.rel_attribute(query_products, grade=2, column="category")

    assert result.loc[("q1", "A")] == 2
    assert ("q1", "B") not in result.index or result.get(("q1", "B"), 0) == 0


def test_rel_attribute_default_grade():
    query_products = pd.DataFrame({
        "query": ["q1", "q1", "q1"],
        "grade": [2, 2, 1],
        "category": ["A", "A", "B"],
    })

    result = wands_data.rel_attribute(query_products, column="category")

    assert result.loc[("q1", "A")] == 2


@patch('cheat_at_search.wands_data.fetch_wands')
@patch('pandas.read_csv')
def test_corpus_loads_and_normalizes(mock_read_csv, mock_fetch_wands):
    mock_fetch_wands.return_value = MagicMock()
    mock_df = pd.DataFrame({
        "product_id": ["1", "2"],
        "product_name": ["A", "B"],
        "product_description": ["Desc A", None],
        "product_features": ["feat1|feat2", "feat3"],
        "category hierarchy": ["Cat/Sub", "Other/Sub2"],
    })
    mock_read_csv.return_value = mock_df

    result = wands_data._corpus()

    assert "doc_id" in result.columns
    assert "title" in result.columns
    assert "description" in result.columns
    assert "category" in result.columns
    assert "sub_category" in result.columns
    assert result.iloc[0]["description"] == "Desc A"
    assert result.iloc[1]["description"] == ""
    assert result.iloc[0]["category"] == "Cat"
    assert result.iloc[0]["sub_category"] == "Sub"


@patch('cheat_at_search.wands_data.fetch_wands')
@patch('pandas.read_csv')
def test_corpus_raises_when_file_missing(mock_read_csv, mock_fetch_wands):
    mock_read_csv.side_effect = FileNotFoundError("File not found")
    mock_fetch_wands.return_value = MagicMock()

    with pytest.raises(FileNotFoundError):
        wands_data._corpus()


@patch('cheat_at_search.wands_data.fetch_wands')
@patch('pandas.read_csv')
def test_queries_loads(mock_read_csv, mock_fetch_wands):
    mock_fetch_wands.return_value = MagicMock()
    mock_df = pd.DataFrame({
        "query_id": ["1", "2"],
        "query": ["test1", "test2"],
    })
    mock_read_csv.return_value = mock_df

    result = wands_data._queries()

    assert len(result) == 2
    assert "query_id" in result.columns


@patch('cheat_at_search.wands_data.fetch_wands')
@patch('pandas.read_csv')
def test_labels_loads_and_normalizes(mock_read_csv, mock_fetch_wands):
    mock_fetch_wands.return_value = MagicMock()
    mock_df = pd.DataFrame({
        "query_id": ["1", "1", "2"],
        "product_id": ["d1", "d2", "d3"],
        "label": ["Exact", "Partial", "Irrelevant"],
    })
    mock_read_csv.return_value = mock_df

    result = wands_data._labels()

    assert "doc_id" in result.columns
    assert "grade" in result.columns
    assert result[result["label"] == "Exact"].iloc[0]["grade"] == 2
    assert result[result["label"] == "Partial"].iloc[0]["grade"] == 1
    assert result[result["label"] == "Irrelevant"].iloc[0]["grade"] == 0
