import base64
import pandas as pd
from unittest.mock import patch, MagicMock
from cheat_at_search import bc_plus_data


def test_derive_key_length():
    key = bc_plus_data._derive_key("password", 32)
    assert len(key) == 32


def test_derive_key_deterministic():
    key1 = bc_plus_data._derive_key("password", 32)
    key2 = bc_plus_data._derive_key("password", 32)
    assert key1 == key2


def test_derive_key_different_passwords():
    key1 = bc_plus_data._derive_key("password1", 32)
    key2 = bc_plus_data._derive_key("password2", 32)
    assert key1 != key2


def test_decrypt_roundtrip():
    password = "test_password"
    plaintext = "Hello, World!"
    key = bc_plus_data._derive_key(password, len(plaintext.encode()))
    encrypted = bytes(a ^ b for a, b in zip(plaintext.encode(), key))
    ciphertext_b64 = base64.b64encode(encrypted).decode()

    result = bc_plus_data._decrypt_string(ciphertext_b64, password)
    assert result == plaintext


def test_transform_decrypt_string():
    password = "test"
    plaintext = "hello"
    key = bc_plus_data._derive_key(password, len(plaintext.encode()))
    encrypted = bytes(a ^ b for a, b in zip(plaintext.encode(), key))
    ciphertext_b64 = base64.b64encode(encrypted).decode()

    result = bc_plus_data._transform_decrypt(ciphertext_b64, password, set())
    assert result == plaintext


def test_transform_decrypt_list():
    password = "test"
    plaintext = "hello"
    key = bc_plus_data._derive_key(password, len(plaintext.encode()))
    encrypted = bytes(a ^ b for a, b in zip(plaintext.encode(), key))
    ciphertext_b64 = base64.b64encode(encrypted).decode()

    result = bc_plus_data._transform_decrypt([ciphertext_b64], password, set())
    assert result == [plaintext]


def test_transform_decrypt_dict():
    password = "test"
    plaintext = "hello"
    key = bc_plus_data._derive_key(password, len(plaintext.encode()))
    encrypted = bytes(a ^ b for a, b in zip(plaintext.encode(), key))
    ciphertext_b64 = base64.b64encode(encrypted).decode()

    result = bc_plus_data._transform_decrypt({"key": ciphertext_b64}, password, set())
    assert result == {"key": plaintext}


def test_transform_decrypt_dict_skips_keys():
    password = "test"
    result = bc_plus_data._transform_decrypt({"skip": "value"}, password, {"skip"})
    assert result == {"skip": "value"}


def test_transform_decrypt_non_string():
    result = bc_plus_data._transform_decrypt(42, "password", set())
    assert result == 42


def test_doc_id_from_dict():
    assert bc_plus_data._doc_id_from_entry({"docid": "123"}) == "123"
    assert bc_plus_data._doc_id_from_entry({"doc_id": "456"}) == "456"


def test_doc_id_from_string():
    assert bc_plus_data._doc_id_from_entry("abc") == "abc"


def test_doc_id_from_none():
    assert bc_plus_data._doc_id_from_entry(None) is None


def test_build_judgments_basic():
    query_df = pd.DataFrame({
        "query_id": ["q1"],
        "query": ["test query"],
        "answer": ["answer"],
        "negative_docs": [[{"docid": "d1"}]],
        "evidence_docs": [[{"docid": "d2"}]],
        "gold_docs": [[{"docid": "d3"}]],
    })

    result = bc_plus_data._build_judgments(query_df)

    assert len(result) == 3
    grades = set(result["grade"].tolist())
    assert grades == {0, 1, 2}


def test_build_judgments_deduplicates_higher_grade():
    query_df = pd.DataFrame({
        "query_id": ["q1"],
        "query": ["test query"],
        "answer": ["answer"],
        "negative_docs": [[{"docid": "d1"}]],
        "evidence_docs": [[{"docid": "d1"}]],
        "gold_docs": [[]],
    })

    result = bc_plus_data._build_judgments(query_df)

    assert len(result) == 1
    assert result.iloc[0]["grade"] == 1


def test_build_judgments_handles_none():
    query_df = pd.DataFrame({
        "query_id": ["q1"],
        "query": ["test query"],
        "answer": [None],
        "negative_docs": [None],
        "evidence_docs": [None],
        "gold_docs": [None],
    })

    result = bc_plus_data._build_judgments(query_df)

    assert len(result) == 0


def test_build_judgments_handles_float_nan():
    query_df = pd.DataFrame({
        "query_id": ["q1"],
        "query": ["test query"],
        "answer": [float("nan")],
        "negative_docs": [float("nan")],
        "evidence_docs": [float("nan")],
        "gold_docs": [float("nan")],
    })

    result = bc_plus_data._build_judgments(query_df)

    assert len(result) == 0


def test_pick_split_dataframe_returns_none():
    df = pd.DataFrame({"a": [1]})
    assert bc_plus_data._pick_split(df) is None


def test_pick_split_dict_returns_preferred():
    ds = {"train": [1, 2], "test": [3, 4]}
    assert bc_plus_data._pick_split(ds, preferred="train") == "train"


def test_pick_split_dict_fallback_to_test():
    ds = {"test": [1, 2]}
    assert bc_plus_data._pick_split(ds, preferred="train") == "test"


def test_pick_split_dict_fallback_to_first_key():
    ds = {"val": [1, 2]}
    assert bc_plus_data._pick_split(ds, preferred="train") == "val"


@patch('cheat_at_search.bc_plus_data._load_dataset')
def test_load_corpus_with_doc_id(mock_load_dataset):
    mock_ds = MagicMock()
    mock_ds.to_pandas.return_value = pd.DataFrame({
        "doc_id": ["1", "2"],
        "title": ["A", "B"],
        "description": ["Desc A", "Desc B"],
    })
    mock_load_dataset.return_value = mock_ds

    result = bc_plus_data._load_corpus()

    assert "doc_id" in result.columns
    assert len(result) == 2


@patch('cheat_at_search.bc_plus_data._load_dataset')
def test_load_corpus_adds_doc_id_from_docid(mock_load_dataset):
    mock_ds = MagicMock()
    mock_ds.to_pandas.return_value = pd.DataFrame({
        "docid": ["1", "2"],
        "title": ["A", "B"],
    })
    mock_load_dataset.return_value = mock_ds

    result = bc_plus_data._load_corpus()

    assert "doc_id" in result.columns
    assert result.iloc[0]["doc_id"] == "1"


@patch('cheat_at_search.bc_plus_data._load_dataset')
def test_load_corpus_adds_doc_id_from_id(mock_load_dataset):
    mock_ds = MagicMock()
    mock_ds.to_pandas.return_value = pd.DataFrame({
        "id": ["1", "2"],
        "title": ["A", "B"],
    })
    mock_load_dataset.return_value = mock_ds

    result = bc_plus_data._load_corpus()

    assert "doc_id" in result.columns
    assert result.iloc[0]["doc_id"] == "1"


@patch('cheat_at_search.bc_plus_data._load_dataset')
def test_load_corpus_adds_description_from_text(mock_load_dataset):
    mock_ds = MagicMock()
    mock_ds.to_pandas.return_value = pd.DataFrame({
        "doc_id": ["1"],
        "text": ["Description"],
    })
    mock_load_dataset.return_value = mock_ds

    result = bc_plus_data._load_corpus()

    assert "description" in result.columns
    assert result.iloc[0]["description"] == "Description"


@patch('cheat_at_search.bc_plus_data._load_dataset')
def test_load_corpus_adds_description_from_body(mock_load_dataset):
    mock_ds = MagicMock()
    mock_ds.to_pandas.return_value = pd.DataFrame({
        "doc_id": ["1"],
        "body": ["Description"],
    })
    mock_load_dataset.return_value = mock_ds

    result = bc_plus_data._load_corpus()

    assert "description" in result.columns
    assert result.iloc[0]["description"] == "Description"


@patch('cheat_at_search.bc_plus_data._load_dataset')
def test_load_corpus_adds_description_from_content(mock_load_dataset):
    mock_ds = MagicMock()
    mock_ds.to_pandas.return_value = pd.DataFrame({
        "doc_id": ["1"],
        "content": ["Description"],
    })
    mock_load_dataset.return_value = mock_ds

    result = bc_plus_data._load_corpus()

    assert "description" in result.columns
    assert result.iloc[0]["description"] == "Description"


@patch('cheat_at_search.bc_plus_data._load_dataset')
def test_load_corpus_adds_empty_description(mock_load_dataset):
    mock_ds = MagicMock()
    mock_ds.to_pandas.return_value = pd.DataFrame({
        "doc_id": ["1"],
    })
    mock_load_dataset.return_value = mock_ds

    result = bc_plus_data._load_corpus()

    assert "description" in result.columns
    assert result.iloc[0]["description"] == ""


@patch('cheat_at_search.bc_plus_data._load_dataset')
def test_load_corpus_dict_split(mock_load_dataset):
    mock_ds = {"train": MagicMock()}
    mock_ds["train"].to_pandas.return_value = pd.DataFrame({
        "doc_id": ["1"],
        "title": ["A"],
    })
    mock_load_dataset.return_value = mock_ds

    result = bc_plus_data._load_corpus()

    assert "doc_id" in result.columns
