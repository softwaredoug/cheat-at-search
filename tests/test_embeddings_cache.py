import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from cheat_at_search.data_dir import mount, ensure_data_subdir
from cheat_at_search.embeddings import load_or_create_embeddings


class DummyModel:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        self.calls += 1
        rows = len(texts)
        data = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
        return data


def passage_fn(row):
    return f"{row['title']} {row['description']}".strip()


@pytest.fixture(scope="function")
def mounted_data_dir():
    data_dir = tempfile.mkdtemp()
    mount(use_gdrive=False, manual_path=data_dir, load_keys=False)
    return Path(data_dir)


@patch("cheat_at_search.embeddings._load_model")
def test_embeddings_cache_reuse(mock_load_model, mounted_data_dir):
    dummy = DummyModel()
    mock_load_model.return_value = dummy
    corpus = pd.DataFrame({
        "doc_id": [1, 2, 3],
        "title": ["one", "two", "three"],
        "description": ["alpha", "beta", "gamma"],
    })

    first = load_or_create_embeddings(
        corpus,
        passage_fn,
        model_name="test-model",
        chunk_size=2,
        show_progress=False,
    )
    assert mock_load_model.call_count == 1
    assert dummy.calls == 2

    dummy_second = DummyModel()
    mock_load_model.return_value = dummy_second
    second = load_or_create_embeddings(
        corpus,
        passage_fn,
        model_name="test-model",
        chunk_size=2,
        show_progress=False,
    )
    assert mock_load_model.call_count == 1
    assert dummy_second.calls == 0
    assert np.array_equal(first, second)

    cache_dir = Path(ensure_data_subdir("embeddings"))
    manifests = list(cache_dir.glob("embeddings_*.manifest.json"))
    assert len(manifests) == 1


@patch("cheat_at_search.embeddings._load_model")
def test_embeddings_cache_changes_with_model(mock_load_model, mounted_data_dir):
    dummy = DummyModel()
    mock_load_model.return_value = dummy
    corpus = pd.DataFrame({
        "doc_id": [10, 20],
        "title": ["hello", "world"],
        "description": ["first", "second"],
    })

    load_or_create_embeddings(
        corpus,
        passage_fn,
        model_name="test-model-a",
        chunk_size=2,
        show_progress=False,
    )
    load_or_create_embeddings(
        corpus,
        passage_fn,
        model_name="test-model-b",
        chunk_size=2,
        show_progress=False,
    )

    cache_dir = Path(ensure_data_subdir("embeddings"))
    manifests = list(cache_dir.glob("embeddings_*.manifest.json"))
    assert len(manifests) == 2


def test_embeddings_require_doc_id(mounted_data_dir):
    corpus = pd.DataFrame({
        "title": ["one"],
        "description": ["alpha"],
    })
    with pytest.raises(ValueError, match="doc_id"):
        load_or_create_embeddings(
            corpus,
            passage_fn,
            model_name="test-model",
            chunk_size=1,
            show_progress=False,
        )
