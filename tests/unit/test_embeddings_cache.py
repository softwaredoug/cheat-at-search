import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from cheat_at_search.data_dir import mount, ensure_data_subdir
from cheat_at_search.embeddings import (
    DEFAULT_CLIP_MODEL,
    DEFAULT_HF_CACHE_REPO,
    DEFAULT_IMAGE_CHUNK_SIZE,
    DEFAULT_MODEL_NAME,
    NumpyArrayIterator,
    clip_image_embeddings,
    default_image_fn,
    default_passage_fn,
    load_or_create_embeddings,
    text_embeddings,
)


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


def download_remote_file(repo_id, remote_path, local_path):
    if remote_path.endswith(".manifest.json"):
        return False
    np.save(local_path, np.zeros((2, 3), dtype=np.float32))
    return True


@pytest.fixture(scope="function")
def mounted_data_dir():
    data_dir = tempfile.mkdtemp()
    mount(use_gdrive=False, manual_path=data_dir, load_keys=False)
    return Path(data_dir)


@patch("cheat_at_search.embeddings._upload_remote_file", return_value=False)
@patch("cheat_at_search.embeddings._remote_file_exists", return_value=False)
@patch("cheat_at_search.embeddings._download_remote_file", return_value=False)
@patch("cheat_at_search.embeddings._load_model")
def test_embeddings_cache_reuse(
    mock_load_model,
    mock_download_remote_file,
    mock_remote_file_exists,
    mock_upload_remote_file,
    mounted_data_dir,
):
    dummy = DummyModel()
    mock_load_model.return_value = dummy
    corpus = pd.DataFrame({
        "doc_id": [1, 2, 3],
        "title": ["one", "two", "three"],
        "description": ["alpha", "beta", "gamma"],
    })

    first, _ = load_or_create_embeddings(
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
    second, _ = load_or_create_embeddings(
        corpus,
        passage_fn,
        model_name="test-model",
        chunk_size=2,
        show_progress=False,
    )
    assert mock_load_model.call_count == 1
    assert dummy_second.calls == 0
    assert np.array_equal(np.stack(list(first)), np.stack(list(second)))

    cache_dir = Path(ensure_data_subdir("embeddings"))
    manifests = list(cache_dir.glob("embeddings_*.manifest.json"))
    assert len(manifests) == 1


@patch("cheat_at_search.embeddings._upload_remote_file", return_value=False)
@patch("cheat_at_search.embeddings._remote_file_exists", return_value=False)
@patch("cheat_at_search.embeddings._download_remote_file", return_value=False)
@patch("cheat_at_search.embeddings._load_model")
def test_embeddings_cache_changes_with_model(
    mock_load_model,
    mock_download_remote_file,
    mock_remote_file_exists,
    mock_upload_remote_file,
    mounted_data_dir,
):
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


@patch("cheat_at_search.embeddings._download_remote_file", return_value=False)
def test_embeddings_require_doc_id(mock_download_remote_file, mounted_data_dir):
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


def test_numpy_array_iterator_yields_vectors(tmp_path):
    first_path = tmp_path / "first.npy"
    second_path = tmp_path / "second.npy"
    np.save(first_path, np.array([[1, 2], [3, 4]]))
    np.save(second_path, np.array([[5, 6]]))

    vectors = list(NumpyArrayIterator([str(first_path), str(second_path)]))

    assert [vector.tolist() for vector in vectors] == [[1, 2], [3, 4], [5, 6]]

    mmap_vectors = list(NumpyArrayIterator([str(first_path)], mmap=True))
    assert [vector.tolist() for vector in mmap_vectors] == [[1, 2], [3, 4]]


@patch("cheat_at_search.embeddings._upload_remote_file", return_value=True)
@patch("cheat_at_search.embeddings._remote_file_exists", return_value=False)
@patch("cheat_at_search.embeddings._download_remote_file", return_value=False)
@patch("cheat_at_search.embeddings._load_model")
def test_complete_local_chunks_skip_remote_cache(
    mock_load_model,
    mock_download_remote_file,
    mock_remote_file_exists,
    mock_upload_remote_file,
    mounted_data_dir,
):
    dummy = DummyModel()
    mock_load_model.return_value = dummy
    corpus = pd.DataFrame({
        "doc_id": [1, 2, 3],
        "title": ["one", "two", "three"],
        "description": ["", "", ""],
    })

    load_or_create_embeddings(
        corpus,
        passage_fn,
        model_name="test-model",
        chunk_size=2,
        show_progress=False,
        remote_repo_id=None,
    )
    mock_remote_file_exists.reset_mock()
    mock_upload_remote_file.reset_mock()

    load_or_create_embeddings(
        corpus,
        passage_fn,
        model_name="test-model",
        chunk_size=2,
        show_progress=False,
    )

    mock_remote_file_exists.assert_not_called()
    mock_upload_remote_file.assert_not_called()


@patch("cheat_at_search.embeddings._upload_remote_file", return_value=False)
@patch("cheat_at_search.embeddings._remote_file_exists", return_value=True)
@patch("cheat_at_search.embeddings._download_remote_file", side_effect=download_remote_file)
@patch("cheat_at_search.embeddings._load_model")
@patch.dict("cheat_at_search.embeddings._MODEL_REGISTRY", {}, clear=True)
def test_remote_chunks_are_restored(
    mock_load_model,
    mock_download_remote_file,
    mock_remote_file_exists,
    mock_upload_remote_file,
    mounted_data_dir,
):
    dummy = DummyModel()
    mock_load_model.return_value = dummy
    corpus = pd.DataFrame({
        "doc_id": [1, 2],
        "title": ["one", "two"],
        "description": ["", ""],
    })

    embeddings, model = load_or_create_embeddings(
        corpus,
        passage_fn,
        model_name="test-model",
        chunk_size=2,
        show_progress=False,
    )

    assert model is dummy
    assert dummy.calls == 0
    assert np.stack(list(embeddings)).shape == (2, 3)


@patch("cheat_at_search.embeddings.load_or_create_embeddings")
def test_clip_image_embeddings_forwards_defaults(mock_load):
    expected = (object(), object())
    mock_load.return_value = expected
    corpus = pd.DataFrame()

    result = clip_image_embeddings(corpus, device="mps", show_progress=False)

    assert result == expected
    mock_load.assert_called_once_with(
        corpus,
        passage_fn=default_image_fn,
        model_name=DEFAULT_CLIP_MODEL,
        device="mps",
        show_progress=False,
        chunk_size=DEFAULT_IMAGE_CHUNK_SIZE,
        remote_repo_id=DEFAULT_HF_CACHE_REPO,
    )


@patch("cheat_at_search.embeddings.load_or_create_embeddings")
def test_text_embeddings_forwards_defaults(mock_load):
    expected = (object(), object())
    mock_load.return_value = expected
    corpus = pd.DataFrame()

    result = text_embeddings(corpus, device="cpu", show_progress=False, chunk_size=4)

    assert result == expected
    mock_load.assert_called_once_with(
        corpus,
        passage_fn=default_passage_fn,
        model_name=DEFAULT_MODEL_NAME,
        device="cpu",
        show_progress=False,
        chunk_size=4,
        remote_repo_id=DEFAULT_HF_CACHE_REPO,
    )
