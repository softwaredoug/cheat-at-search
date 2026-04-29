import tempfile
from pathlib import Path

import numpy as np
import pytest

from cheat_at_search.data_dir import mount, ensure_data_subdir
from cheat_at_search.embeddings import load_or_create_embeddings, _signature, DEFAULT_MODEL_NAME
from cheat_at_search import wands_data


@pytest.fixture(scope="module")
def mounted_data_dir():
    data_dir = tempfile.mkdtemp()
    mount(use_gdrive=False, manual_path=data_dir, load_keys=False)
    return Path(data_dir)


def passage_fn(row):
    title = row.get("title") or ""
    description = row.get("description") or ""
    return f"{title} {description}".strip()


def test_wands_minilm_embedding_cache(mounted_data_dir):
    try:
        import lzma  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("lzma module not available in this Python build")
    pytest.importorskip("sentence_transformers")
    corpus = wands_data.corpus
    model_name = DEFAULT_MODEL_NAME
    signature = _signature(corpus, model_name, passage_fn)

    embeddings_first = load_or_create_embeddings(
        corpus,
        passage_fn,
        model_name=model_name,
        chunk_size=5000,
        show_progress=True,
    )

    assert isinstance(embeddings_first, np.ndarray)
    assert embeddings_first.shape[0] == len(corpus)

    cache_dir = Path(ensure_data_subdir("embeddings"))
    chunk_files = sorted(cache_dir.glob(f"embeddings_{signature}_chunk_*.npy"))
    assert chunk_files

    mtimes = {path: path.stat().st_mtime for path in chunk_files}

    embeddings_second = load_or_create_embeddings(
        corpus,
        passage_fn,
        model_name=model_name,
        chunk_size=5000,
        show_progress=True,
    )

    assert np.array_equal(embeddings_first, embeddings_second)

    for path in chunk_files:
        assert path.stat().st_mtime == mtimes[path]
