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

    embeddings_first, _ = load_or_create_embeddings(
        corpus,
        passage_fn,
        model_name=model_name,
        chunk_size=5000,
        show_progress=True,
    )

    embeddings_first = np.stack(list(embeddings_first))
    assert isinstance(embeddings_first, np.ndarray)
    assert embeddings_first.shape[0] == len(corpus)

    cache_dir = Path(ensure_data_subdir("embeddings"))
    chunk_files = sorted(cache_dir.glob(f"embeddings_{signature}_chunk_*.npy"))
    assert chunk_files

    mtimes = {path: path.stat().st_mtime for path in chunk_files}

    embeddings_second, _ = load_or_create_embeddings(
        corpus,
        passage_fn,
        model_name=model_name,
        chunk_size=5000,
        show_progress=True,
    )

    assert np.array_equal(embeddings_first, np.stack(list(embeddings_second)))

    for path in chunk_files:
        assert path.stat().st_mtime == mtimes[path]


def test_wands_embeddings_match_direct_encoding(mounted_data_dir):
    try:
        import lzma  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("lzma module not available in this Python build")
    sentence_transformers = pytest.importorskip("sentence_transformers")

    corpus = wands_data.corpus
    model = sentence_transformers.SentenceTransformer(DEFAULT_MODEL_NAME)
    texts = [passage_fn(row) for _, row in corpus.iterrows()]
    direct_embeddings = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    cached_embeddings, _ = load_or_create_embeddings(
        corpus,
        passage_fn,
        model_name=DEFAULT_MODEL_NAME,
        chunk_size=5000,
        show_progress=False,
    )

    cached_embeddings = np.stack(list(cached_embeddings))
    assert direct_embeddings.shape == cached_embeddings.shape
    assert np.allclose(direct_embeddings, cached_embeddings, rtol=1e-6, atol=1e-6)

    queries = [
        "red sofa",
        "wood table",
        "blue chair",
    ]
    query_embeddings = model.encode(
        queries,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    scores_direct = query_embeddings @ direct_embeddings.T
    scores_cached = query_embeddings @ cached_embeddings.T
    assert np.allclose(scores_direct, scores_cached, rtol=1e-6, atol=1e-6)
