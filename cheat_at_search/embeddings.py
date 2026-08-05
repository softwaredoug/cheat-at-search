from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from tqdm import tqdm

from cheat_at_search.data_dir import ensure_data_subdir
from cheat_at_search.logger import log_to_stdout


logger = log_to_stdout("embeddings")
_MODEL_REGISTRY: dict[tuple[str, str | None], object] = {}

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 10000


class NumpyArrayIterator:
    """Iterate over vectors from a list of NumPy array paths."""

    def __init__(self, paths: list[str]):
        self._paths = paths

    def __iter__(self) -> Iterator[np.ndarray]:
        for path in self._paths:
            array = np.load(path, mmap_mode="r")
            for vector in array:
                yield vector


def _load_model(model_name: str, device: str | None = None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for embedding search. "
            "Install it with `poetry add sentence-transformers` or `pip install sentence-transformers`."
        ) from exc
    if device:
        return SentenceTransformer(model_name, device=device)
    return SentenceTransformer(model_name)


def load_model(model_name: str, device: str | None = None):
    key = (model_name, device)
    model = _MODEL_REGISTRY.get(key)
    if model is not None:
        return model
    model = _load_model(model_name, device=device)
    _MODEL_REGISTRY[key] = model
    return model


def _passage_fn_id(passage_fn) -> str:
    module = getattr(passage_fn, "__module__", "<unknown>")
    qualname = getattr(passage_fn, "__qualname__", getattr(passage_fn, "__name__", "<unknown>"))
    base = f"{module}:{qualname}"
    try:
        source = inspect.getsource(passage_fn)
    except (OSError, TypeError):
        return base
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return f"{base}:{source_hash}"


def _hash_doc_ids(corpus) -> str:
    if "doc_id" not in corpus.columns:
        raise ValueError("Corpus must include doc_id for embedding caching.")
    hasher = hashlib.sha256()
    hasher.update(str(len(corpus)).encode("utf-8"))
    hasher.update(b"|")
    for value in corpus["doc_id"].tolist():
        hasher.update(str(value).encode("utf-8"))
        hasher.update(b"|")
    return hasher.hexdigest()


def _signature(corpus, model_name: str, passage_fn) -> str:
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
    hasher.update(b"|")
    hasher.update(_passage_fn_id(passage_fn).encode("utf-8"))
    hasher.update(b"|")
    hasher.update(_hash_doc_ids(corpus).encode("utf-8"))
    return hasher.hexdigest()


def _cache_root() -> Path:
    root = Path(ensure_data_subdir("embeddings"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _manifest_path(signature: str) -> Path:
    return _cache_root() / f"embeddings_{signature}.manifest.json"


def _chunk_path(signature: str, chunk_index: int) -> Path:
    return _cache_root() / f"embeddings_{signature}_chunk_{chunk_index}.npy"


def _load_manifest(signature: str, model_name: str, passage_fn_id: str):
    manifest_path = _manifest_path(signature)
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if meta.get("signature") != signature:
        return None
    if meta.get("model") != model_name:
        return None
    if meta.get("passage_fn_id") != passage_fn_id:
        return None
    return meta


def _save_manifest(signature: str, meta: dict) -> None:
    manifest_path = _manifest_path(signature)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle)


def default_passage_fn(row: Any) -> str:
    """Build the text used to embed one corpus row."""
    title = row.get("title")
    description = row.get("description", "")

    if title:
        return f"{title}\n\n{description}"
    return description


def load_or_create_embeddings(
    corpus,
    passage_fn=None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    show_progress: bool = True,
):
    if passage_fn is None:
        passage_fn = default_passage_fn
    signature = _signature(corpus, model_name, passage_fn)
    passage_fn_id = _passage_fn_id(passage_fn)
    total_count = len(corpus)

    manifest = _load_manifest(signature, model_name, passage_fn_id)
    if manifest is None:
        manifest = {
            "signature": signature,
            "model": model_name,
            "passage_fn_id": passage_fn_id,
            "dim": None,
            "count": total_count,
            "chunk_size": chunk_size,
            "num_chunks": int(math.ceil(total_count / chunk_size)) if chunk_size > 0 else 0,
            "completed_chunks": [],
        }
    else:
        chunk_size = int(manifest.get("chunk_size", chunk_size))

    num_chunks = int(math.ceil(total_count / chunk_size)) if chunk_size > 0 else 0
    dim = manifest.get("dim")
    model = None
    completed = set(manifest.get("completed_chunks", []))
    chunk_paths = []

    chunk_iter = range(num_chunks)
    if show_progress and num_chunks > 0:
        chunk_iter = tqdm(chunk_iter, desc="Embedding chunks")

    for chunk_index in chunk_iter:
        chunk_file = _chunk_path(signature, chunk_index)
        start = chunk_index * chunk_size
        end = min(start + chunk_size, total_count)
        expected_rows = end - start

        if chunk_file.exists():
            chunk = np.load(chunk_file)
            if chunk.ndim == 2 and chunk.shape[0] == expected_rows:
                if dim is None:
                    dim = int(chunk.shape[1])
                completed.add(chunk_index)
                chunk_paths.append(str(chunk_file))
                continue

        if model is None:
            model = load_model(model_name, device=device)
        texts = [passage_fn(row) for _, row in corpus.iloc[start:end].iterrows()]
        chunk = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        if chunk.ndim != 2:
            chunk = np.asarray(chunk)
        if dim is None:
            dim = int(chunk.shape[1])
        np.save(chunk_file, chunk)
        completed.add(chunk_index)
        chunk_paths.append(str(chunk_file))
        manifest.update({
            "dim": dim,
            "count": total_count,
            "chunk_size": chunk_size,
            "num_chunks": num_chunks,
            "completed_chunks": sorted(completed),
        })
        _save_manifest(signature, manifest)

    manifest.update({
        "dim": dim,
        "count": total_count,
        "chunk_size": chunk_size,
        "num_chunks": num_chunks,
        "completed_chunks": sorted(completed),
    })
    _save_manifest(signature, manifest)

    if model is None:
        model = _MODEL_REGISTRY.get((model_name, device))

    return NumpyArrayIterator(chunk_paths), model
