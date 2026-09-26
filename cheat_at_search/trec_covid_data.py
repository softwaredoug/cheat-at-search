import zipfile
from pathlib import Path

import pandas as pd

from cheat_at_search.data_dir import download_file, ensure_data_subdir
from cheat_at_search.indexing import load_or_build_lexical_corpus
from cheat_at_search.scifact_data import _load_corpus, _load_judgments, _load_queries


TREC_COVID_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"


def download_trec_covid() -> Path:
    data_dir = Path(ensure_data_subdir("trec_covid"))
    archive_path = download_file(TREC_COVID_URL, data_dir)
    dataset_path = data_dir / "trec-covid"
    if not dataset_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(data_dir)
    return dataset_path


def _load_cached_dataset(dataset_path: Path, cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    queries_path = cache_dir / "queries.parquet"
    judgments_path = cache_dir / "judgments.parquet"
    if queries_path.exists() and judgments_path.exists():
        return pd.read_parquet(queries_path), pd.read_parquet(judgments_path)

    queries = _load_queries(dataset_path)
    judgments = _load_judgments(dataset_path, queries)
    queries.to_parquet(queries_path, index=False)
    judgments.to_parquet(judgments_path, index=False)
    return queries, judgments


def __getattr__(name):
    if name in globals():
        return globals()[name]

    dataset_path = download_trec_covid()
    cache_dir = Path(ensure_data_subdir("trec_covid"))
    if name in {"queries", "judgments"}:
        queries_df, judgments_df = _load_cached_dataset(dataset_path, cache_dir)
        globals()["queries"] = queries_df
        globals()["judgments"] = judgments_df
        return globals()[name]
    if name == "corpus":
        corpus_path = cache_dir / "corpus.parquet"
        if corpus_path.exists():
            corpus = pd.read_parquet(corpus_path)
        else:
            corpus = _load_corpus(dataset_path)
            corpus.to_parquet(corpus_path, index=False)
        corpus = load_or_build_lexical_corpus(corpus, "trec_covid")
        globals()["corpus"] = corpus
        return corpus
    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}")
