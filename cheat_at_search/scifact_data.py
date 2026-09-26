import json
import zipfile
from pathlib import Path

import pandas as pd

from cheat_at_search.data_dir import download_file, ensure_data_subdir
from cheat_at_search.indexing import load_or_build_lexical_corpus
from cheat_at_search.logger import log_to_stdout


logger = log_to_stdout("scifact_data")

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def download_scifact() -> Path:
    data_dir = Path(ensure_data_subdir("scifact"))
    archive_path = download_file(SCIFACT_URL, data_dir)
    dataset_path = data_dir / "scifact"
    if not dataset_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(data_dir)
    return dataset_path


def _source_path(dataset_path: Path, filename: str) -> Path:
    path = dataset_path / filename
    if path.exists():
        return path
    nested_path = dataset_path / "scifact" / filename
    if nested_path.exists():
        return nested_path
    raise FileNotFoundError(f"SciFact file not found: {filename}")


def _read_jsonl(path: Path) -> pd.DataFrame:
    with path.open() as handle:
        return pd.DataFrame(json.loads(line) for line in handle if line.strip())


def _load_corpus(dataset_path: Path) -> pd.DataFrame:
    corpus = _read_jsonl(_source_path(dataset_path, "corpus.jsonl"))
    corpus = corpus.rename(columns={"_id": "doc_id", "text": "description"})
    corpus["doc_id"] = corpus["doc_id"].astype(str)
    if "title" not in corpus:
        corpus["title"] = ""
    corpus["title"] = corpus["title"].fillna("")
    corpus["description"] = corpus["description"].fillna("")
    return corpus[["doc_id", "title", "description"]]


def _load_queries(dataset_path: Path) -> pd.DataFrame:
    queries = _read_jsonl(_source_path(dataset_path, "queries.jsonl"))
    queries = queries.rename(columns={"_id": "query_id", "text": "query"})
    queries["query_id"] = queries["query_id"].astype(str)
    queries["query"] = queries["query"].fillna("")
    return queries[["query_id", "query"]]


def _load_judgments(dataset_path: Path, queries: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    qrels_path = _source_path(dataset_path, f"qrels/{split}.tsv")
    qrels = pd.read_csv(qrels_path, sep="\t")
    qrels = qrels.rename(
        columns={"query-id": "query_id", "corpus-id": "doc_id", "score": "grade"}
    )
    qrels["query_id"] = qrels["query_id"].astype(str)
    qrels["doc_id"] = qrels["doc_id"].astype(str)
    judgments = qrels.merge(queries, on="query_id", how="left", validate="many_to_one")
    return judgments[["query_id", "query", "doc_id", "grade"]]


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

    dataset_path = download_scifact()
    cache_dir = Path(ensure_data_subdir("scifact"))
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
        corpus = load_or_build_lexical_corpus(corpus, "scifact")
        globals()["corpus"] = corpus
        return corpus
    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}")
