from pathlib import Path

import pandas as pd
from searcharray import SearchArray

from cheat_at_search.data_dir import ensure_data_subdir
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.tokenizers import snowball_tokenizer


logger = log_to_stdout("indexing")


def _normalize_text_column(corpus: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in corpus.columns:
        corpus[column] = ""
        return corpus
    corpus[column] = corpus[column].fillna("")
    return corpus


def _build_lexical_indexes(corpus: pd.DataFrame) -> pd.DataFrame:
    if "title_snowball" not in corpus.columns:
        corpus["title_snowball"] = SearchArray.index(
            corpus["title"], tokenizer=snowball_tokenizer
        )
    if "description_snowball" not in corpus.columns:
        corpus["description_snowball"] = SearchArray.index(
            corpus["description"], tokenizer=snowball_tokenizer
        )
    return corpus


def lexical_index_cache_path(dataset_name: str) -> Path:
    cache_dir = Path(ensure_data_subdir("lexical_indexes"))
    return cache_dir / f"{dataset_name}_corpus.pkl"


def load_or_build_lexical_corpus(corpus: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    cache_path = lexical_index_cache_path(dataset_name)
    if cache_path.exists():
        cached = pd.read_pickle(cache_path)
        if "title_snowball" in cached.columns and "description_snowball" in cached.columns:
            return cached
        logger.info(
            "Cached lexical corpus for %s missing snowball columns; rebuilding.",
            dataset_name,
        )

    corpus = corpus.copy()
    corpus = _normalize_text_column(corpus, "title")
    corpus = _normalize_text_column(corpus, "description")
    corpus = _build_lexical_indexes(corpus)
    corpus.to_pickle(cache_path)
    return corpus
