import pandas as pd

from cheat_at_search import doug_blog_data


_QUESTION_SPECS = [
    ("bm25", "bm25"),
    ("learning_to_rank", "learning to rank"),
    ("elasticsearch", "elasticsearch"),
    ("ai", "ai"),
    ("pandas", "pandas"),
    ("search relevance", "search relevance"),
    ("twitter", "twitter"),
    ("json", "json"),
    ("search_engine", "search engine"),
    ("plugin", "plugin"),
]


def _judgments() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"query_id": query_id, "query": query, "answer": ""}
            for query_id, query in _QUESTION_SPECS
        ],
        columns=["query_id", "query", "answer"],
    )


def __getattr__(name):
    """Load the answer dataset lazily."""
    if name in globals():
        return globals()[name]
    if name == "judgments" or name == "queries":
        judgments = _judgments()
        globals()["judgments"] = judgments
        queries = judgments[["query", "query_id"]].drop_duplicates().reset_index(
            drop=True
        )
        globals()["queries"] = queries
        return globals()[name]
    if name == "corpus":
        corpus = doug_blog_data.corpus
        globals()["corpus"] = corpus
        return corpus
    raise AttributeError(f"Module {__name__} has no attribute {name}")
