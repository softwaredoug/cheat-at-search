from cheat_at_search.logger import log_to_stdout
from cheat_at_search import msmarco_data
from cheat_at_search.tokenizers import snowball_tokenizer
import pandas as pd


logger = log_to_stdout("minimarco_data")


def _docs():
    msmarco_data.download_msmarco()
    collection_path = msmarco_data.msmarco_path / "collection.tsv"
    passages = pd.read_csv(collection_path, sep="\t", names=["doc_id", "description"])

    judgments = globals().get("judgments")
    if judgments is None:
        judgments = _qrels()
        globals()["judgments"] = judgments

    required_doc_ids = set(judgments.loc[judgments["grade"] > 0, "doc_id"].tolist())
    query_rows = judgments[["query_id", "query"]].drop_duplicates().reset_index(drop=True)
    for _, row in query_rows.iterrows():
        terms = snowball_tokenizer(row["query"])
        for term in terms:
            with_term = passages[
                passages["description"].str.contains(term, case=False, regex=False, na=False)
            ]
            if with_term.empty:
                continue
            sample_size = min(100, len(with_term))
            sampled = with_term.sample(n=sample_size, random_state=42)
            required_doc_ids.update(sampled["doc_id"].tolist())

    required_docs = passages[passages["doc_id"].isin(required_doc_ids)]
    remaining = passages[~passages["doc_id"].isin(required_doc_ids)]
    remaining_needed = 300_000 - len(required_docs)
    if remaining_needed <= 0:
        corpus = required_docs.sample(n=300_000, random_state=42).reset_index(drop=True)
    else:
        remaining_sample = remaining.sample(
            n=min(remaining_needed, len(remaining)), random_state=42
        )
        corpus = pd.concat([required_docs, remaining_sample], ignore_index=True)

    corpus["title"] = ""
    return corpus


def _qrels(variant="dev"):
    qrels = msmarco_data._qrels(variant)
    qrels = qrels[qrels["grade"] > 0].reset_index(drop=True)
    queries = qrels[["query_id", "query"]].drop_duplicates().reset_index(drop=True)
    sample_size = min(len(queries), 500)
    sampled_queries = queries.sample(n=sample_size, random_state=42)
    qrels = qrels[qrels["query_id"].isin(sampled_queries["query_id"])].reset_index(drop=True)
    return qrels


def __getattr__(name):
    """Load dataset lazily."""
    ds = None
    if name in globals():
        return globals()[name]
    if name == "judgments" or name == "queries":
        ds = _qrels()
        globals()["judgments"] = ds
        queries = ds[["query", "query_id"]].drop_duplicates().reset_index(drop=True)
        globals()["queries"] = queries
        return globals()[name]
    elif name == "corpus":
        ds = _docs()
        globals()["corpus"] = ds
        return globals()[name]
    else:
        raise AttributeError(f"Module {__name__} has no attribute {name}")


if __name__ == "__main__":
    _qrels()
