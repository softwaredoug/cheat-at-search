from cheat_at_search.logger import log_to_stdout
from cheat_at_search import msmarco_data
import pandas as pd


logger = log_to_stdout("minimarco_data")


def _docs():
    msmarco_data.download_msmarco()
    collection_path = msmarco_data.msmarco_path / "collection.tsv"
    passages = pd.read_csv(collection_path, sep="\t", names=["doc_id", "description"])
    sample_size = min(len(passages), 300_000)
    passages = passages.sample(n=sample_size, random_state=42).reset_index(drop=True)
    passages["title"] = ""
    return passages


def _qrels(variant="dev"):
    qrels = msmarco_data._qrels(variant)
    corpus = globals().get("corpus")
    if corpus is None:
        corpus = _docs()
        globals()["corpus"] = corpus
    corpus_doc_ids = set(corpus["doc_id"].tolist())
    return qrels[qrels["doc_id"].isin(corpus_doc_ids)].reset_index(drop=True)


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
