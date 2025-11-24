from cheat_at_search.logger import log_to_stdout
from cheat_at_search.data_dir import ensure_data_subdir, download_file
from pathlib import Path
import tarfile
import pandas as pd


msmarco_path = Path(ensure_data_subdir("msmarco"))

logger = log_to_stdout("msmarco_data")


def download_msmarco():
    # Download to fixtures
    print("Downloading MSMARCO Passage Retrieval")

    url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz"
    local_path = download_file(url, msmarco_path)

    test_file = "collection.tsv"
    if Path(msmarco_path / test_file).exists():
        pass
    else:
        with tarfile.open(f"{local_path}", "r:gz") as tar:
            tar.extractall(path=msmarco_path)


def _docs():
    download_msmarco()
    collection_path = msmarco_path / "collection.tsv"
    passages = pd.read_csv(collection_path, sep="\t", names=["doc_id", "description"])
    passages["title"] = ""
    return passages


def _qrels(variant="dev"):
    download_msmarco()
    qrels_path = None
    queries_path = None
    if variant == "dev":
        qrels_path = msmarco_path / "qrels.dev.small.tsv"
        queries_path = msmarco_path / "queries.dev.small.tsv"
    elif variant == "train":
        qrels_path = msmarco_path / "qrels.train.tsv"
        queries_path = msmarco_path / "queries.train.tsv"
    else:
        raise ValueError(f"Unknown variant {variant} for qrels")

    qrels = pd.read_csv(qrels_path,
                        usecols=[0, 2, 3],
                        sep="\t", names=["query_id", "doc_id", "grade"])
    queries = pd.read_csv(queries_path, sep="\t", names=["query_id", "query"])
    qrels = qrels.merge(queries, on="query_id", how="left")
    return qrels


def __getattr__(name):
    """Load dataset lazily."""
    ds = None
    if name in globals():
        return globals()[name]
    if name == "judgments" or name == "queries":
        ds = _qrels()
        globals()["judgments"] = ds
        queries = ds[['query', 'query_id']].drop_duplicates().reset_index(drop=True)
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
