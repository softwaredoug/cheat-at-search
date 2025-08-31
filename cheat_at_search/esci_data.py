# https://github.com/softwaredoug/esci-data
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.data_dir import DATA_PATH, sync_git_repo
from pathlib import Path
import pandas as pd


logger = log_to_stdout("esci_data")

esci_path = Path(DATA_PATH) / "esci"


def fetch_esci(data_dir=esci_path, repo_url="https://github.com/softwaredoug/esci-data.git"):
    """
    Clone the Amazon Science ESCI dataset from GitHub into a data directory.

    Args:
        data_dir: Path where the ESCI dataset will be stored
                  (default: "data/esci")
        repo_url: URL of the ESCI repository
                  (default: "https://github.com/softwaredoug/esci-data.git")

    Returns:
        Path object pointing to the cloned repository
    """
    return sync_git_repo(data_dir, repo_url)


def _judgments():
    esci_queries_path = fetch_esci() / "shopping_queries_dataset" / "shopping_queries_dataset_examples.parquet"
    return pd.read_parquet(esci_queries_path)


def _products():
    products_path = fetch_esci() / "shopping_queries_dataset" / "shopping_queries_dataset_products.parquet"
    return pd.read_parquet(products_path)


def __getattr__(name):
    """Load dataset lazily."""
    ds = None
    if name in globals():
        return globals()[name]
    if name == "judgments" or name == "queries":
        ds = _judgments()
        globals()["judgments"] = ds
        queries = ds['query'].unique()
        globals()["queries"] = queries
        return globals()[name]
    elif name == "products":
        ds = _products()
        globals()["products"] = ds
        return globals()[name]
    else:
        raise AttributeError(f"Module {__name__} has no attribute {name}")
