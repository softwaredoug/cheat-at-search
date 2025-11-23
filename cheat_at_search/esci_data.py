# https://github.com/softwaredoug/esci-data
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.data_dir import DATA_PATH, sync_git_repo
from pathlib import Path
import pandas as pd
from lxml.etree import ParserError
from lxml import html


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
    judgments = pd.read_parquet(esci_queries_path)
    #  RelevanceLevels = Literal['🤩', '🙂', '😐', '😭']
    judgments['grade'] = judgments['esci_label'].map({'E': 3,
                                                      'S': 2,
                                                      'C': 1,
                                                      'I': 0})
    judgments['label'] = judgments['esci_label'].map({'E': '🤩',
                                                      'S': '🙂',
                                                      'C': '😐',
                                                      'I': '😭'})
    judgments['doc_id'] = judgments['product_id']
    return judgments


def _docs():
    products_path = fetch_esci() / "shopping_queries_dataset" / "shopping_queries_dataset_products.parquet"
    products = pd.read_parquet(products_path)
    products['doc_id'] = products['product_id']
    products['product_name'] = products['product_title']
    products['title'] = products['product_title']

    # Cleanup description HTML
    def _clean_html(desc):
        if pd.isna(desc):
            return ""
        if desc.strip() == "":
            return ""
        try:
            tree = html.fromstring(desc)
            text = tree.text_content()
            return ' '.join(text.split())
        except ParserError:
            if desc.startswith('<') and desc.endswith('>'):
                return ""
            return desc

    products['product_description'] = products['product_description'].apply(_clean_html)
    logger.info("Cleaned HTML from product descriptions.")
    products['description'] = products['product_description']
    return products


def __getattr__(name):
    """Load dataset lazily."""
    ds = None
    if name in globals():
        return globals()[name]
    if name == "judgments" or name == "queries":
        ds = _judgments()
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
