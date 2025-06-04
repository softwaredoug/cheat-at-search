import subprocess
import logging
from pathlib import Path
import pandas as pd
from cheat_at_search.logger import log_to_stdout


logger = logging.getLogger(__name__)


def fetch_wands(data_dir="data/wands", repo_url="https://github.com/wayfair/WANDS.git"):
    """
    Clone the Wayfair Annotated Dataset (WANDS) from GitHub into a data directory.

    Args:
        data_dir: Path where the WANDS dataset will be stored
                  (default: "data/wands")
        repo_url: URL of the WANDS repository
                  (default: "https://github.com/wayfair/WANDS.git")

    Returns:
        Path object pointing to the cloned repository
    """
    # Convert to absolute path
    data_path = Path(data_dir).absolute()

    # Create directories if they don't exist
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if the directory already exists
    if data_path.exists():
        logger.info(f"Directory {data_path} already exists. Skipping clone.")
        return data_path

    logger.info(f"Cloning WANDS dataset from {repo_url} to {data_path}")

    try:
        # Clone the repository
        subprocess.run(
            ["git", "clone", repo_url, str(data_path)],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Successfully cloned WANDS dataset to {data_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone WANDS dataset: {e.stderr}")
        raise

    return data_path


def _products(data_dir="data/wands"):
    """
    Load WANDS products into a pandas DataFrame.

    Args:
        data_dir: Path to the WANDS dataset directory
                 (default: "data/wands")

    Returns:
        pandas.DataFrame containing product data
    """
    # Ensure we have the data
    data_path = fetch_wands(data_dir)

    # Path to the products CSV file
    products_file = data_path / "dataset" / "product.csv"

    if not products_file.exists():
        logger.error(f"Products file not found at {products_file}")
        raise FileNotFoundError(f"Products file not found at {products_file}")

    logger.info(f"Loading products from {products_file}")

    # Load the tab-delimited CSV file
    df = pd.read_csv(products_file, sep='\t')

    logger.info(f"Loaded {len(df)} products")

    split_features = df['product_features'].str.split('|')
    df['features'] = split_features
    df['product_description'].fillna('', inplace=True)
    df['product_name'].fillna('', inplace=True)

    # Parse category and subcategory from 'category hierarchy'
    cat_as_list = df['category hierarchy'].fillna('').str.split('/')
    df['category'] = cat_as_list.apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else '')
    df['sub_category'] = cat_as_list.apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else '')
    df['cat_subcat'] = df['category'] + ' / ' + df['sub_category']

    return df


def _queries(data_dir="data/wands"):
    """
    Load WANDS queries into a pandas DataFrame.

    Args:
        data_dir: Path to the WANDS dataset directory
                 (default: "data/wands")

    Returns:
        pandas.DataFrame containing query data
    """
    # Ensure we have the data
    data_path = fetch_wands(data_dir)

    # Path to the queries CSV file
    queries_file = data_path / "dataset" / "query.csv"

    if not queries_file.exists():
        logger.error(f"Queries file not found at {queries_file}")
        raise FileNotFoundError(f"Queries file not found at {queries_file}")

    logger.info(f"Loading queries from {queries_file}")

    # Load the tab-delimited CSV file
    df = pd.read_csv(queries_file, sep='\t')

    logger.info(f"Loaded {len(df)} queries")

    return df


def _labels(data_dir="data/wands"):
    """
    Load WANDS relevance labels into a pandas DataFrame.

    Args:
        data_dir: Path to the WANDS dataset directory
                 (default: "data/wands")

    Returns:
        pandas.DataFrame containing relevance label data
    """
    logger = log_to_stdout(logger_name=__name__)

    # Ensure we have the data
    data_path = fetch_wands(data_dir)

    # Path to the labels CSV file
    labels_file = data_path / "dataset" / "label.csv"

    if not labels_file.exists():
        logger.error(f"Labels file not found at {labels_file}")
        raise FileNotFoundError(f"Labels file not found at {labels_file}")

    logger.info(f"Loading relevance labels from {labels_file}")

    # Load the tab-delimited CSV file
    df = pd.read_csv(labels_file, sep='\t')
    df.loc[df['label'] == 'Exact', 'grade'] = 2
    df.loc[df['label'] == 'Partial', 'grade'] = 1
    df.loc[df['label'] == 'Irrelevant', 'grade'] = 0
    df = df.groupby(['query_id', 'product_id']).first().reset_index()
    logger.info(f"Loaded {len(df)} relevance labels")
    return df


labels = _labels()
queries = _queries()
products = _products()

labeled_queries = queries.merge(labels, how='left', on='query_id')
labeled_query_products = labeled_queries.merge(products, how='left', on='product_id')


def rel_attribute(query_products=labeled_query_products, grade=2, column='category'):
    """Relevant categories in the labeled data useful for ground truth of different attributes."""
    return query_products[query_products['grade'] == 2].groupby(['query', column])[column].count().sort_values(ascending=False)

# In [13]: ndcgs(graded_category).mean()
# Out[13]: np.float64(0.5561577962937873)

if __name__ == "__main__":
    # Configure root logger
    root_logger = log_to_stdout(logger_name=None, level="INFO")

    # Test the functions
    fetch_wands()

    # Load and display product info
    products_df = products()
    print(f"Products shape: {products_df.shape}")
    print(f"Products columns: {products_df.columns.tolist()}")

    # Load and display query info
    queries_df = queries()
    print(f"Queries shape: {queries_df.shape}")
    print(f"Queries columns: {queries_df.columns.tolist()}")

    # Load and display label info
    labels_df = labels()
    print(f"Labels shape: {labels_df.shape}")
    print(f"Labels columns: {labels_df.columns.tolist()}")
