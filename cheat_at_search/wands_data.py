import subprocess
import logging
from pathlib import Path
import pandas as pd
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.data_dir import DATA_PATH


logger = logging.getLogger(__name__)


wands_path = Path(DATA_PATH) / "wands_enriched"


def fetch_wands(data_dir=wands_path, repo_url="https://github.com/softwaredoug/WANDS.git"):
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
        logger.info(f"Directory {data_path} already exists. Checking for updates...")
        try:
            subprocess.run(
                ["git", "-C", str(data_path), "fetch", "origin"],
                check=True,
                capture_output=True,
                text=True
            )
            subprocess.run(
                ["git", "-C", str(data_path), "reset", "--hard", "origin/main"],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Updated WANDS dataset at {data_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to update WANDS dataset: {e.stderr}")
            raise
        return data_path
        return data_path

    logger.info(f"Cloning WANDS dataset from {repo_url} to {data_path}")

    try:
        # Clone the repository
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, str(data_path)],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Successfully cloned WANDS dataset to {data_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone WANDS dataset: {e.stderr}")
        raise

    return data_path


def _products():
    """
    Load WANDS products into a pandas DataFrame.

    Args:
        data_dir: Path to the WANDS dataset directory
                 (default: "data/wands")

    Returns:
        pandas.DataFrame containing product data
    """
    # Ensure we have the data
    data_path = fetch_wands()

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


def _enriched_products():
    """Queries enriched with LLM information."""
    # Ensure we have the data
    data_path = fetch_wands()

    # Path to the products CSV file
    products_file = data_path / "dataset" / "enriched" / "enriched_products.csv.gz"

    if not products_file.exists():
        logger.error(f"Enriched products file not found at {products_file}")
        raise FileNotFoundError(f"Enriched products file not found at {products_file}")

    logger.info(f"Loading enriched products from {products_file}")

    df = pd.read_csv(products_file, compression='gzip')

    df['classification'] = df['category hierarchy']
    for col in df.columns:
        df[col].fillna('', inplace=True)

    logger.info(f"Loaded {len(df)} enriched products")
    return df


def _queries():
    """
    Load WANDS queries into a pandas DataFrame.

    Args:
        data_dir: Path to the WANDS dataset directory
                 (default: "data/wands")

    Returns:
        pandas.DataFrame containing query data
    """
    # Ensure we have the data
    data_path = fetch_wands()

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


def _enriched_queries():
    """Queries enriched with LLM information."""
    # Ensure we have the data
    data_path = fetch_wands()

    # Path to the queries CSV file
    queries_file = data_path / "dataset" / "enriched" / "query_attributes.csv"

    if not queries_file.exists():
        logger.error(f"Enriched queries file not found at {queries_file}")
        raise FileNotFoundError(f"Enriched queries file not found at {queries_file}")

    logger.info(f"Loading enriched queries from {queries_file}")

    df = pd.read_csv(queries_file)
    df['materials'].fillna('unknown', inplace=True)
    df['classification'] = df['query_classification']
    for col in df.columns:
        df[col].fillna('unknown', inplace=True)

    logger.info(f"Loaded {len(df)} enriched queries")
    return df


def _query_bags():
    """Load the query bags helping measure similarity."""
    # Ensure we have the data
    data_path = fetch_wands()

    # Path to the query bags CSV file
    query_bags_file = data_path / "dataset" / "query_bags.pkl"

    if not query_bags_file.exists():
        logger.error(f"Query bags file not found at {query_bags_file}")
        raise FileNotFoundError(f"Query bags file not found at {query_bags_file}")

    logger.info(f"Loading query bags from {query_bags_file}")

    # Load the tab-delimited CSV file
    df = pd.read_pickle(query_bags_file)

    logger.info(f"Loaded {len(df)} query bags")

    return df


def _labels():
    """
    Load WANDS relevance labels into a pandas DataFrame.

    Args:
        data_dir: Path to the WANDS dataset directory
                 (default: "data/wands")

    Returns:
        pandas.DataFrame containing relevance label data
    """
    # Ensure we have the data
    data_path = fetch_wands()

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
enriched_products = _enriched_products()
enriched_queries = _enriched_queries()
query_bags = _query_bags()

labeled_queries = queries.merge(labels, how='left', on='query_id')
labeled_query_products = labeled_queries.merge(products, how='left', on='product_id')


def _ideal10(labeled_queries):

    ideal_results = labeled_queries.sort_values(['query_id', 'grade'], ascending=(True, False))
    ideal_results['rank'] = ideal_results.groupby('query_id').cumcount() + 1
    ideal_top_10 = ideal_results[ideal_results['rank'] <= 10] \
        .add_prefix('ideal_') \
        .rename(columns={'ideal_query_id': 'query_id', 'ideal_query': 'query'})

    ideal_top_10 = ideal_top_10.merge(
        products[['product_id', 'product_name']], how='left', left_on='ideal_product_id', right_on='product_id'
    ).rename(columns={'product_name': 'ideal_product_name'}).drop(columns='ideal_query_class')

    return ideal_top_10


ideal_top_10 = _ideal10(labeled_queries)


def rel_attribute(query_products=labeled_query_products, grade=2, column='category'):
    """Relevant categories in the labeled data useful for ground truth of different attributes."""
    return query_products[query_products['grade'] == 2].groupby(['query', column])[column].count().sort_values(ascending=False)


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
