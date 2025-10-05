import logging
from pathlib import Path
import pandas as pd
import numpy as np
import sys
from cheat_at_search.data_dir import DATA_PATH, sync_git_repo


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
    return sync_git_repo(data_dir, repo_url)


def _corpus():
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
    df['product_description'] = df['product_description'].fillna('')
    df['product_name'] = df['product_name'].fillna('')

    # Normalize to common columns
    df['doc_id'] = df['product_id']
    df['title'] = df['product_name']
    df['description'] = df['product_description']

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

    # Normalize to common columns
    df['doc_id'] = df['product_id']
    df['title'] = df['product_name']
    df['description'] = df['product_description']
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


def _product_embeddings():
    """Load product embeddings from the WANDS dataset."""
    # Ensure we have the data
    data_path = fetch_wands()

    # Path to the product embeddings CSV file
    embeddings_file = data_path / "dataset" / "enriched" / "product_embeddings.npy.npz"

    if not embeddings_file.exists():
        logger.error(f"Product embeddings file not found at {embeddings_file}")
        raise FileNotFoundError(f"Product embeddings file not found at {embeddings_file}")

    logger.info(f"Loading product embeddings from {embeddings_file}")

    embeddings = np.load(embeddings_file, allow_pickle=True)
    return embeddings['arr_0']


def _product_embeddings_all():
    """Load product embeddings from the WANDS dataset."""
    # Ensure we have the data
    data_path = fetch_wands()

    # Path to the product embeddings CSV file
    embeddings_file = data_path / "dataset" / "enriched" / "product_embeddings_all_fields.npy.npz"

    if not embeddings_file.exists():
        logger.error(f"Product embeddings file not found at {embeddings_file}")
        raise FileNotFoundError(f"Product embeddings file not found at {embeddings_file}")

    logger.info(f"Loading product embeddings from {embeddings_file}")

    embeddings = np.load(embeddings_file, allow_pickle=True)
    return embeddings['arr_0']


def _query_bags():
    """Load the query bags helping measure similarity."""
    # Ensure we have the data
    data_path = fetch_wands()

    # Path to the query bags CSV file
    query_bags_file = data_path / "dataset" / "enriched" / "query_bags.pkl"

    if not query_bags_file.exists():
        logger.error(f"Query bags file not found at {query_bags_file}")
        raise FileNotFoundError(f"Query bags file not found at {query_bags_file}")

    logger.info(f"Loading query bags from {query_bags_file}")

    # Load the tab-delimited CSV file
    df = pd.read_pickle(query_bags_file)
    df = df.rename(columns={
        'category hierarchy_bag': 'classifications_bag'})

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
    df['doc_id'] = df['product_id']
    df = df.groupby(['query_id', 'doc_id']).first().reset_index()
    logger.info(f"Loaded {len(df)} relevance labels")
    return df


def _ideal10(products, labeled_queries):

    ideal_results = labeled_queries.sort_values(['query_id', 'grade'], ascending=(True, False))
    ideal_results['rank'] = ideal_results.groupby('query_id').cumcount() + 1
    ideal_top_10 = ideal_results[ideal_results['rank'] <= 10] \
        .add_prefix('ideal_') \
        .rename(columns={'ideal_query_id': 'query_id', 'ideal_query': 'query'})

    ideal_top_10 = ideal_top_10.merge(
        products[['doc_id', 'title']], how='left', left_on='ideal_doc_id', right_on='doc_id'
    ).rename(columns={'title': 'ideal_title'}).drop(columns='ideal_query_class')

    return ideal_top_10


def __getattr__(name):
    """Load dataset lazily."""
    ds = None
    if name in globals():
        return globals()[name]

    logging.info(f"Loading dataset: {name} for the first time")

    if name == 'judgments':
        ds = _labels()
    elif name == 'queries':
        ds = _queries()
    elif name == 'corpus':
        ds = _corpus()
    elif name == 'products':
        ds = _corpus()
    elif name == 'enriched_products':
        ds = _enriched_products()
    elif name == 'enriched_queries':
        ds = _enriched_queries()
    elif name == 'query_bags':
        ds = _query_bags()
    elif name == 'product_embeddings':
        ds = _product_embeddings()
    elif name == 'product_embeddings_all':
        ds = _product_embeddings_all()
    elif name in ['labeled_queries', 'labeled_query_products', 'ideal_top_10']:
        if 'queries' not in globals():
            globals()['queries'] = _queries()
        if 'labels' not in globals():
            globals()['labels'] = _labels()
        if 'products' not in globals():
            globals()['products'] = _corpus()
        queries = globals()['queries']
        labels = globals()['labels']
        products = globals()['products']
        labeled_queries = queries.merge(labels, how='left', on='query_id')
        labeled_query_products = labeled_queries.merge(products, how='left', on='doc_id')
        ideal_top_10 = _ideal10(products, labeled_queries)
        globals()['labeled_queries'] = labeled_queries
        globals()['labeled_query_products'] = labeled_query_products
        globals()['ideal_top_10'] = ideal_top_10
        if name == 'labeled_queries':
            ds = labeled_queries
        elif name == 'labeled_query_products':
            ds = labeled_query_products
        elif name == 'ideal_top_10':
            ds = ideal_top_10

    globals()[name] = ds
    return ds


def rel_attribute(query_products=None, grade=2, column='category'):
    """Relevant categories in the labeled data useful for ground truth of different attributes."""
    if query_products is None:
        # Call module to get labeled_query_products
        query_products = getattr(sys.modules[__name__], "labeled_query_products")
    return query_products[query_products['grade'] == 2].groupby(['query', column])[column].count().sort_values(ascending=False)
