from cheat_at_search.strategy import BM25Search
from cheat_at_search.wands_data import products, queries, ideal_top_10
from cheat_at_search.eval import grade_results
from cheat_at_search.data_dir import ensure_data_subdir
from cheat_at_search.logger import log_to_stdout
import pandas as pd


logger = log_to_stdout(logger_name="search")


def run_strategy(strategy):

    results = strategy.search_all(queries)
    graded = grade_results(
        results,
        max_grade=2,
        k=10,
    )

    idcg = graded['idcg'].iloc[0]
    dcgs = graded.groupby(["query", 'query_id'])["discounted_gain"].sum().sort_values(ascending=False).rename('dcg')
    ndcgs = dcgs / idcg
    ndcgs = ndcgs.rename('ndcg')

    graded = graded.merge(dcgs, on=['query', 'query_id'])
    graded = graded.merge(ndcgs, on=['query', 'query_id'])

    return graded


def ndcgs(graded):
    return graded.groupby('query')['ndcg'].mean().sort_values(ascending=False)


def ndcg_delta(variant_graded, baseline_graded):
    variant_ndcgs = ndcgs(variant_graded)
    baseline_ndcgs = ndcgs(baseline_graded)
    delta = variant_ndcgs - baseline_ndcgs
    delta = delta[delta != 0]
    return delta.sort_values(ascending=False)


def vs_ideal(graded):
    cols = ['query_id', 'query', 'rank', 'product_id', 'product_name', 'grade', 'dcg', 'ndcg']
    graded_view = graded[cols].rename(
        columns={'product_id': 'product_id_actual', 'product_name': 'product_name_actual'}
    )

    sxs = ideal_top_10.merge(graded_view,
                             how='left',
                             left_on=['query_id', 'query', 'ideal_rank'],
                             right_on=['query_id', 'query', 'rank'])
    return sxs


def run_bm25():
    try:
        bm25_results_path = ensure_data_subdir('bm25_results')
        graded_bm25 = pd.read_pickle(bm25_results_path / 'graded_bm25.pkl')
        return graded_bm25
    except Exception:
        logger.warning("BM25 results not found, running BM25 search strategy.")
        bm25_results_path = ensure_data_subdir('bm25_results')
        bm25 = BM25Search(products)
        graded_bm25 = run_strategy(bm25)
        graded_bm25.to_pickle(bm25_results_path / 'graded_bm25.pkl')
        return graded_bm25


graded_bm25 = run_bm25()
