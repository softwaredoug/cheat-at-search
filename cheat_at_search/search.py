from cheat_at_search.strategy import BM25Search
from cheat_at_search.wands_data import products, queries
from cheat_at_search.eval import grade_results


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


bm25 = BM25Search(products)
graded_bm25 = run_strategy(bm25)
graded_bm25
