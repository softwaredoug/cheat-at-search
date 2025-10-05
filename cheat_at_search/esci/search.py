from cheat_at_search.esci.data import queries, docs, judgments
from cheat_at_search.esci.baseline import BM25Search
import pandas as pd


def _grade_results(search_results: pd.DataFrame, max_grade=2, k=10) -> pd.DataFrame:
    """Grade search results based on the labeled queries."""
    import pdb; pdb.set_trace()
    pass


def run_strategy(strategy, queries=queries):

    results = strategy.search_all(queries)
    graded = _grade_results(
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


if __name__ == "__main__":
    bm25 = BM25Search(docs.sample(1000, random_state=1))
    results = run_strategy(bm25)


