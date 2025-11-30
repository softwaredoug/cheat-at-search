from cheat_at_search.strategy import BM25Search
from cheat_at_search.eval import grade_results
from cheat_at_search.data_dir import ensure_data_subdir
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.types import Optional
import pandas as pd


logger = log_to_stdout(logger_name="search")


def run_strategy(strategy, judgments,
                 queries: Optional[list[str]] = None,
                 num_queries=None, seed=42,
                 sub_sample_n=None, sub_sample_seed=42):

    available_queries = judgments[['query', 'query_id']].drop_duplicates()
    max_grade = judgments['grade'].max()

    if queries:
        available_queries = available_queries[available_queries['query'].isin(queries)]
        # Check any missing queries
        missing_queries = set(queries) - set(available_queries['query'])
        if missing_queries:
            raise ValueError(f"You asked to search over these queries, but they are missing from judgments: {missing_queries}")
        judgments = judgments[judgments['query_id'].isin(available_queries['query_id'])]

    if num_queries:
        available_queries = available_queries.sample(num_queries, random_state=seed)
        judgments = judgments[judgments['query_id'].isin(available_queries['query_id'])]

        if sub_sample_n:
            if sub_sample_n >= len(available_queries):
                raise ValueError("sub_sample_n must be less than the number of queries after num_queries filtering.")
            available_queries = available_queries.sample(sub_sample_n, random_state=sub_sample_seed)
            judgments = judgments[judgments['query_id'].isin(available_queries['query_id'])]

    results = strategy.search_all(available_queries)
    graded = grade_results(
        judgments,
        results,
        max_grade=max_grade,
        k=10,
    )

    if len(graded):
        idcg = graded['idcg'].iloc[0]
        dcgs = graded.groupby(["query", 'query_id'])["discounted_gain"].sum().sort_values(ascending=False).rename('dcg')
        ndcgs = dcgs / idcg
        ndcgs = ndcgs.rename('ndcg')

        graded = graded.merge(dcgs, on=['query', 'query_id'])
        graded = graded.merge(ndcgs, on=['query', 'query_id'])
    else:
        graded['dcg'] = 0
        graded['ndcg'] = 0

    return graded


def ndcgs(graded):
    return graded.groupby('query')['ndcg'].mean().sort_values(ascending=False)


def ndcg_delta(variant_graded, baseline_graded):
    variant_ndcgs = ndcgs(variant_graded)
    baseline_ndcgs = ndcgs(baseline_graded)
    delta = variant_ndcgs - baseline_ndcgs
    delta = delta[delta != 0]
    return delta.sort_values(ascending=False)


def run_bm25(corpus):
    try:
        bm25_results_path = ensure_data_subdir('bm25_results')
        graded_bm25 = pd.read_pickle(bm25_results_path / 'graded_bm25.pkl')
        return graded_bm25
    except Exception:
        logger.warning("BM25 results not found, running BM25 search strategy.")
        bm25_results_path = ensure_data_subdir('bm25_results')
        bm25 = BM25Search(corpus)
        graded_bm25 = run_strategy(bm25)
        graded_bm25.to_pickle(bm25_results_path / 'graded_bm25.pkl')
        return graded_bm25
