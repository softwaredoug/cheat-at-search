from cheat_at_search.wands_data import labeled_queries
import pandas as pd
import numpy as np


def idcg_max(max_grade=2, k=10):
    """IDCG assuming max label at each location is max_grade."""
    rank_discounts = 1 / np.log2(2 ** np.arange(1, k + 1))
    numerator = (2**max_grade) - 1
    gains = rank_discounts * numerator
    return np.sum(gains)


def grade_results(search_results: pd.DataFrame, max_grade=2, k=10) -> pd.DataFrame:
    """Grade search results based on the labeled queries."""
    search_results = search_results[search_results['rank'] <= k]
    graded_results = search_results.merge(labeled_queries, on=['query_id', 'query', 'product_id'], how='left')
    graded_results['grade'].fillna(0, inplace=True)
    rank_discounts = 1 / np.log2(2 ** graded_results['rank'])
    graded_results['discounted_gain'] = ((2 ** graded_results['grade']) - 1) * rank_discounts
    graded_results['idcg'] = idcg_max(max_grade=max_grade, k=k)
    return graded_results
