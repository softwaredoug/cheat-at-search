from cheat_at_search.wands_data import labeled_queries
import pandas as pd
import numpy as np


def idcg_max(max_grade = 2, k=10):
    """IDCG assuming max label at each location."""
    return sum([(((2 ** max_grade) - 1)  / np.log2(2 ** i))
                for i in range(1, k + 1)])


def grade_results(search_results: pd.DataFrame, k=10) -> pd.DataFrame:
    search_results = search_results[search_results['rank'] <= k]
    graded_results = search_results.merge(labeled_queries, on=['query_id', 'query', 'product_id'], how='left')
    graded_results['grade'].fillna(0, inplace=True)
    graded_results['discount'] = 1 / np.log2(2 ** graded_results['rank'])
    graded_results['discounted_gain'] = ((2 ** graded_results['grade']) - 1) * graded_results['discount']
    graded_results['idcg'] = idcg_max(k)
    return graded_results
