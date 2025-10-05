import pandas as pd
import numpy as np


def idcg_max(max_grade=2, k=10):
    """IDCG assuming max label at each location is max_grade."""
    rank_discounts = 1 / np.log2(2 ** np.arange(1, k + 1))
    numerator = (2**max_grade) - 1
    gains = rank_discounts * numerator
    return np.sum(gains)


def grade_results(judgments: pd.DataFrame,
                  search_results: pd.DataFrame,
                  max_grade=None,
                  k=10) -> pd.DataFrame:
    """Grade search results based on the labeled queries."""
    search_results = search_results[search_results['rank'] <= k]
    assert 'doc_id' in judgments.columns, "judgments must have a 'doc_id' column"
    assert 'doc_id' in search_results.columns, "search_results must have a 'doc_id' column"
    if not max_grade:
        max_grade = judgments['grade'].max()
    graded_results = search_results.merge(judgments, on=['query_id', 'query', 'doc_id'], how='left')
    graded_results['grade'] = graded_results['grade'].fillna(0)
    rank_discounts = 1 / np.log2(2 ** graded_results['rank'])
    graded_results['discounted_gain'] = ((2 ** graded_results['grade']) - 1) * rank_discounts
    graded_results['idcg'] = idcg_max(max_grade=max_grade, k=k)
    return graded_results
