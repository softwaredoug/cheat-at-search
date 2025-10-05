from cheat_at_search.strategy.strategy import SearchStrategy
from cheat_at_search.wands_data import labeled_queries


class BestPossibleResults(SearchStrategy):
    def __init__(self,
                 products):
        super().__init__(products)
        self.index = products

    def search(self, query, k=10):
        """Dumb baseline lexical search, but add a constant boost when
           the desired category or subcategory"""
        # ****
        # Lookup labels for query
        # ****
        query_labels = labeled_queries[labeled_queries['query'] == query].copy()
        sorted = query_labels.sort_values('grade', ascending=False).head(k)

        top_k = sorted['doc_id'].tolist()
        scores = sorted['grade'].values.tolist()

        return top_k, scores
