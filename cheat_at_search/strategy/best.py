from cheat_at_search.strategy.strategy import SearchStrategy


class BestPossibleResults(SearchStrategy):
    def __init__(self,
                 corpus,
                 judgments):
        super().__init__(corpus)
        self.index = corpus
        self.judgments = judgments

    def search(self, query, k=10):
        """Dumb baseline lexical search, but add a constant boost when
           the desired category or subcategory"""
        # ****
        # Lookup labels for query
        # ****
        query_labels = self.judgments[self.judgments['query'] == query].copy()
        sorted = query_labels.sort_values('grade', ascending=False).head(k)

        top_k = []
        scores = []
        for _, row in sorted.iterrows():
            doc_id = row['doc_id']
            doc = self.index[self.index['doc_id'] == doc_id]
            if doc.empty:
                continue
            idx = doc.index[0]
            scores.append(row['grade'])
            top_k.append(idx)

        return top_k, scores
