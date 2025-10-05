from cheat_at_search.strategy.strategy import SearchStrategy
from cheat_at_search.wands.data import labeled_queries
from searcharray import SearchArray
from cheat_at_search.tokenizers import snowball_tokenizer
import numpy as np
from cheat_at_search.logger import log_to_stdout


logger = log_to_stdout(logger_name="wands_baseline")


class BM25Search(SearchStrategy):
    def __init__(self, products,
                 name_boost=9.3,
                 description_boost=4.1):
        super().__init__(products)
        self.index = products
        self.name_boost = name_boost
        self.description_boost = description_boost
        self.index['product_name_snowball'] = SearchArray.index(
            products['product_name'], snowball_tokenizer)
        self.index['product_description_snowball'] = SearchArray.index(
            products['product_description'], snowball_tokenizer)

    def search(self, query, k=10):
        """Dumb baseline lexical search"""
        tokenized = snowball_tokenizer(query)
        bm25_scores = np.zeros(len(self.index))
        for token in tokenized:
            bm25_scores += self.index['product_name_snowball'].array.score(token) * self.name_boost
            bm25_scores += self.index['product_description_snowball'].array.score(
                token) * self.description_boost
        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]
        return top_k, scores


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

        top_k = sorted['product_id'].tolist()
        scores = sorted['grade'].values.tolist()

        return top_k, scores
