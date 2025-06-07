from searcharray import SearchArray
from cheat_at_search.tokenizers import snowball_tokenizer
from cheat_at_search.strategy.strategy import SearchStrategy
import numpy as np


class BM25Search(SearchStrategy):
    def __init__(self, products,
                 name_boost=5,
                 description_boost=1):
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
