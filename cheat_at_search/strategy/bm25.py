from searcharray import SearchArray
from cheat_at_search.tokenizers import snowball_tokenizer
from cheat_at_search.strategy.strategy import SearchStrategy
import numpy as np
from cheat_at_search.logger import log_to_stdout


logger = log_to_stdout(logger_name="search")


class BM25Search(SearchStrategy):
    def __init__(self, products,
                 name_boost=9.3,
                 description_boost=4.1):
        super().__init__(products)
        self.index = products
        self.name_boost = name_boost
        self.description_boost = description_boost
        self.index['title_snowball'] = SearchArray.index(
            products['title'], snowball_tokenizer)
        self.index['description_snowball'] = SearchArray.index(
            products['description'], snowball_tokenizer)

    def search(self, query, k=10):
        """Dumb baseline lexical search"""
        tokenized = snowball_tokenizer(query)
        bm25_scores = np.zeros(len(self.index))
        for token in tokenized:
            bm25_scores += self.index['title_snowball'].array.score(token) * self.name_boost
            bm25_scores += self.index['description_snowball'].array.score(
                token) * self.description_boost
        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]
        return top_k, scores
