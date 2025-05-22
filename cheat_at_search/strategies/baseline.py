from searcharray import SearchArray
from cheat_at_search.tokenizers import snowball_tokenizer
from cheat_at_search.wands_data import products, queries
from cheat_at_search.strategies.strategy import SearchStrategy
from cheat_at_search.eval import grade_results
import numpy as np
import pandas as pd


class WandsSearch(SearchStrategy):
    def __init__(self, products):
        super().__init__(products)
        self.index = products
        self.index['product_name_snowball'] = SearchArray.index(
            products['product_name'], snowball_tokenizer)
        self.index['product_description_snowball'] = SearchArray.index(
            products['product_description'], snowball_tokenizer)

    def search(self, query, k=10):
        """Dumb baseline lexical search"""
        tokenized = snowball_tokenizer(query)
        bm25_scores = np.zeros(len(self.index))
        for token in tokenized:
            bm25_scores += self.index['product_name_snowball'].array.score(token)
            bm25_scores += self.index['product_description_snowball'].array.score(
                token)
        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]
        return top_k, scores


if __name__ == '__main__':
    search = WandsSearch(products)
    all_results = search.search_all(queries)
    graded = grade_results(all_results)
    dcgs = graded.groupby(['query', 'query_id'])['discounted_gain'].sum().sort_values(ascending=False)
    print(dcgs)

    import pdb; pdb.set_trace()
