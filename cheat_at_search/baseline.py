from searcharray import SearchArray
from cheat_at_search.tokenizers import snowball_tokenizer
from cheat_at_search.wands_data import products, queries
import numpy as np
import pandas as pd


class WandsSearch:
    def __init__(self, products):
        self.index = products
        self.index['product_name_snowball'] = SearchArray.index(
            products['product_name'], snowball_tokenizer)
        self.index['product_description_snowball'] = SearchArray.index(
            products['product_description'], snowball_tokenizer)

    def search(self, queries: pd.DataFrame, k=10) -> pd.DataFrame:
        """Dumb baseline lexical search"""
        all_results = []
        for query in queries['query']:
            tokenized = snowball_tokenizer(query)
            bm25_scores = np.zeros(len(self.index))
            for token in tokenized:
                bm25_scores += self.index['product_name_snowball'].array.score(token)
                bm25_scores += self.index['product_description_snowball'].array.score(
                    token)
            top_k = np.argsort(-bm25_scores)[:k]
            top_k_products = self.index.iloc[top_k].copy()
            top_k_products['score'] = bm25_scores[top_k]
            top_k_products['query'] = query
            all_results.append(top_k_products)

        return pd.concat(all_results)


if __name__ == '__main__':
    search = WandsSearch(products())
    all_results = search.search(queries())
    import pdb; pdb.set_trace()
