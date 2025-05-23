import pandas as pd
import numpy as np


class SearchStrategy:

    def __init__(self, products, top_k=5):
        self.products = products
        self.top_k = top_k

    def search_all(self, queries, k=10):
        all_results = []
        for _, query_row in queries.iterrows():
            top_k, scores = self.search(query_row['query'], k)
            query_id = query_row['query_id']
            ranks = np.arange(len(top_k)) + 1
            top_k_products = self.products.iloc[top_k].copy()
            top_k_products['score'] = scores
            top_k_products['query'] = query_row['query']
            top_k_products['query_id'] = query_id
            top_k_products['rank'] = ranks
            all_results.append(top_k_products)
        return pd.concat(all_results)

    def search(self, query, k):
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses should implement this method.")
