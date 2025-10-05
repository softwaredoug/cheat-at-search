import pandas as pd
import numpy as np
from searcharray import SearchArray
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class SearchStrategy:

    def __init__(self, products, top_k=5,
                 workers=1):
        self.products = products
        self.top_k = top_k
        self.workers = workers

    def search_all(self, queries, k=10):
        all_results = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {}
            for _, query_row in queries.iterrows():
                future = executor.submit(self.search, query_row['query'], k)
                futures[future] = query_row

            for future in tqdm(as_completed(futures), total=len(futures), desc="Searching"):
                query_row = futures[future]
                top_k, scores = future.result()
                query_id = query_row['query_id']
                ranks = np.arange(len(top_k)) + 1
                search_array_cols = [
                    col for col in self.products.columns
                    if isinstance(self.products[col].array, SearchArray)
                ]
                # Ensure we drop only SearchArray columns
                top_k_products = self.products.drop(columns=search_array_cols, errors='ignore')
                top_k_products = top_k_products.iloc[top_k].copy()
                top_k_products['score'] = scores
                top_k_products['query'] = query_row['query']
                top_k_products['query_id'] = query_id
                top_k_products['rank'] = ranks
                # Remove any columns where .array is SearchArray

                all_results.append(top_k_products)
        return pd.concat(all_results)

    def search(self, query, k):
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses should implement this method.")
