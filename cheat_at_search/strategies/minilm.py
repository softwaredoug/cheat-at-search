from cheat_at_search.tokenizers import snowball_tokenizer
from cheat_at_search.wands_data import products, queries
from cheat_at_search.eval import grade_results
import numpy as np
import pandas as pd
from cheat_at_search.embedder import TextEmbedder


class MiniLMSearch:
    def __init__(self, products):
        self.embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.index = self.embedder(products['product_name'] + " " + products['product_description'])
        self.products = products

    def search(self, queries):
        all_results = []
        for _, query_row in queries.iterrows():
            query = query_row['query']
            query_id = query_row['query_id']

            query_embedding = self.embedder(query)
            scores = np.dot(self.index, query_embedding)
            sorted_indices = np.argsort(scores)[::-1]
            return sorted_indices[:10]
