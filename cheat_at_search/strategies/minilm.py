import numpy as np
from cheat_at_search.embedder import TextEmbedder
from cheat_at_search.strategies.strategy import SearchStrategy


class MiniLMSearch(SearchStrategy):
    def __init__(self, products):
        self.embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.index = self.embedder(products['product_name'] + " " + products['product_description'])
        super().__init__(products)

    def search(self, query, k=10):
        query_embedding = self.embedder(query)
        scores = np.dot(self.index, query_embedding)
        sorted_indices = np.argsort(scores)[::-1][:k]
        similarities = scores[sorted_indices]
        return sorted_indices[:k], similarities[:k]
