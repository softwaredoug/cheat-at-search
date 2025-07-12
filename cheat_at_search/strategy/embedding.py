import numpy as np
from cheat_at_search.strategy.strategy import SearchStrategy
from cheat_at_search.logger import log_to_stdout
import logging

logger = logging.getLogger(__name__)


class EmbeddingSearch(SearchStrategy):
    def __init__(self, products, embedder=None):
        self.embedder = embedder
        logger.info(f"Embedding {len(products)} products")
        self.index = self.embedder.document()
        logger.info(f"Embedding completed, index shape: {self.index.shape}")
        super().__init__(products)

    def search(self, query, k=10):
        query_embedding = self.embedder.query(query)
        scores = np.dot(self.index, query_embedding)
        sorted_indices = np.argsort(scores)[::-1][:k]
        similarities = scores[sorted_indices]
        return sorted_indices[:k], similarities[:k]
