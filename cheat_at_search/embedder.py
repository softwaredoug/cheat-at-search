import torch
from sentence_transformers import SentenceTransformer
import logging
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd


logger = logging.getLogger(__name__)


class TextEmbedder(ABC):
    @abstractmethod
    def query(self, keywords: str) -> np.ndarray:
        """Embed the query tower."""
        pass

    @abstractmethod
    def document(self, doc_ids: list) -> np.ndarray:
        """Embed the document tower."""
        pass


class SentenceTransformerEmbedder(TextEmbedder):
    """Use a SentenceTransformer model to embed text."""
    def __init__(self, products: pd.DataFrame,
                 model_name="sentence-transformers/all-MiniLM-L6-v2",
                 device=None):
        self.device = device
        if self.device is None:
            self.device = (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        logger.info(f"Using device: {self.device}")
        if self.device == torch.device("cpu"):
            logger.warning("Using CPU, this will be slow")
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        self.products = products

    def query(self, keywords: str) -> np.ndarray:
        """Embed the query tower."""
        return self.model.encode(keywords, convert_to_tensor=False, device=self.device)

    def document(self) -> np.ndarray:
        """Embed the document tower."""
        # Get product name and description for the given doc_ids
        texts = []
        texts = self.products['product_name'] + " " + self.products['product_description']

        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, convert_to_tensor=False, device=self.device)
