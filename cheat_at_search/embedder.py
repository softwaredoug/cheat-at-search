import torch
from sentence_transformers import SentenceTransformer
from cheat_at_search.embedder import TextEmbedder
import logging
import numpy as np


logger = logging.getLogger(__name__)


class TextEmbedder:

    def __init__(self, model_name: str, device=None):
        self.device = device
        if self.device is None:
            self.device = (
                torch.device("mps") if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            )
        logger.info(f"Using device: {self.device}")
        if self.device == torch.device("cpu"):
            logger.warning("Using CPU, this will be slow")
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)

    def __call__(self, texts) -> np.ndarray:
        return self.model.encode(texts, convert_to_tensor=False, device=self.device)
