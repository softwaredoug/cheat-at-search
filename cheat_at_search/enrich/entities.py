import numpy as np
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingModel(Protocol):
    def encode(self, text: str, **kwargs) -> np.ndarray:
        ...


class Entities:
    """Track a set of entities and resolve new ones via vector similarity."""

    def __init__(self, model: EmbeddingModel):
        self.names: list[str] = []
        self._embeddings: np.ndarray | None = None
        self.model = model

    def add_entity(self, name: str):
        self.names.append(name)
        embedding = np.asarray(self.model.encode(name))
        if self._embeddings is None:
            self._embeddings = embedding.reshape(1, -1)
        else:
            self._embeddings = np.vstack([self._embeddings, embedding])

    def __len__(self) -> int:
        return len(self.names)

    def most_similar(self, name: str, top_k: int = 5, threshold: float = 0.95) -> list[str]:
        """Return stored entities most similar to name, above threshold."""
        if not self.names:
            return []
        embedding = np.asarray(self.model.encode(name))
        similarity = np.dot(self._embeddings, embedding)
        top_k = min(top_k, len(self.names))
        top_k_indices = np.argsort(similarity)[-top_k:][::-1]
        return [
            self.names[i]
            for i in top_k_indices
            if similarity[i] > threshold
        ]
