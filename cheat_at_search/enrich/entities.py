import numpy as np
from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingModel(Protocol):
    def encode(self, text: str, **kwargs) -> np.ndarray:
        ...


class Entities:
    """Track a set of entities and resolve new ones via vector similarity."""

    def __init__(self, model: EmbeddingModel):
        self.names: set[str] = set()
        self._names_in_order: list[str] = []
        self._embeddings: np.ndarray | None = None
        self.model = model

    def add(self, names: str | Sequence[str]):
        if isinstance(names, str):
            names_to_add = [names] if names not in self.names else []
        else:
            seen: set[str] = set()
            names_to_add = []
            for name in names:
                if name in seen or name in self.names:
                    continue
                seen.add(name)
                names_to_add.append(name)

        for name in names_to_add:
            self.names.add(name)
            self._names_in_order.append(name)
            embedding = np.asarray(self.model.encode(name))
            if self._embeddings is None:
                self._embeddings = embedding.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, embedding])

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self.names

    def __repr__(self) -> str:
        return f"Entities(names={self._names_in_order!r})"

    def most_similar(self, name: str, top_k: int = 5, threshold: float = 0.95) -> list[str]:
        """Return stored entities most similar to name, above threshold."""
        if not self._names_in_order:
            return []
        embedding = np.asarray(self.model.encode(name))
        similarity = np.dot(self._embeddings, embedding)
        top_k = min(top_k, len(self._names_in_order))
        top_k_indices = np.argsort(similarity)[-top_k:][::-1]
        return [
            self._names_in_order[i]
            for i in top_k_indices
            if similarity[i] > threshold
        ]
