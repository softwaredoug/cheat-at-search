from typing import Optional

import numpy as np
from pydantic import BaseModel

from .enrich import AutoEnricher
from cheat_at_search.embeddings import DEFAULT_MODEL_NAME, load_model


class VocabularyResponse(BaseModel):
    """Structured response containing the value extracted by the LLM."""

    value: str | None = None


class VocabularyEnricher:
    """Resolve an LLM-extracted value to the closest controlled vocabulary item."""

    def __init__(self, model: str, system_prompt: str, vocabulary: list[str],
                 temperature: Optional[float] = None,
                 reasoning_effort: Optional[str] = None,
                 verbosity: Optional[str] = None,
                 embedding_model_name: str = DEFAULT_MODEL_NAME,
                 device: Optional[str] = None):
        if not vocabulary:
            raise ValueError("Vocabulary must contain at least one item.")
        if not all(isinstance(item, str) and item for item in vocabulary):
            raise ValueError("Vocabulary items must be non-empty strings.")

        self.vocabulary = vocabulary
        self.enricher = AutoEnricher(
            model=model,
            system_prompt=system_prompt,
            response_model=VocabularyResponse,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
        )
        self.embedding_model = load_model(embedding_model_name, device=device)
        self.vocabulary_embeddings = self._embed(vocabulary)

    def _embed(self, values: list[str]) -> np.ndarray:
        embeddings = self.embedding_model.encode(values, convert_to_numpy=True)
        embeddings = np.asarray(embeddings, dtype=float)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.where(norms == 0, 1, norms)

    def resolve(self, prompt: str, force_refresh: bool = False) -> Optional[str]:
        """Return the closest vocabulary item for the model's extracted value."""
        response = self.enricher.enrich(prompt, force_refresh=force_refresh)
        if response is None:
            return None

        value = response.value if isinstance(response, VocabularyResponse) else response
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if not isinstance(value, str) or not value.strip():
            return None

        embedding = self._embed([value])[0]
        similarities = self.vocabulary_embeddings @ embedding
        return self.vocabulary[int(np.argmax(similarities))]
