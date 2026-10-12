from hashlib import sha256
from typing import Any, Mapping, Optional

from typesafe_sdk import Choice, TypeSafeClient

from .enrich_client import DecisionEnrichClient


class JevEnrichClient(DecisionEnrichClient):
    """Make constrained classification decisions using Typesafe's Jev API."""

    def __init__(self, model: str = "jev-latest", client: Optional[TypeSafeClient] = None,
                 probability_threshold: float = 0.0, confidence_threshold: float = 0.0):
        super().__init__(
            probability_threshold=probability_threshold,
            confidence_threshold=confidence_threshold,
        )
        self.model = model
        self.client = client if client is not None else TypeSafeClient(model=model)

    def score(self, state: Any, question: str, choices: Mapping[str, Optional[str]],
              force_refresh: bool = False) -> tuple[dict[str, float], float]:
        response = self.client.system_one(
            state=state,
            questions={
                "decision": Choice(
                    instructions=question,
                    criteria=dict(choices),
                )
            },
        )
        answer = response.choices["decision"]
        return dict(answer.probabilities), answer.confidence

    def str_hash(self) -> str:
        """Return a stable hash for this Jev model configuration."""
        return sha256(
            f"jev:{self.model}:{self.probability_threshold}:{self.confidence_threshold}".encode()
        ).hexdigest()
