from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional
from pydantic import BaseModel


class DebugMetaData:
    def __init__(self, model: str,
                 prompt_tokens: int,
                 completion_tokens: int,
                 reasoning_tokens: int,
                 response_id: Optional[str] = None,
                 output: BaseModel = None):
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.reasoning_tokens = reasoning_tokens
        self.total_tokens = prompt_tokens + completion_tokens + reasoning_tokens
        self.response_id = response_id
        self.output = output

    def __repr__(self):
        return f"DebugMetaData(model={self.model}, prompt_tokens={self.prompt_tokens}, completion_tokens={self.completion_tokens}, reasoning_tokens={self.reasoning_tokens}, total_tokens={self.total_tokens}, response_id={self.response_id}, output={self.output})"


class LLMEnrichClient(ABC):

    def __init__(self, response_model: BaseModel):
        self.response_model = response_model

    @abstractmethod
    def enrich(self, prompt: str) -> Optional[BaseModel]:
        pass

    @abstractmethod
    def debug(self, prompt: str) -> Optional[DebugMetaData]:
        pass

    @abstractmethod
    def str_hash(self) -> str:
        pass


def _validate_decision_threshold(name: str, threshold: float) -> float:
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError(f"{name} must be a number between 0 and 1 inclusive.")
    return float(threshold)


class DecisionEnrichClient(ABC):
    """Interface for making a constrained decision about a state."""

    def __init__(self, probability_threshold: float = 0.0, confidence_threshold: float = 0.0):
        self.probability_threshold = _validate_decision_threshold(
            "probability_threshold", probability_threshold
        )
        self.confidence_threshold = _validate_decision_threshold(
            "confidence_threshold", confidence_threshold
        )

    @abstractmethod
    def score(self, state: Any, question: str, choices: Mapping[str, Optional[str]],
              force_refresh: bool = False) -> tuple[dict[str, float], float]:
        """Return the probability for every choice and overall confidence."""
        pass

    def decide(self, state: Any, question: str, choices: Mapping[str, Optional[str]],
               force_refresh: bool = False) -> Optional[str]:
        """Return the most likely choice if its probability and confidence pass thresholds."""
        probabilities, confidence_score = self.score(
            state=state,
            question=question,
            choices=choices,
            force_refresh=force_refresh,
        )
        if not probabilities:
            return None

        choice = max(probabilities, key=probabilities.get)
        if (
            probabilities[choice] < self.probability_threshold
            or confidence_score < self.confidence_threshold
        ):
            return None
        return choice

    @abstractmethod
    def str_hash(self) -> str:
        """Return a stable hash of the client's configuration."""
        pass
