import logging
import random
import time
from hashlib import sha256
from typing import Any, Mapping, Optional

from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient

from cheat_at_search.data_dir import key_for_provider

from .enrich_client import DecisionEnrichClient


logger = logging.getLogger("cheat_at_search")


class JevEnrichClient(DecisionEnrichClient):
    """Make constrained classification decisions using Typesafe's Jev API."""

    MAX_RETRIES = 2
    INITIAL_BACKOFF_SECONDS = 0.5
    MAX_BACKOFF_SECONDS = 5.0

    def __init__(self, model: str = "jev-latest", client: Optional[TypeSafeClient] = None,
                 probability_threshold: float = 0.0, confidence_threshold: float = 0.0):
        super().__init__(
            probability_threshold=probability_threshold,
            confidence_threshold=confidence_threshold,
        )
        self.model = model
        self.client = client if client is not None else TypeSafeClient(
            model=model,
            api_key=key_for_provider("typesafe"),
        )

    def score(self, state: Any, question: str, choices: Mapping[str, Optional[str]],
              force_refresh: bool = False) -> tuple[dict[str, float], float]:
        questions = {
            "decision": Choice(
                instructions=question,
                criteria=dict(choices),
            )
        }
        sdk_retry_policy = RetryPolicy(max_retries=0)
        for retry in range(self.MAX_RETRIES + 1):
            try:
                response = self.client.system_one(
                    state=state,
                    questions=questions,
                    retry=sdk_retry_policy,
                )
                break
            except Exception as error:
                if not self._is_retryable_error(error) or retry == self.MAX_RETRIES:
                    raise

                backoff_cap = min(
                    self.INITIAL_BACKOFF_SECONDS * (2 ** retry),
                    self.MAX_BACKOFF_SECONDS,
                )
                delay = random.uniform(0, backoff_cap)
                logger.warning(
                    "TypeSafe request failed with %s; retrying in %.2fs (request_id=%s)",
                    getattr(error, "status", type(error).__name__),
                    delay,
                    getattr(error, "request_id", None),
                )
                time.sleep(delay)

        answer = response.choices["decision"]
        return dict(answer.probabilities), answer.confidence

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        """Return whether an API failure is plausibly transient."""
        status = getattr(error, "status", None)
        if status is not None:
            return status in {408, 425, 429, 499} or 500 <= status < 600
        return isinstance(error, (ConnectionError, TimeoutError))

    def str_hash(self) -> str:
        """Return a stable hash for this Jev model configuration."""
        return sha256(
            f"jev:{self.model}:{self.probability_threshold}:{self.confidence_threshold}".encode()
        ).hexdigest()
