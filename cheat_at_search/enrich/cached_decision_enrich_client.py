import json
import os
import tempfile
from hashlib import sha256
from threading import Lock, RLock
from time import perf_counter
from typing import Any, Mapping, Optional

from cheat_at_search.data_dir import ensure_data_subdir
from cheat_at_search.logger import log_to_stdout

from .enrich_client import DecisionEnrichClient


logger = log_to_stdout(logger_name="cached_decision_enrich_client")


class CachedDecisionEnrichClient(DecisionEnrichClient):
    """Cache decisions by client configuration and canonical request content."""

    CACHE_KEY_VERSION = 2
    CACHE_FORMAT_VERSION = 2

    def __init__(self, enricher: DecisionEnrichClient, save_interval_seconds: float = 60):
        super().__init__(
            probability_threshold=enricher.probability_threshold,
            confidence_threshold=enricher.confidence_threshold,
        )
        self.enricher = enricher
        self.cache_path = ensure_data_subdir("enrich_cache")
        client_signature = sha256(
            f"{enricher.__class__.__name__}:{enricher.str_hash()}".encode()
        ).hexdigest()
        self.cache_file = os.path.join(
            self.cache_path,
            f"decision_v{self.CACHE_FORMAT_VERSION}_{client_signature}_cache.json",
        )
        self.cache: dict[str, dict[str, Any]] = {}
        self.save_interval_seconds = save_interval_seconds
        self.last_cache_save_time = perf_counter()
        self._cache_lock = RLock()
        self._key_locks: dict[str, Lock] = {}
        self.load_cache()

    def load_cache(self):
        logger.info(f"Loading decision cache from {self.cache_file}")
        if not os.path.exists(self.cache_file):
            self.cache = {}
            return

        try:
            with open(self.cache_file, encoding="utf-8") as cache_file:
                cache = json.load(cache_file)
            if not isinstance(cache, dict) or any(
                not self._is_score_entry(value) for value in cache.values()
            ):
                raise ValueError("Decision cache must map keys to probability and confidence scores.")
            self.cache = cache
            logger.info(f"Loaded {len(self.cache)} decisions from cache.")
        except Exception as error:
            logger.error(f"Error loading decision cache {self.cache_file}: {error}")
            logger.warning("Starting with an empty decision cache.")
            try:
                os.remove(self.cache_file)
            except OSError as remove_error:
                logger.warning(f"Could not remove invalid decision cache: {remove_error}")
            self.cache = {}

    def save_cache(self) -> bool:
        with self._cache_lock:
            cache_snapshot = self.cache.copy()

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.cache_path,
                prefix=f".{os.path.basename(self.cache_file)}.",
                suffix=".tmp",
                delete=False,
            ) as cache_file:
                temporary_path = cache_file.name
                json.dump(cache_snapshot, cache_file, sort_keys=True, separators=(",", ":"))
            os.replace(temporary_path, self.cache_file)
            temporary_path = None
            return True
        except Exception as error:
            logger.error(f"Error saving decision cache {self.cache_file}: {error}")
            return False
        finally:
            if temporary_path is not None:
                try:
                    os.remove(temporary_path)
                except FileNotFoundError:
                    pass
                except OSError as error:
                    logger.warning(f"Could not remove temporary cache file {temporary_path}: {error}")

    def delete_cache(self):
        """Delete the persisted decision cache and clear in-memory entries."""
        with self._cache_lock:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
            self.cache.clear()

    def decision_key(self, state: Any, question: str, choices: Mapping[str, Optional[str]]) -> str:
        """Build a stable key from the client configuration and complete request."""
        request = {
            "cache_key_version": self.CACHE_KEY_VERSION,
            "client_hash": self.enricher.str_hash(),
            "model": getattr(self.enricher, "model", None),
            "probability_threshold": self.probability_threshold,
            "confidence_threshold": self.confidence_threshold,
            "state": state,
            "question": question,
            "choices": dict(choices),
        }
        serialized_request = json.dumps(
            request,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return sha256(serialized_request.encode("utf-8")).hexdigest()

    def _key_lock(self, key: str) -> Lock:
        with self._cache_lock:
            return self._key_locks.setdefault(key, Lock())

    @staticmethod
    def _is_score_entry(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        probabilities = value.get("probabilities")
        confidence_score = value.get("confidence_score")
        if not isinstance(probabilities, dict) or not probabilities:
            return False
        if any(
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not 0.0 <= probability <= 1.0
            for probability in probabilities.values()
        ):
            return False
        return (
            isinstance(confidence_score, (int, float))
            and not isinstance(confidence_score, bool)
            and 0.0 <= confidence_score <= 1.0
        )

    def score(self, state: Any, question: str, choices: Mapping[str, Optional[str]],
              force_refresh: bool = False) -> tuple[dict[str, float], float]:
        key = self.decision_key(state=state, question=question, choices=choices)
        with self._cache_lock:
            if not force_refresh and key in self.cache:
                score = self.cache[key]
                return dict(score["probabilities"]), score["confidence_score"]

        key_lock = self._key_lock(key)
        try:
            with key_lock:
                with self._cache_lock:
                    if not force_refresh and key in self.cache:
                        score = self.cache[key]
                        return dict(score["probabilities"]), score["confidence_score"]

                probabilities, confidence_score = self.enricher.score(
                    state=state,
                    question=question,
                    choices=choices,
                    force_refresh=force_refresh,
                )
                score = {
                    "probabilities": dict(probabilities),
                    "confidence_score": confidence_score,
                }
                with self._cache_lock:
                    self.cache[key] = score
        finally:
            with self._cache_lock:
                if self._key_locks.get(key) is key_lock:
                    del self._key_locks[key]

        if perf_counter() - self.last_cache_save_time >= self.save_interval_seconds:
            if self.save_cache():
                self.last_cache_save_time = perf_counter()
        return dict(score["probabilities"]), score["confidence_score"]

    def str_hash(self) -> str:
        return self.enricher.str_hash()
