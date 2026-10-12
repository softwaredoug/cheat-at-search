from cheat_at_search.logger import log_to_stdout
from .cached_enrich_client import CachedEnrichClient
from .cached_decision_enrich_client import CachedDecisionEnrichClient
from .enrich_client import DebugMetaData, _validate_decision_threshold
from .jev_enrich_client import JevEnrichClient
from .openai_enrich_client import OpenAIEnricher
from collections.abc import Sequence
from typing import Any, Mapping, Optional
from pydantic import BaseModel
from typing import Tuple
import pandas as pd
from tqdm import tqdm
from searcharray import SearchArray
from concurrent.futures import ThreadPoolExecutor, as_completed


logger = log_to_stdout(logger_name="query_parser")


class AutoLLMEnricher:
    """Either serial cached or batch enriched, depending on the context."""

    def __init__(self, model: str, system_prompt: str, response_model: type[BaseModel],
                 temperature: Optional[float] = None,
                 reasoning_effort: Optional[str] = None,
                 verbosity: Optional[str] = None):
        self.provider = model.split('/')[0]
        self.system_prompt = system_prompt
        if self.provider == 'openai':
            self.enricher = OpenAIEnricher(response_model=response_model,
                                           model=model,
                                           system_prompt=self.system_prompt,
                                           temperature=temperature,
                                           reasoning_effort=reasoning_effort,
                                           verbosity=verbosity)
            self.cached_enricher = CachedEnrichClient(self.enricher)
        else:
            raise ValueError(f"Provider {self.provider} is not supported. Supported provider: 'openai'")

    def enrich(self, prompt: str, force_refresh: bool = False) -> BaseModel:
        """Enrich a single prompt, now, and cache the result."""
        return self.cached_enricher.enrich(prompt, force_refresh=force_refresh)

    def enrich_all(self, prompts: list[str], workers=5, batch_size=100) -> list[BaseModel]:
        """Enrich a list of prompts, using multiple threads, and cache the results."""
        results = [None] * len(prompts)
        results_len = len(results)

        if not isinstance(prompts, list):
            raise ValueError(f"Prompts must be a list of strings. Found type {type(prompts)}")

        for prompt in prompts:
            if not isinstance(prompt, str):
                raise ValueError(f"All prompts must be strings. Found prompt of type {type(prompt)}")

        def enrich_one(idx, prompt):
            return idx, self.cached_enricher.enrich(prompt)

        logger.info(f"Enriching {len(prompts)} prompts with {workers} workers and batch size {batch_size}")
        error_idxes = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            with tqdm(total=len(prompts), desc="Enriching prompts") as pbar:
                for i in range(0, len(prompts), batch_size):
                    result_idx_base = i
                    batch_prompts = prompts[i:i + batch_size]

                    futures = {
                        executor.submit(enrich_one, idx + result_idx_base, prompt): idx + result_idx_base
                        for idx, prompt in enumerate(batch_prompts)
                    }

                    fail = False
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            res_idx, enriched_data = future.result()
                        except Exception as e:
                            logger.debug(f"(Forgiven) Error enriching prompt at index {idx}: {str(e)}")
                            error_idxes.append(idx)
                            continue

                        if res_idx > len(prompts):
                            logger.error(f"Result index {res_idx} out of bounds for prompts of length {len(prompts)}")
                            fail = True
                            break
                        if len(results) != results_len:
                            logger.error("Results list size changed during enrichment, possible concurrency issue.")
                            fail = True
                            break
                        results[res_idx] = enriched_data
                    if fail:
                        raise ValueError("Enrichment failed due to errors in processing.")
                    pbar.update(len(batch_prompts))
        self.cached_enricher.save_cache()
        for idx in error_idxes:
            results[idx] = enrich_one(idx, prompts[idx])[1]
            logger.warning(f"Enrichment failed for prompt at index {idx}, setting result to None.")
        self.cached_enricher.save_cache()
        return results

    def debug(self, prompt: str) -> Optional[DebugMetaData]:
        """Enrich a single prompt, now, and return debug metadata."""
        return self.cached_enricher.debug(prompt)

    def get_num_tokens(self, prompt: str) -> Tuple[int, int]:
        """Get the number of tokens for a prompt (runs directly, does not cache)."""
        return self.cached_enricher.enricher.get_num_tokens(prompt)

    def delete_cache(self):
        """Delete the cache."""
        self.cached_enricher.delete_cache()

    @property
    def response_model(self):
        """Return the output class of the enricher."""
        return self.enricher.response_model


AutoEnricher = AutoLLMEnricher


class AutoDecisionEnricher:
    """Run one or many Jev decisions using a fixed set of choices."""

    def __init__(self, choices: Optional[Mapping[str, Optional[str]]] = None,
                 model: str = "jev/jev-latest",
                 probability_threshold: float = 0.0,
                 confidence_threshold: float = 0.0):
        provider, separator, provider_model = model.partition('/')
        if not separator or not provider_model or provider != "jev":
            raise ValueError("AutoDecisionEnricher requires a Jev model such as 'jev/jev-latest'.")
        if not choices:
            raise ValueError("choices must contain at least one decision option.")

        self.choices = dict(choices)
        self.model = provider_model
        self.probability_threshold = _validate_decision_threshold(
            "probability_threshold", probability_threshold
        )
        self.confidence_threshold = _validate_decision_threshold(
            "confidence_threshold", confidence_threshold
        )
        self.enricher = JevEnrichClient(
            model=provider_model,
            probability_threshold=self.probability_threshold,
            confidence_threshold=self.confidence_threshold,
        )
        self.cached_enricher = CachedDecisionEnrichClient(self.enricher)

    def decide(self, state: Any, question: str, force_refresh: bool = False) -> Optional[str]:
        """Return the selected choice for one state and question."""
        return self.cached_enricher.decide(
            state=state,
            question=question,
            choices=self.choices,
            force_refresh=force_refresh,
        )

    def score(self, state: Any, question: str, force_refresh: bool = False) -> tuple[dict[str, float], float]:
        """Return choice probabilities and confidence for one state and question."""
        return self.cached_enricher.score(
            state=state,
            question=question,
            choices=self.choices,
            force_refresh=force_refresh,
        )

    def decide_all(self, states: Sequence[Any], questions: str | Sequence[str], workers: int = 5,
                   force_refresh: bool = False) -> list[Optional[str]]:
        """Decide for multiple states, preserving input order.

        A single question is applied to every state. A sequence of questions
        must have one entry per state.
        """
        if isinstance(states, (str, bytes)) or not isinstance(states, Sequence):
            raise ValueError("states must be a sequence of state values.")
        states = list(states)
        if isinstance(questions, str):
            questions = [questions] * len(states)
        else:
            if not isinstance(questions, Sequence):
                raise ValueError("questions must be a string or a sequence of question strings.")
            questions = list(questions)
            if len(questions) != len(states):
                raise ValueError("questions must have the same length as states.")

        if any(not isinstance(question, str) for question in questions):
            raise ValueError("All questions must be strings.")

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                return list(
                    executor.map(
                        lambda state_and_question: self.decide(
                            state=state_and_question[0],
                            question=state_and_question[1],
                            force_refresh=force_refresh,
                        ),
                        zip(states, questions),
                    )
                )
        finally:
            self.cached_enricher.save_cache()

    def delete_cache(self):
        """Delete cached decisions."""
        self.cached_enricher.delete_cache()


class DataframeEnricher:
    """Enrich a dataframe of rows."""

    def __init__(self, enricher: AutoLLMEnricher, prompt_fn, attrs=None,
                 separator: str = " sep "):
        self.enricher = enricher
        self.prompt_fn = prompt_fn
        self.separator = separator
        if attrs is None:  # Inferred from the BaseModel
            output_cls = enricher.response_model
            attrs = output_cls.model_fields.keys()
            attrs = set(attrs)  # Ensure unique attributes
            # Get properties too
            attrs = attrs.union(attr for attr in dir(output_cls) if isinstance(getattr(output_cls, attr), property))
            # Remove any beginning with __
            attrs = {attr for attr in attrs if not attr.startswith('__')}
            # Remove pydantic internal attributes
            attrs = {attr for attr in attrs if attr not in
                     ["model_fields_set", "model_extra", "model_config", "model_json_schema"]}
            logger.info(f"Enriching rows with attributes: {attrs}")
        self.attrs = attrs

    def _slice_out_searcharray_cols(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Slice out columns that are SearchArray columns."""
        searcharray_cols = [
            col for col in rows.columns
            if isinstance(rows[col].array, SearchArray)
        ]
        return rows.drop(columns=searcharray_cols, errors='ignore')

    def enrich_one(self, row: dict):
        prompt = self.prompt_fn(row)
        return self.enricher.enrich(prompt)

    def enrich_all(self, rows: pd.DataFrame, workers=5, batch_size=100) -> pd.DataFrame:
        rows = self._slice_out_searcharray_cols(rows)

        # Generate prompts for all rows
        prompts = [self.prompt_fn(row.to_dict()) for _, row in rows.iterrows()]

        # Use AutoEnricher's enrich_all method
        enriched_results = self.enricher.enrich_all(prompts, workers=workers, batch_size=batch_size)

        # Post-process results back into the dataframe
        for idx, (_, row) in enumerate(rows.iterrows()):
            enriched_data = enriched_results[idx]
            if enriched_data:
                for attr in self.attrs:
                    value = getattr(enriched_data, attr, None)
                    # If iterable, join
                    if isinstance(value, (list, tuple)):
                        value = self.separator.join(map(str, value))
                    elif value is None:
                        value = ""
                    rows.at[row.name, attr] = value if value is not None else ""
            else:
                logger.warning(f"Enrichment failed for row {row.get('doc_id', 'unknown')}")
                for attr in self.attrs:
                    rows.at[row.name, attr] = ""

        return rows
