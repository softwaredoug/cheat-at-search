from cheat_at_search.logger import log_to_stdout
from .cached_enrich_client import CachedEnrichClient
from .enrich_client import DebugMetaData
from .openai_enrich_client import OpenAIEnricher
from .google_enrich_client import GoogleEnrichClient
from typing import Optional
from pydantic import BaseModel
from typing import Tuple
import pandas as pd
from tqdm import tqdm
from searcharray import SearchArray
from concurrent.futures import ThreadPoolExecutor, as_completed


logger = log_to_stdout(logger_name="query_parser")


class AutoEnricher:
    """Either serial cached or batch enriched, depending on the context."""

    def __init__(self, model: str, system_prompt: str, response_model: BaseModel,
                 temperature: Optional[float] = None,
                 reasoning_effort: Optional[str] = None,
                 verbosity: Optional[str] = None):
        self.provider = model.split('/')[0]
        self.system_prompt = system_prompt

        if self.provider == 'openai':
            self.enricher = OpenAIEnricher(response_model=response_model,
                                           model=model,
                                           system_prompt=self.system_prompt,
                                           temperature=temperature if temperature is not None else 0.0)
        elif self.provider == 'google':
            self.enricher = GoogleEnrichClient(response_model=response_model,
                                               model=model,
                                               system_prompt=self.system_prompt)
        else:
            raise ValueError(f"Provider {self.provider} is not supported. Supported providers are: ['openai', 'google']")
        self.cached_enricher = CachedEnrichClient(self.enricher)

    def enrich(self, prompt: str) -> BaseModel:
        """Enrich a single prompt, now, and cache the result."""
        return self.cached_enricher.enrich(prompt)

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


class ProductEnricher:
    """Enrich a dataframe of products."""

    def __init__(self, enricher: AutoEnricher, prompt_fn, attrs=None,
                 separator: str = " sep "):
        self.enricher = enricher
        self.prompt_fn = prompt_fn
        self.separator = separator
        if attrs is None:  # Inferred from the BaseModel
            output_cls = enricher.response_model
            attrs = output_cls.__fields__.keys()
            attrs = set(attrs)  # Ensure unique attributes
            # Get properties too
            attrs = attrs.union(attr for attr in dir(output_cls) if isinstance(getattr(output_cls, attr), property))
            # Remove any beginning with __
            attrs = {attr for attr in attrs if not attr.startswith('__')}
            # Remove pydantic internal attributes
            attrs = {attr for attr in attrs if attr not in
                     ["model_fields_set", "model_extra", "model_config", "model_json_schema"]}
            logger.info(f"Enriching products with attributes: {attrs}")
        self.attrs = attrs

    def _slice_out_searcharray_cols(self, products: pd.DataFrame) -> pd.DataFrame:
        """Slice out columns that are SearchArray columns."""
        searcharray_cols = [
            col for col in products.columns
            if isinstance(products[col].array, SearchArray)
        ]
        return products.drop(columns=searcharray_cols, errors='ignore')

    def enrich_one(self, product: dict):
        prompt = self.prompt_fn(product)
        return self.enricher.enrich(prompt)

    def enrich_all(self, products: pd.DataFrame, workers=5, batch_size=100) -> pd.DataFrame:
        products = self._slice_out_searcharray_cols(products)

        def enrich_one(product):
            prompt = self.prompt_fn(product)
            return self.enricher.enrich(prompt)

        def post_process(enriched_data):
            if enriched_data:
                for attr in self.attrs:
                    value = getattr(enriched_data, attr, None)
                    # If iterable, join
                    if isinstance(value, (list, tuple)):
                        value = self.separator.join(map(str, value))
                    elif value is None:
                        value = ""
                    products.at[idx, attr] = value if value is not None else ""
            else:
                logger.warning(f"Enrichment failed for product {products.get('product_id', 'unknown')}")
                for attr in self.attrs:
                    products.at[idx, attr] = ""

        logger.info(f"Enriching {len(products)} products immediately (non-batch)")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            with tqdm(total=len(products), desc="Enriching products") as pbar:
                for i in range(0, len(products), batch_size):
                    batch_df = products.iloc[i:i + batch_size]

                    futures = {
                        executor.submit(enrich_one, row.to_dict()): idx
                        for idx, row in batch_df.iterrows()
                    }

                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            enriched_data = future.result()
                            post_process(enriched_data)
                        except Exception as e:
                            logger.error(f"Error enriching product at index {idx}: {str(e)}")
                        pbar.update(1)
        return products
