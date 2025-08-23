from cheat_at_search.agent.enrich_client import EnrichClient, DebugMetaData
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.data_dir import ensure_data_subdir
import os
import pickle
from hashlib import md5
from typing import Optional, Tuple
from pydantic import BaseModel


logger = log_to_stdout(logger_name="query_parser")


class CachedEnrichClient(EnrichClient):
    def __init__(self, enricher: EnrichClient):
        enricher_class = enricher.__class__.__name__
        # Get hash of system prompt
        self.cache_path = ensure_data_subdir("enrich_cache")
        cache_file = f"{self.cache_path}/{enricher_class.lower()}"
        if hasattr(enricher, 'str_hash'):
            cache_file += f"_{enricher.str_hash()}"
        cache_file += "_cache.pkl"
        self.enricher = enricher
        self.cache_file = cache_file
        self.cache = {}
        self.load_cache()

    def load_cache(self):
        logger.info(f"Loading enrich cache from {self.cache_file}")
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                    logger.info(f"Loaded {len(self.cache)} entries from cache.")
            except Exception as e:
                logger.error(f"Error loading cache file {self.cache_file}: {str(e)}")
                logger.error("Starting with empty cache due to error.")
                # Delete file
                os.remove(self.cache_file)
                self.cache = {}
        else:
            logger.warning(f"Cache file {self.cache_file} does not exist, starting with empty cache.")
            self.cache = {}

    def save_cache(self):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except KeyboardInterrupt:
            logger.warning("Cache saving interrupted. Retrying save before raising exception.")
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            raise

    def prompt_key(self, prompt: str) -> str:
        """Clean up the prompt to ensure it is suitable for caching."""
        # Remove all whitespace
        return md5("_".join(prompt.split()).encode()).hexdigest()

    def get_num_tokens(self, prompt: str) -> Tuple[int, int]:
        """Run the response directly and return the number of tokens"""
        cls_value, num_input_tokens, num_output_tokens = self.enrich(prompt, return_num_tokens=True)
        return num_input_tokens, num_output_tokens

    def debug(self, prompt: str) -> Optional[DebugMetaData]:
        """Enrich a single prompt, now, and return debug metadata."""
        return self.enricher.debug(prompt)

    def enrich(self, prompt: str) -> Optional[BaseModel]:
        prompt_key = self.prompt_key(prompt)
        if prompt_key in self.cache:
            logger.debug(f"Cache hit for prompt: {prompt_key}")
            return self.cache[prompt_key]
        logger.debug(f"Cache miss for prompt: {prompt_key}, enriching...")
        enriched_data = self.enricher.enrich(prompt)
        if enriched_data:
            self.cache[prompt_key] = enriched_data
            self.save_cache()
        return enriched_data

    def str_hash(self) -> str:
        return self.enricher.str_hash()
