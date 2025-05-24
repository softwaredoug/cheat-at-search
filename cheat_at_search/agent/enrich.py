from ollama import chat
from openai import OpenAI
from cheat_at_search.logger import log_to_stdout
from typing import Optional
from pydantic import BaseModel
import pickle
import os
from hashlib import md5


logger = log_to_stdout(logger_name="query_parser")


class Enricher:

    def enrich(self, prompt: str):
        raise NotImplementedError("Subclasses must implement this method.")


class OllamaEnricher(Enricher):
    def __init__(self, cls: BaseModel, model: str = "llama3.2", system_prompt: str = None):
        self.model = model
        self.system_prompt = system_prompt
        self.cls = cls

    def enrich(self, prompt: str) -> Optional[BaseModel]:
        try:
            prompts = []
            if self.system_prompt:
                prompts.append({"role": "system", "content": self.system_prompt})
                prompts.append({"role": "user", "content": prompt})
            response = chat(
                model=self.model,
                messages=prompts,
                format=self.cls.model_json_schema()
            )
            if response.message.content:
                return self.cls.model_validate_json(response.message.content)
        except Exception as e:
            logger.error(f"Error parsing prompt '{prompt}': {str(e)}")
            # Return a default object with keywords in case of errors
            raise e
        return None


class OpenAIEnricher(Enricher):
    def __init__(self, cls: BaseModel, model: str = "gpt-4o", system_prompt: str = None):
        self.model = model
        self.cls = cls
        self.system_prompt = system_prompt
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def enrich(self, prompt: str) -> Optional[BaseModel]:
        try:
            prompts = []
            if self.system_prompt:
                prompts.append({"role": "system", "content": self.system_prompt})
                prompts.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompts,
                response_format=self.cls.model_json_schema()
            )
            if response.choices and response.choices[0].message.content:
                return self.cls.model_validate_json(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error parsing prompt '{prompt}': {str(e)}")
            # Return a default object with keywords in case of errors
            raise e
        return None


class CachedEnricher(Enricher):
    def __init__(self, enricher: Enricher, cache_file: str = None):
        if cache_file is None:
            enricher_class = enricher.__class__.__name__
            # Get hash of system prompt
            cache_file = f"data/{enricher_class.lower()}_cache.pkl"
            if hasattr(enricher, 'system_prompt'):
                system_prompt_hash = md5(enricher.system_prompt.encode()).hexdigest()
                cache_file = f"data/{enricher_class.lower()}_cache_{system_prompt_hash}.pkl"
        self.enricher = enricher
        self.cache_file = cache_file
        self.cache = {}
        self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)

    def save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)

    def enrich(self, prompt: str) -> Optional[BaseModel]:
        if prompt in self.cache:
            return self.cache[prompt]
        enriched_data = self.enricher.enrich(prompt)
        if enriched_data:
            self.cache[prompt] = enriched_data
            self.save_cache()
        return enriched_data
