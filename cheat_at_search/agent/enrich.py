from ollama import chat
from openai import OpenAI
from cheat_at_search.logger import log_to_stdout
from typing import Optional
from pydantic import BaseModel
import pickle
import json
import os
from hashlib import md5
from time import sleep

from openai.lib._parsing._completions import type_to_response_format_param


logger = log_to_stdout(logger_name="query_parser")


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(FILE_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


class Enricher:

    def enrich(self, prompt: str, task_id: str = None) -> Optional[BaseModel]:
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


def to_openai_batched(task_id, model, prompts, cls: BaseModel,
                      temperature: float = 0.1):
    cls_to_json = type_to_response_format_param(cls)
    task = {
        "custom_id": task_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": prompts,
            "temperature": temperature,
            "response_format": cls_to_json,
        }
    }
    return task


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
            response = self.client.responses.parse(
                model=self.model,
                input=prompts,
                text_format=self.cls
            )
            cls_value = response.output_parsed
            if cls_value:
                return cls_value
        except Exception as e:
            logger.error(f"Error parsing prompt '{prompt}': {str(e)}")
            # Return a default object with keywords in case of errors
            raise e
        return None


class BatchOpenAIEnricher(Enricher):

    def __init__(self, enricher: OpenAIEnricher):
        self.enricher = enricher
        self.batch_lines = []
        self.task_cache = {}
        try:
            with open(f"{DATA_DIR}/enrich_cache/batch_enrich.pkl", 'rb') as f:
                self.task_cache = pickle.load(f)
        except FileNotFoundError:
            logger.warning("Batch enrichment cache file not found, starting with empty cache.")
            self.task_cache = {}

    def enrich(self, prompt, task_id: str = None) -> Optional[BaseModel]:
        # For batch processing, you would collect prompts and send them in bulk
        # Here we just call the enricher directly for simplicity
        if task_id is None:
            raise ValueError("task_id must be provided for batch enrichment.")

        # Prepend prompt hash, system prompt and model to task id
        task_id = f"{task_id}_{md5(prompt.encode()).hexdigest()}_{md5(self.enricher.system_prompt.encode()).hexdigest()}_{self.enricher.model}"
        if task_id in self.task_cache:
            logger.info(f"Task ID {task_id} enrichment found in cache.")
            # If in cache, just return
            return self.task_cache[task_id]

        prompts = []
        if self.enricher.system_prompt:
            prompts.append({"role": "system", "content": self.enricher.system_prompt})
            prompts.append({"role": "user", "content": prompt})

        batch_line = to_openai_batched(
            task_id,
            self.enricher.model,
            prompts,
            self.enricher.cls
        )
        self.batch_lines.append(batch_line)

    def submit(self, block=True):
        if not self.batch_lines:
            logger.warning("No prompts to submit for batch enrichment.")
            return []

        with open(f"{DATA_DIR}/batch.jsonl", 'w') as f:
            for line in self.batch_lines:
                f.write(json.dumps(line) + "\n")
        batch_input_file = self.enricher.client.files.create(
            file=open(f"{DATA_DIR}/batch.jsonl", "rb"),
            purpose="batch"
        )

        # Add empty entry to task cache if not exists
        for task in self.batch_lines:
            task_id = task['custom_id']
            if task_id not in self.task_cache:
                self.task_cache[task_id] = {}

        batch_input_file_id = batch_input_file.id
        batch = self.enricher.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "nightly eval job"
            }
        )
        # Wait for batch to be valid
        batch = self.enricher.client.batches.retrieve(batch.id)
        backoff = 4
        while batch.status != "completed":
            logger.info(f"Batch {batch.id} is {batch.status}, waiting for {backoff} seconds...")
            sleep(backoff)
            backoff *= 2
            if backoff > 200:
                backoff = 256
            batch = self.enricher.client.batches.retrieve(batch.id)
            print(batch.status)

        resp = self.enricher.client.files.content(batch.output_file_id).text
        for line in resp.splitlines():
            resp = json.loads(line)
            if 'error' in resp and resp['error']:
                logger.error(f"Error in batch response: {resp['error']}")
                continue
            task_id = resp.get('custom_id')
            choices = resp.get('response', {}).get('body', {}).get('choices', [])
            if not choices:
                logger.warning(f"No choices found for task ID {task_id}.")
                continue
            choice = choices[0]
            if 'message' not in choice or 'content' not in choice['message']:
                logger.warning(f"No content found for task ID {task_id}.")
                continue
            content = choice['message']['content']
            # Parse into cls
            try:
                cls_value = self.enricher.cls.model_validate_json(content)
                self.task_cache[task_id] = cls_value
                logger.info(f"Task ID {task_id} enriched successfully.")
            except Exception as e:
                logger.error(f"Error parsing content for task ID {task_id}: {str(e)}")
                continue
        # Save to cache
        with open(f"{DATA_DIR}/enrich_cache/batch_enrich.pkl", 'wb') as f:
            pickle.dump(self.task_cache, f)

        return batch.id


class CachedEnricher(Enricher):
    def __init__(self, enricher: Enricher, cache_file: str = None):
        if cache_file is None:
            enricher_class = enricher.__class__.__name__
            # Get hash of system prompt
            cache_file = f"{DATA_DIR}/enrich_cache/{enricher_class.lower()}"
            if hasattr(enricher, 'system_prompt'):
                system_prompt_hash = md5(enricher.system_prompt.encode()).hexdigest()
                cache_file += f"_{system_prompt_hash}"
            if hasattr(enricher, 'model'):
                cache_file += f"_{enricher.model}"
            schema = json.dumps(enricher.cls.model_json_schema(mode='serialization'))
            schema_hash = md5(schema.encode()).hexdigest()
            cache_file += f"{schema_hash}_cache.pkl"
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
