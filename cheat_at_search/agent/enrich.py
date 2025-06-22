from openai import OpenAI
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.data_dir import ensure_data_subdir, DATA_PATH
from typing import Optional
from pydantic import BaseModel
import pickle
import json
import os
from hashlib import md5
from time import sleep

from openai.lib._parsing._completions import type_to_response_format_param


logger = log_to_stdout(logger_name="query_parser")


CACHE_PATH = ensure_data_subdir("enrich_cache")
KEY_PATH = f"{DATA_PATH}/openai_key.txt"
openai_key = None
if os.getenv("OPENAI_API_KEY"):
    openai_key = os.getenv("OPENAI_API_KEY")
else:
    try:
        logger.info(f"Reading OpenAI API key from {KEY_PATH}")
        with open(KEY_PATH, "r") as f:
            openai_key = f.read().strip()
    except FileNotFoundError:
        logger.warning(f"Either set OPENAI_API_KEY environment variable or create a key file at {KEY_PATH} holding the key.")


class Enricher:
    def enrich(self, prompt: str, task_id: str = None) -> Optional[BaseModel]:
        raise NotImplementedError("Subclasses must implement this method.")


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
    def __init__(self, cls: BaseModel, model: str, system_prompt: str = None):
        self.model = model
        self.cls = cls
        self.system_prompt = system_prompt
        if not openai_key:
            raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or create a key file in the cache directory.")
        self.client = OpenAI(
            api_key=openai_key,
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
            with open(f"{CACHE_PATH}/batch_enrich.pkl", 'rb') as f:
                self.task_cache = pickle.load(f)
        except FileNotFoundError:
            logger.warning("Batch enrichment cache file not found, starting with empty cache.")
            self.task_cache = {}

    @property
    def cls(self):
        return self.enricher.cls

    def enrich(self, prompt, task_id: str = None) -> Optional[BaseModel]:
        # For batch processing, you would collect prompts and send them in bulk
        # Here we just call the enricher directly for simplicity
        if task_id is None:
            raise ValueError("task_id must be provided for batch enrichment.")

        # Prepend prompt hash, system prompt and model to task id

        schema = json.dumps(self.enricher.cls.model_json_schema(mode='serialization'))
        schema_hash = md5(schema.encode()).hexdigest()
        task_id = f"{task_id}_{md5(prompt.encode()).hexdigest()}_{md5(self.enricher.system_prompt.encode()).hexdigest()}_{self.enricher.model}_{schema_hash}"
        if task_id in self.task_cache:
            logger.debug("Task ID {task_id} enrichment found in cache.")
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

    def submit(self, entries_per_batch=1000):
        if not self.batch_lines:
            logger.warning("No prompts to submit for batch enrichment.")
            return []

        batches = []
        for i in range(0, len(self.batch_lines), entries_per_batch):

            with open(f"{CACHE_PATH}/batch.jsonl", 'w') as f:
                for line in self.batch_lines[i:i + entries_per_batch]:
                    f.write(json.dumps(line) + "\n")
            batch_input_file = self.enricher.client.files.create(
                file=open(f"{CACHE_PATH}/batch.jsonl", "rb"),
                purpose="batch"
            )

            # Add empty entry to task cache if not exists
            for task in self.batch_lines:
                task_id = task['custom_id']
                if task_id not in self.task_cache:
                    self.task_cache[task_id] = {}

            batch_input_file_id = batch_input_file.id
            batch = self.enriicher.client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": "nightly eval job"
                }
            )
            batches.append(batch)
        for batch in batches:
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
            with open(f"{CACHE_PATH}/batch_enrich.pkl", 'wb') as f:
                pickle.dump(self.task_cache, f)
        # Clear batch lines after successfully blocking
        self.batch_lines = []


class CachedEnricher(Enricher):
    def __init__(self, enricher: Enricher, cache_file: str = None, identifier: str = None):
        if cache_file is None:
            enricher_class = enricher.__class__.__name__
            # Get hash of system prompt
            cache_file = f"{CACHE_PATH}/{enricher_class.lower()}"
            if hasattr(enricher, 'system_prompt'):
                system_prompt_hash = md5(enricher.system_prompt.encode()).hexdigest()
                cache_file += f"_{system_prompt_hash}"
            if hasattr(enricher, 'model'):
                cache_file += f"_{enricher.model}"
            if identifier:
                cache_file += f"_{identifier}"
            schema = json.dumps(enricher.cls.model_json_schema(mode='serialization'))
            schema_hash = md5(schema.encode()).hexdigest()
            cache_file += f"{schema_hash}_cache.pkl"
        self.enricher = enricher
        self.cache_file = cache_file
        self.cache = {}
        self.load_cache()

    def load_cache(self):
        logger.info(f"Loading enrich cache from {self.cache_file}")
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} entries from cache.")
        else:
            logger.warning(f"Cache file {self.cache_file} does not exist, starting with empty cache.")
            self.cache = {}

    def save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)

    def enrich(self, prompt: str) -> Optional[BaseModel]:
        if prompt in self.cache:
            logger.debug(f"Cache hit for prompt: {prompt}")
            return self.cache[prompt]
        logger.debug(f"Cache miss for prompt: {prompt}, enriching...")
        enriched_data = self.enricher.enrich(prompt)
        if enriched_data:
            self.cache[prompt] = enriched_data
            self.save_cache()
        return enriched_data
