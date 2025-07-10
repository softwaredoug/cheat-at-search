from openai import OpenAI, APIError
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.data_dir import ensure_data_subdir, DATA_PATH
from typing import Optional
from pydantic import BaseModel
import pickle
import json
import os
from hashlib import md5
from time import sleep
import getpass
from typing import Tuple
import pandas as pd
from tqdm import tqdm

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
        key = getpass.getpass("Enter your openai key: ")
        with open(os.path.join(KEY_PATH), 'w') as f:
            logger.info(f"Saving OpenAI API key to {KEY_PATH}")
            f.write(key)
            openai_key = key


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
    def __init__(self, cls: BaseModel, model: str, system_prompt: str = None, temperature: float = 0.0):
        self.model = model
        self.cls = cls
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.last_exception = None
        if not openai_key:
            raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or create a key file in the cache directory.")
        self.client = OpenAI(
            api_key=openai_key,
        )

    def str_hash(self):
        output_schema_hash = md5(json.dumps(self.cls.model_json_schema(mode='serialization')).encode()).hexdigest()
        return md5(f"{self.model}_{self.system_prompt}_{self.temperature}_{output_schema_hash}".encode()).hexdigest()

    def get_num_tokens(self, prompt: str) -> Tuple[int, int]:
        """Run the response directly and return teh number of tokens"""
        cls_value, num_input_tokens, num_output_tokens = self.enrich(prompt, return_num_tokens=True)
        return num_input_tokens, num_output_tokens

    def enrich(self, prompt: str, return_num_tokens=False) -> Optional[BaseModel]:
        response_id = None
        prev_response_id = None
        try:
            prompts = []
            if self.system_prompt:
                prompts.append({"role": "system", "content": self.system_prompt})
                prompts.append({"role": "user", "content": prompt})
            response = self.client.responses.parse(
                model=self.model,
                temperature=self.temperature,
                input=prompts,
                text_format=self.cls
            )
            response_id = response.id
            prev_response_id = response_id
            num_input_tokens = response.usage.input_tokens
            num_output_tokens = response.usage.output_tokens

            cls_value = response.output_parsed
            if cls_value and return_num_tokens:
                return cls_value, num_input_tokens, num_output_tokens
            elif cls_value:
                return cls_value
        except APIError as e:
            self.last_exception = e
            logger.error(f"""
                type: {type(e).__name__}

                Error parsing response (resp_id: {response_id} | prev_resp_id: {prev_response_id})

                Prompt:
                {prompt}:

                Exception:
                {str(e)}
                {repr(e)}

            """)
            # Return a default object with keywords in case of errors
            raise e
        return None


class BatchOpenAIEnricher(Enricher):

    def __init__(self, enricher: OpenAIEnricher):
        self.enricher = enricher
        self.batch_lines = []
        self.task_cache = {}
        enr_hash = self.enricher.str_hash()
        self.batch_cache_file = f"{CACHE_PATH}/batch_enrich_{enr_hash}.pkl"
        try:
            with open(self.batch_cache_file, 'rb') as f:
                self.task_cache = pickle.load(f)
            logger.info(f"Loaded {len(self.task_cache)} entries from batch enrichment cache at {self.batch_cache_file}")
        except FileNotFoundError:
            logger.warning(f"Batch enrichment cache file not found, starting with empty cache at {self.batch_cache_file}")
            self.task_cache = {}

    @property
    def cls(self):
        return self.enricher.cls

    def build_task_id(self, task_id, prompt: str):
        schema = json.dumps(self.enricher.cls.model_json_schema(mode='serialization'))
        schema_hash = md5(schema.encode()).hexdigest()
        task_id = f"{task_id}_{md5(prompt.encode()).hexdigest()}_{md5(self.enricher.system_prompt.encode()).hexdigest()}_{self.enricher.model}_{schema_hash}"
        return task_id

    def enrich(self, prompt, task_id: str = None) -> Optional[BaseModel]:
        # For batch processing, you would collect prompts and send them in bulk
        # Here we just call the enricher directly for simplicity
        if task_id is None:
            raise ValueError("task_id must be provided for batch enrichment.")

        # Prepend prompt hash, system prompt and model to task id
        task_id = self.build_task_id(task_id, prompt)
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
            self.enricher.cls,
            temperature=self.enricher.temperature
        )
        self.batch_lines.append(batch_line)
        # Check for uniqueness of all task_ids in the batch
        task_ids = [line['custom_id'] for line in self.batch_lines]
        if len(task_ids) != len(set(task_ids)):
            logger.error(f"Duplicate task_id detected in batch: {task_id}. This may lead to errors in batch processing.")

    def get_output(self, task_id: str, prompt: str) -> Optional[BaseModel]:
        """Get the output of a batch enrichment."""
        task_id = self.build_task_id(task_id, prompt)
        if task_id in self.task_cache:
            logger.debug(f"Retrieving task ID {task_id} from batch cache.")
            return self.task_cache[task_id]
        logger.warning(f"Task ID {task_id} not found in batch cache.")
        return None

    def submit(self, entries_per_batch=1000):
        if not self.batch_lines:
            logger.info("No prompts to submit for batch enrichment (they're probably cached)")
            return []

        batches = []
        submitted_tasks = set()
        for i in range(0, len(self.batch_lines), entries_per_batch):

            with open(f"{CACHE_PATH}/batch.jsonl", 'w') as f:
                for line in self.batch_lines[i:i + entries_per_batch]:
                    task_id = line['custom_id']
                    if task_id in submitted_tasks:
                        logger.warning(f"Task ID {task_id} already submitted, skipping.")
                        continue
                    submitted_tasks.add(task_id)
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
            try:
                batch = self.enricher.client.batches.create(
                    input_file_id=batch_input_file_id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "description": "nightly eval job"
                    }
                )
                batches.append(batch)
            except APIError as e:
                logger.error(f"Error creating batch: {str(e)}")
                logger.error("Batch enrichment failed, clearing batch lines.")
                logger.error("Will wait for other batches to complete before retrying.")
        for batch in batches:
            # Wait for batch to be valid
            logger.info(f"Batch {batch.id}, waiting for completion...")
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

            logger.info(f"Batch {batch.id} completed successfully.")

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
                    logger.debug(f"Task ID {task_id} enriched successfully.")
                except Exception as e:
                    logger.error(f"Error parsing content for task ID {task_id}: {str(e)}")
                    continue
            # Save to cache
            try:
                with open(self.batch_cache_file, 'wb') as f:
                    pickle.dump(self.task_cache, f)
            except KeyboardInterrupt:
                logger.warning("Batch enrichment cache saving interrupted. Retrying save before raising exception.")
                with open(self.batch_cache_file, 'wb') as f:
                    pickle.dump(self.task_cache, f)
                raise
            logger.info(f"Batch {batch.id} result saved")
        # Clear batch lines after successfully blocking
        self.batch_lines = []


class CachedEnricher(Enricher):
    def __init__(self, enricher: Enricher):
        enricher_class = enricher.__class__.__name__
        # Get hash of system prompt
        cache_file = f"{CACHE_PATH}/{enricher_class.lower()}"
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


class AutoEnricher(Enricher):
    """Either serial cached or batch enriched, depending on the context."""

    def __init__(self, model: str, system_prompt: str, output_cls: BaseModel):
        self.system_prompt = system_prompt
        self.enricher = OpenAIEnricher(cls=output_cls, model=model, system_prompt=self.system_prompt)
        self.cached_enricher = CachedEnricher(self.enricher)
        self.batch_enricher = BatchOpenAIEnricher(self.enricher)

    def enrich(self, prompt: str, task_id: str = None) -> BaseModel:
        """Enrich a single prompt, now, and cache the result."""
        return self.cached_enricher.enrich(prompt)

    def get_num_tokens(self, prompt: str) -> Tuple[int, int]:
        """Get the number of tokens for a prompt (runs directly, does not cache)."""
        return self.cached_enricher.enricher.get_num_tokens(prompt)

    def batch(self, prompt: str, task_id) -> None:
        """Add prompt to batch for processing."""
        self.batch_enricher.enrich(prompt, task_id=task_id)

    def submit_batch(self, entries_per_batch=1000):
        """Submit the batch for enrichment."""
        self.batch_enricher.submit(entries_per_batch=entries_per_batch)

    def get_batch_output(self, prompt: str, task_id: str) -> Optional[BaseModel]:
        """Get the output of a batch enrichment."""
        return self.batch_enricher.get_output(task_id, prompt)

    @property
    def output_cls(self):
        """Return the output class of the enricher."""
        return self.enricher.cls


class ProductEnricher:
    """Enrich a dataframe of products."""

    def __init__(self, enricher: AutoEnricher, prompt_fn, attrs=None,
                 separator: str = " sep "):
        self.enricher = enricher
        self.prompt_fn = prompt_fn
        self.separator = separator
        if attrs is None:  # Inferred from the BaseModel
            output_cls = enricher.output_cls
            attrs = output_cls.__fields__.keys()
        self.attrs = attrs

    def enrich_one(self, product_name: str, product_description: str, product_id: str):
        prompt = self.prompt_fn(product_name, product_description)
        return self.enricher.enrich(prompt)

    def enrich_all(self, products: pd.DataFrame):
        def enrich_one(product_name, product_description, product_id):
            prompt = self.prompt_fn(product_name, product_description)
            return self.enricher.enrich(prompt)

        logger.info(f"Enriching {len(products)} products immediately (non-batch)")
        for idx, row in tqdm(products.iterrows(), total=len(products), desc="Enriching products"):
            product_name = row['product_name']
            product_description = row['product_description']
            product_id = row['product_id']
            enriched_data = enrich_one(product_name, product_description, product_id)
            logger.info(f"Enriched product {product_id} ({product_name})")
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
                logger.warning(f"Enrichment failed for product {product_id} ({product_name})")
                for attr in self.attrs:
                    products.at[idx, attr] = ""
        return products

    def batch_all(self, products: pd.DataFrame):

        def submit_batch_job(product_name, product_description, product_id):
            prompt = self.prompt_fn(product_name, product_description)
            self.enricher.batch(prompt, product_id)

        products[['product_name', 'product_description', 'product_id']].apply(lambda x: submit_batch_job(*x), axis=1)

    def fetch_all(self, products: pd.DataFrame):

        def fetch_attr_value(product_name, product_description, product_id, attr):
            prompt = self.prompt_fn(product_name, product_description)
            result = self.enricher.get_batch_output(prompt, product_id)
            return getattr(result, attr) if hasattr(result, attr) else ""

        logger.info(f"Submitting batch job for {len(products)} products")

        logger.info("Batch done, fetching results")
        for attr in self.attrs:
            products[attr] = products[['product_name', 'product_description', 'product_id']].apply(
                lambda x: fetch_attr_value(*x, attr=attr), axis=1)
        return products
