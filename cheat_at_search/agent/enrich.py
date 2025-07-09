from openai import OpenAI, APIError
from openai import AzureOpenAI  # Add AzureOpenAI import
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

from openai.lib._parsing._completions import type_to_response_format_param


logger = log_to_stdout(logger_name="query_parser")


CACHE_PATH = ensure_data_subdir("enrich_cache")

# OpenAI key setup (only if not using Azure)
openai_key = None
use_azure = bool(os.getenv("USE_AZURE_OPENAI", "").lower() in ["true", "1", "yes"])
if not use_azure:
    KEY_PATH = f"{DATA_PATH}/openai_key.txt"
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
else:
    # Prompt for Azure OpenAI config if not set in environment
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    if not AZURE_OPENAI_API_KEY:
        AZURE_OPENAI_API_KEY = getpass.getpass("Enter your Azure OpenAI API key: ")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not AZURE_OPENAI_ENDPOINT:
        AZURE_OPENAI_ENDPOINT = input("Enter your Azure OpenAI endpoint (e.g., https://your-resource.openai.azure.com/): ").strip()
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    if not AZURE_OPENAI_API_VERSION:
        AZURE_OPENAI_API_VERSION = input("Enter your Azure OpenAI API version (e.g., 2025-01-01-preview): ").strip()

def get_azure_client():
    if not AZURE_OPENAI_API_KEY:
        raise ValueError("No Azure OpenAI API key provided. Set AZURE_OPENAI_API_KEY environment variable.")
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

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
        # Auto-detect Azure usage from environment variable
        self.use_azure = use_azure or bool(os.getenv("USE_AZURE_OPENAI", "").lower() in ["true", "1", "yes"])
        if self.use_azure:
            self.client = get_azure_client()
        else:
            if not openai_key:
                raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or create a key file in the cache directory.")
            self.client = OpenAI(
                api_key=openai_key,
            )

    def str_hash(self):
        output_schema_hash = md5(json.dumps(self.cls.model_json_schema(mode='serialization')).encode()).hexdigest()
        return md5(f"{self.model}_{self.system_prompt}_{self.temperature}_{output_schema_hash}_{self.use_azure}".encode()).hexdigest()

    def get_num_tokens(self, prompt: str) -> Tuple[int, int]:
        """Run the response directly and return the number of tokens"""
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

            if self.use_azure:
                # Azure expects deployment name as model
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=prompts,
                    temperature=self.temperature,
                )
                # Azure returns choices[0].message.content as the output
                content = completion.choices[0].message.content
                # Try to parse as JSON for pydantic
                try:
                    cls_value = self.cls.model_validate_json(content)
                except Exception:
                    # fallback: try to parse as dict
                    cls_value = self.cls.model_validate(content)
                # Azure does not provide token usage in the same way
                num_input_tokens = getattr(completion.usage, "prompt_tokens", None)
                num_output_tokens = getattr(completion.usage, "completion_tokens", None)
                if cls_value and return_num_tokens:
                    return cls_value, num_input_tokens, num_output_tokens
                elif cls_value:
                    return cls_value
            else:
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
        if task_id is None:
            raise ValueError("task_id must be provided for batch enrichment.")

        task_id = self.build_task_id(task_id, prompt)
        if task_id in self.task_cache:
            logger.debug("Task ID {task_id} enrichment found in cache.")
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
        task_ids = [line['custom_id'] for line in self.batch_lines]
        if len(task_ids) != len(set(task_ids)):
            logger.error(f"Duplicate task_id detected in batch: {task_id}. This may lead to errors in batch processing.")

    def get_output(self, task_id: str, prompt: str) -> Optional[BaseModel]:
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

        if getattr(self.enricher, "use_azure", False):
            logger.warning("Batch enrichment is not implemented for Azure OpenAI. Skipping batch submission.")
            self.batch_lines = []
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
                try:
                    cls_value = self.enricher.cls.model_validate_json(content)
                    self.task_cache[task_id] = cls_value
                    logger.debug(f"Task ID {task_id} enriched successfully.")
                except Exception as e:
                    logger.error(f"Error parsing content for task ID {task_id}: {str(e)}")
                    continue
            try:
                with open(self.batch_cache_file, 'wb') as f:
                    pickle.dump(self.task_cache, f)
            except KeyboardInterrupt:
                logger.warning("Batch enrichment cache saving interrupted. Retrying save before raising exception.")
                with open(self.batch_cache_file, 'wb') as f:
                    pickle.dump(self.task_cache, f)
                raise
        self.batch_lines = []

class CachedEnricher(Enricher):
    def __init__(self, enricher: Enricher):
        enricher_class = enricher.__class__.__name__
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
        return md5("_".join(prompt.split()).encode()).hexdigest()

    def get_num_tokens(self, prompt: str) -> Tuple[int, int]:
        cls_value, num_input_tokens, num_output_tokens = self.enricher.enrich(prompt, return_num_tokens=True)
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

    def __init__(self, model: str, system_prompt: str, output_cls: BaseModel, use_azure: bool = False):
        self.system_prompt = system_prompt
        self.enricher = OpenAIEnricher(cls=output_cls, model=model, system_prompt=self.system_prompt)
        self.cached_enricher = CachedEnricher(self.enricher)
        self.batch_enricher = BatchOpenAIEnricher(self.enricher)

    def enrich(self, prompt: str, task_id: str = None) -> BaseModel:
        return self.cached_enricher.enrich(prompt)

    def get_num_tokens(self, prompt: str) -> Tuple[int, int]:
        return self.cached_enricher.enricher.get_num_tokens(prompt)

    def batch(self, prompt: str, task_id) -> None:
        self.batch_enricher.enrich(prompt, task_id=task_id)

    def submit_batch(self, entries_per_batch=1000):
        self.batch_enricher.submit(entries_per_batch=entries_per_batch)

    def get_batch_output(self, prompt: str, task_id: str) -> Optional[BaseModel]:
        return self.batch_enricher.get_output(task_id, prompt)
