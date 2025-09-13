from .enrich_client import EnrichClient, DebugMetaData
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.data_dir import key_for_provider
from typing import Optional, Tuple
from pydantic import BaseModel
import json
from hashlib import md5
from google import genai


logger = log_to_stdout("google_enrich_client")


def maybe_strip_markdown(text: str) -> str:
    if text.startswith("```json") and text.endswith("```"):
        return text[len("```json"): -len("```")].strip()
    return text.strip()


class GoogleEnrichClient(EnrichClient):
    def __init__(self, response_model: BaseModel, model: str, system_prompt: str = None,
                 temperature: float = 0.0):
        super().__init__(response_model=response_model)
        self.provider = model.split('/')[0]
        self.model = model.split('/')[-1]
        if self.provider != 'google':
            raise ValueError(f"Provider {self.provider} is not supported. This client only supports Google.")
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.last_exception = None

        google_key = key_for_provider(self.provider)

        if not google_key:
            raise ValueError("No Google API key provided. Set GOOGLE_API_KEY environment variable or create a key file in the cache directory.")
        self.client = genai.Client(
            api_key=google_key,
        )

    def str_hash(self):
        output_schema_hash = md5(json.dumps(self.response_model.model_json_schema(mode='serialization')).encode()).hexdigest()
        return md5(f"{self.model}_{self.system_prompt}_{self.temperature}_{output_schema_hash}".encode()).hexdigest()

    def get_num_tokens(self, prompt: str) -> Tuple[int, int]:
        """Run the response directly and return teh number of tokens"""
        cls_value, num_input_tokens, num_output_tokens = self.enrich(prompt, return_num_tokens=True)
        return num_input_tokens, num_output_tokens

    def _enrich(self, prompt: str) -> Tuple[Optional[BaseModel], Optional[DebugMetaData]]:
        try:
            schema = self.response_model.model_json_schema(mode='serialization')
            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    system_instruction=self.system_prompt,
                    temperature=self.temperature,
                ),
            )
            json_text = maybe_strip_markdown(resp.text)
            usage = resp.usage_metadata
            obj = self.response_model.model_validate_json(json_text)
            debug = DebugMetaData(
                model=self.model,
                prompt_tokens=usage.prompt_token_count,
                completion_tokens=usage.candidates_token_count,
                reasoning_tokens=0,
                response_id=getattr(resp, 'id', None),
                output=obj
            )
            return obj, debug
        except Exception as e:
            logger.error(f"Error during enrichment: {str(e)}")
            self.last_exception = e
            raise e

    def enrich(self, prompt: str) -> Optional[BaseModel]:
        obj, debug = self._enrich(prompt)
        return obj

    def debug(self, prompt: str) -> Optional[BaseModel]:
        obj, debug = self._enrich(prompt)
        return debug
