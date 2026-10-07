from .enrich_client import EnrichClient, DebugMetaData
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.data_dir import key_for_provider
from typing import Optional, Tuple
from pydantic import BaseModel
import json
from hashlib import md5
from openai import OpenAI, APIError


logger = log_to_stdout("openai_enrich_client")


_DEFAULT_REASONING_EFFORT_BY_MODEL = {
    "gpt-5": "minimal",
    "gpt-5-mini": "minimal",
    "gpt-5-nano": "minimal",
    "gpt-5.1": "none",
    "gpt-5.2": "none",
    "gpt-5.4": "none",
    "gpt-5.4-mini": "none",
    "gpt-5.4-nano": "none",
}


def _default_reasoning_effort(model: str) -> Optional[str]:
    """Choose the lowest documented reasoning effort for known GPT-5 models."""
    for model_name, effort in _DEFAULT_REASONING_EFFORT_BY_MODEL.items():
        # Also match dated model snapshots without matching unrelated variants.
        if model == model_name or model.startswith(f"{model_name}-20"):
            return effort
    return None


class OpenAIEnricher(EnrichClient):
    def __init__(self, response_model: BaseModel, model: str, system_prompt: str = None,
                 temperature: Optional[float] = None, verbosity: Optional[str] = 'low',
                 reasoning_effort: Optional[str] = None):
        super().__init__(response_model=response_model)
        self.provider = model.split('/')[0]
        self.model = model.split('/')[-1]
        if self.provider != 'openai':
            raise ValueError(f"Provider {self.provider} is not supported. This client only supports OpenAI.")
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.verbosity = verbosity if verbosity is not None else 'low'
        self.reasoning_effort = (
            reasoning_effort
            if reasoning_effort is not None
            else _default_reasoning_effort(self.model)
        )
        self.last_exception = None

        openai_key = key_for_provider(self.provider)

        if not openai_key:
            raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or create a key file in the cache directory.")
        self.client = OpenAI(
            api_key=openai_key,
        )

    def str_hash(self):
        output_schema_hash = md5(json.dumps(self.response_model.model_json_schema(mode='serialization')).encode()).hexdigest()
        cache_signature = (
            f"{self.model}_{self.system_prompt}_{self.temperature}_"
            f"{self.reasoning_effort}_{self.verbosity}_{output_schema_hash}"
        )
        return md5(cache_signature.encode()).hexdigest()

    def get_num_tokens(self, prompt: str) -> Tuple[int, int]:
        """Run the response directly and return teh number of tokens"""
        cls_value, num_input_tokens, num_output_tokens = self.enrich(prompt, return_num_tokens=True)
        return num_input_tokens, num_output_tokens

    def _gpt5_call(self, inputs: list[str], reasoning_effort: Optional[str], verbosity: str):
        parse_kwargs = {
            "model": self.model,
            "input": inputs,
            "text_format": self.response_model,
            "text": {"verbosity": verbosity},
        }
        if reasoning_effort is not None:
            parse_kwargs["reasoning"] = {"effort": reasoning_effort}
        response = self.client.responses.parse(**parse_kwargs)
        return response

    def _enrich(self, prompt: str) -> Tuple[Optional[BaseModel], Optional[DebugMetaData]]:
        response_id = None
        prev_response_id = None
        try:
            prompts = []
            if self.system_prompt:
                prompts.append({"role": "system", "content": self.system_prompt})
                prompts.append({"role": "user", "content": prompt})
            if 'gpt-5' in self.model:
                response = self._gpt5_call(
                    inputs=prompts,
                    reasoning_effort=self.reasoning_effort,
                    verbosity=self.verbosity
                )
            else:
                parse_kwargs = {
                    "model": self.model,
                    "input": prompts,
                    "text_format": self.response_model,
                }
                if self.temperature is not None:
                    parse_kwargs["temperature"] = self.temperature
                response = self.client.responses.parse(**parse_kwargs)
            response_id = response.id
            prev_response_id = response_id
            num_input_tokens = response.usage.input_tokens
            num_output_tokens = response.usage.output_tokens

            cls_value = response.output_parsed
            debug_metadata = DebugMetaData(
                model=self.model,
                prompt_tokens=num_input_tokens,
                completion_tokens=num_output_tokens,
                reasoning_tokens=0,
                response_id=response_id,
                output=cls_value
            )
            return cls_value, debug_metadata
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

    def debug(self, prompt: str) -> Optional[DebugMetaData]:
        """Enrich a single prompt, now, and return debug metadata."""
        return self._enrich(prompt)[1]

    def enrich(self, prompt: str, return_num_tokens: bool = False) -> Optional[BaseModel]:
        """Enrich a single prompt, now."""
        resp, metadata = self._enrich(prompt)
        if return_num_tokens and metadata:
            return resp, metadata.prompt_tokens, metadata.completion_tokens
        return resp
