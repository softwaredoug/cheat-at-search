import requests
from cheat_at_search.agent.enrich_client import EnrichClient, DebugMetaData
from cheat_at_search.logger import log_to_stdout
from typing import Optional, Tuple
from pydantic import BaseModel
import json
from hashlib import md5
from openai.lib._parsing._completions import type_to_response_format_param


logger = log_to_stdout("openrouter_enrich_client")


def validate_params(model, temperature, verbosity, reasoning_effort):
    if ('gpt-4' in model) or ('gpt-5-main' in model):
        if verbosity is not None:
            raise ValueError("Verbosity is not supported for GPT-4 models.")
        if reasoning_effort is not None:
            raise ValueError("Reasoning effort is not supported for GPT-4 models.")
        if temperature is not None and temperature < 0:
            raise ValueError("Temperature must be non-negative for GPT-4 models.")
        if temperature is None:
            temperature = 0.0
    elif 'gpt-5' in model:
        if verbosity is not None and verbosity not in ['low', 'medium', 'high']:
            raise ValueError("Verbosity must be one of ['low', 'medium', 'high'] for GPT-5 models.")
        elif verbosity is None:
            verbosity = 'low'
        if reasoning_effort is not None and reasoning_effort not in ['minimal', 'low', 'medium', 'high']:
            raise ValueError("Reasoning effort must be one of ['minimal', 'low', 'medium', 'high'] for GPT-5 models.")
        elif reasoning_effort is None:
            reasoning_effort = 'medium'
        if temperature is not None:
            raise ValueError("Temperature is not supported for GPT-5 models.")
    return model, temperature, verbosity, reasoning_effort


def pathify_openai_model(model: str) -> str:
    if 'gpt-4' in model or 'gpt-5' in model:
        return f"openai/{model}"


class OpenRouterEnrichClient(EnrichClient):
    def __init__(self, cls: BaseModel, model: str, system_prompt: str = None,
                 temperature: Optional[float] = None,
                 verbosity: Optional[str] = None,
                 reasoning_effort: Optional[str] = None):
        self.cls = cls
        self.system_prompt = system_prompt
        model, temperature, verbosity, reasoning_effort = validate_params(model, temperature, verbosity, reasoning_effort)
        self.model = model
        self.temperature = temperature
        self.verbosity = verbosity
        self.reasoning_effort = reasoning_effort
        self.last_exception = None

        # Reimport to get openai key in case later mount
        api_key = None
        from cheat_at_search.data_dir import OPENROUTER_API_KEY as api_key
        self.model = pathify_openai_model(model)

        if not api_key:
            raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or create a key file in the cache directory.")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
        }

    def str_hash(self):
        output_schema_hash = md5(json.dumps(self.cls.model_json_schema(mode='serialization')).encode()).hexdigest()
        return md5(f"{self.model}_{self.system_prompt}_{self.temperature}_{self.verbosity}_{self.reasoning_effort}_{output_schema_hash}".encode()).hexdigest()

    def get_num_tokens(self, prompt: str) -> Tuple[int, int]:
        """Run the response directly and return teh number of tokens"""
        cls_value, num_input_tokens, num_output_tokens = self.enrich(prompt, return_num_tokens=True)
        return num_input_tokens, num_output_tokens

    def enrich(self, prompt: str, return_num_tokens=False) -> Optional[BaseModel]:
        """Enrich a single prompt, now."""
        metadata = self._enrich(prompt, return_num_tokens=return_num_tokens)
        if metadata:
            if return_num_tokens:
                return metadata.output, metadata.prompt_tokens, metadata.completion_tokens
            return metadata.output
        return None

    def debug(self, prompt: str) -> Optional[DebugMetaData]:
        """Enrich a single prompt, now, and return debug metadata."""
        return self._enrich(prompt, return_num_tokens=True)

    def _enrich(self, prompt: str, return_num_tokens=False) -> Tuple[Optional[BaseModel],
                                                                     Optional[DebugMetaData]]:
        response_id = None
        prev_response_id = None
        try:
            prompts = []
            if self.system_prompt:
                prompts.append({"role": "system", "content": self.system_prompt})
                prompts.append({"role": "user", "content": prompt})

            data = {
                "model": self.model,
                "messages": prompts,
                "response_format": type_to_response_format_param(self.cls),
            }

            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=self.headers,
                json=data
            )
            resp.raise_for_status()
            out = resp.json()

            content = out["choices"][0]["message"]["content"]
            parsed = self.cls.model_validate(json.loads(content))

            response_id = out.get('id', None)
            prev_response_id = response_id
            num_input_tokens = out['usage']['prompt_tokens'] if 'usage' in out and 'prompt_tokens' in out['usage'] else 0
            num_output_tokens = out['usage']['completion_tokens'] if 'usage' in out and 'completion_tokens' in out['usage'] else 0
            reasoning_tokens = 0  # Not provided by OpenRouter

            metadata = DebugMetaData(
                model=self.model,
                prompt_tokens=num_input_tokens,
                completion_tokens=num_output_tokens,
                reasoning_tokens=reasoning_tokens,
                response_id=response_id,
                output=parsed
            )
            return metadata
        except requests.HTTPError as e:
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
