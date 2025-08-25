import instructor
from cheat_at_search.agent.enrich_client import EnrichClient, DebugMetaData
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.data_dir import key_for_provider
from typing import Optional, Tuple
from pydantic import BaseModel
import json
from hashlib import md5


logger = log_to_stdout("instructor_enrich_client")


def normalize_usage(completion):
    if hasattr(completion, "usage") and hasattr(completion.usage, "input_tokens"):  # OpenAI
        normalized = {
            "prompt_tokens": completion.usage.input_tokens,
            "completion_tokens": completion.usage.output_tokens,
        }
        if hasattr(completion.usage, "total_tokens"):
            normalized["total_tokens"] = completion.usage.total_tokens
        else:
            normalized["total_tokens"] = normalized["prompt_tokens"] + normalized["completion_tokens"]
        return normalized
    elif hasattr(completion, "usage") and hasattr(completion.usage, "prompt_tokens"):  # OpenRouter
        return {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens,
        }
    elif hasattr(completion, "usage_metadata"):  # Gemini
        return {
            "prompt_tokens": completion.usage_metadata.prompt_token_count,
            "completion_tokens": completion.usage_metadata.candidates_token_count,
            "total_tokens": completion.usage_metadata.total_token_count,
        }
    else:
        return {}


class InstructorEnrichClient(EnrichClient):
    def __init__(self,
                 response_model: BaseModel,
                 model: str,
                 system_prompt: str = None,
                 temperature: Optional[float] = None,
                 verbosity: Optional[str] = None,
                 reasoning_effort: Optional[str] = None):
        self._response_model = response_model
        self.model = model
        self.provider = model.split('/')[0]
        self.api_key = key_for_provider(self.provider)
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.verbosity = verbosity
        self.reasoning_effort = reasoning_effort
        self.client = instructor.from_provider(model, api_key=self.api_key)

    def str_hash(self):
        output_schema_hash = md5(json.dumps(self.response_model.model_json_schema(mode='serialization')).encode()).hexdigest()
        return md5(f"{self.model}_{self.system_prompt}_{self.temperature}_{self.verbosity}_{self.reasoning_effort}_{output_schema_hash}".encode()).hexdigest()

    def _enrich(self, prompt: str) -> Tuple[Optional[BaseModel],
                                            Optional[DebugMetaData]]:
        try:
            prompts = []
            if self.system_prompt:
                prompts.append({"role": "system", "content": self.system_prompt})
                prompts.append({"role": "user", "content": prompt})

            resp, completion = self.client.chat.completions.create_with_completion(
                response_model=self.response_model,
                messages=prompts
                # response_format=type_to_response_format_param(self.response_model),
            )
            usage = normalize_usage(completion)
            debug_metadata = DebugMetaData(
                model=self.model,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                reasoning_tokens=usage.get("reasoning_tokens", 0),
                response_id=getattr(completion, 'id', None),
                output=resp
            )
            return resp, debug_metadata
        except Exception as e:
            logger.error(f"Error during enrichment: {str(e)}")
            return None, None

    def enrich(self, prompt: str) -> Optional[BaseModel]:
        """Enrich a single prompt, now."""
        resp, metadata = self._enrich(prompt)
        return resp

    def debug(self, prompt: str) -> Optional[DebugMetaData]:
        """Enrich a single prompt, now, and return debug metadata."""
        return self._enrich(prompt)[1]

    @property
    def response_model(self):
        return self._response_model
