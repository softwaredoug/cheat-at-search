from .enrich_client import EnrichClient, DebugMetaData
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.data_dir import key_for_provider
from typing import Optional, Tuple
from pydantic import BaseModel
import json
from hashlib import md5
from openai import OpenAI, APIError


logger = log_to_stdout("openai_enrich_client")


class OpenAIEnricher(EnrichClient):
    def __init__(self, response_model: BaseModel, model: str, system_prompt: str = None,
                 temperature: float = 0.0, verbosity: str = 'low',
                 reasoning_effort: str = 'minimal'):
        super().__init__(response_model=response_model)
        self.provider = model.split('/')[0]
        self.model = model.split('/')[-1]
        if self.provider != 'openai':
            raise ValueError(f"Provider {self.provider} is not supported. This client only supports OpenAI.")
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.verbosity = verbosity
        self.reasoning_effort = reasoning_effort
        self.last_exception = None

        openai_key = key_for_provider(self.provider)

        if not openai_key:
            raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or create a key file in the cache directory.")
        self.client = OpenAI(
            api_key=openai_key,
        )

    def str_hash(self):
        output_schema_hash = md5(json.dumps(self.response_model.model_json_schema(mode='serialization')).encode()).hexdigest()
        return md5(f"{self.model}_{self.system_prompt}_{self.temperature}_{output_schema_hash}".encode()).hexdigest()

    def get_num_tokens(self, prompt: str) -> Tuple[int, int]:
        """Run the response directly and return teh number of tokens"""
        cls_value, num_input_tokens, num_output_tokens = self.enrich(prompt, return_num_tokens=True)
        return num_input_tokens, num_output_tokens

    def _gpt5_call(self, inputs: list[str], reasoning_effort: str, verbosity: str):
        response = self.client.responses.parse(
            model=self.model,
            reasoning={"effort": reasoning_effort},
            input=inputs,
            text_format=self.response_model,
            text={"verbosity": verbosity}
        )
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
                response = self.client.responses.parse(
                    model=self.model,
                    temperature=self.temperature,
                    input=prompts,
                    text_format=self.response_model
                )
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
