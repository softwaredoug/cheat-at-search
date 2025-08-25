import instructor
from cheat_at_search.agent.enrich_client import EnrichClient, DebugMetaData
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.data_dir import key_for_provider
from typing import Optional, Tuple
from pydantic import BaseModel
import json
from hashlib import md5
from openai.lib._parsing._completions import type_to_response_format_param


logger = log_to_stdout("instructor_enrich_client")


class InstructorEnrichClient(EnrichClient):
    def __init__(self,
                 response_model: BaseModel,
                 model: str,
                 system_prompt: str = None,
                 temperature: Optional[float] = None,
                 verbosity: Optional[str] = None,
                 reasoning_effort: Optional[str] = None):
        self.response_model = response_model
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

            resp = self.client.chat.completions.create(
                response_model=self.response_model,
                messages=prompts,
                response_format=type_to_response_format_param(self.response_model)
            )
            return resp, None
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
