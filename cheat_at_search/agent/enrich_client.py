from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel


class DebugMetaData:
    def __init__(self, model: str,
                 prompt_tokens: int,
                 completion_tokens: int,
                 reasoning_tokens: int,
                 response_id: Optional[str] = None,
                 output: BaseModel = None):
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.reasoning_tokens = reasoning_tokens
        self.total_tokens = prompt_tokens + completion_tokens + reasoning_tokens
        self.response_id = response_id
        self.output = output

    def __repr__(self):
        return f"DebugMetaData(model={self.model}, prompt_tokens={self.prompt_tokens}, completion_tokens={self.completion_tokens}, reasoning_tokens={self.reasoning_tokens}, total_tokens={self.total_tokens}, response_id={self.response_id}, output={self.output})"


class EnrichClient(ABC):
    @abstractmethod
    def enrich(self, prompt: str) -> Optional[BaseModel]:
        pass

    @abstractmethod
    def debug(self, prompt: str) -> Optional[DebugMetaData]:
        pass

    @abstractmethod
    def str_hash(self) -> str:
        pass

    @property
    def response_model(self):
        pass
