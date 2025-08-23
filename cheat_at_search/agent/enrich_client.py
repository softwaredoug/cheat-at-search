from abc import ABC, abstractmethod
import requests
from typing import Optional
from pydantic import BaseModel
import json
from hashlib import md5
from typing import Tuple


class EnrichClient(ABC):
    @abstractmethod
    def enrich(self, prompt: str, task_id: str = None) -> Optional[BaseModel]:
        pass

    def str_hash(self):
        pass

    def get_num_tokens(self, prompt: str) -> Tuple[int, int]:
        pass
