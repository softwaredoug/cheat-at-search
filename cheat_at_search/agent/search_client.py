from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field


class SearchResults(BaseModel):
    """The state of the search agent, which can be used to inform future reasoning and tool use."""

    ranked_results: list[str] = Field(
        description="Top ranked search results (their doc_ids) when complete"
    )


class Agent(ABC):
    """Provider-neutral interface for a tool-using search agent."""

    @abstractmethod
    def loop(
        self,
        inputs=None,
        agent_state: dict | None = None,
        return_usage: bool = False,
        logger=None,
    ) -> tuple[Any, list, dict]:
        pass
