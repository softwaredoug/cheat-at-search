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
        logger=None,
    ) -> tuple[Any, list]:
        pass


def record_resource(agent_state: dict, name: str, amount: int | float) -> None:
    """Accumulate one usage/resource amount in agent state."""
    key = f"resource.{name.removeprefix('resource.')}"
    agent_state[key] = agent_state.get(key, 0) + amount


def initialize_resource_counters(agent_state: dict) -> None:
    """Ensure the standard model-usage counters exist in agent state."""
    for name in ("input_tokens", "output_tokens", "num_tool_calls"):
        agent_state.setdefault(f"resource.{name}", 0)
