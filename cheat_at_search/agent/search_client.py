from abc import ABC, abstractmethod
from typing import Literal
from pydantic import BaseModel, Field


class SearchPlan(BaseModel):
    """Describe the parameters you passed to the search tool to get this result."""
    query: str = Field(
        ..., description="The actual query string you used to search (may be different from the original user query if reformulated)"
    )

    effectiveness: Literal['high', 'medium', 'low'] = Field(
        ..., description="How effective was this search plan in retrieving relevant results?"
    )


class SearchResult(BaseModel):
    """A single search result row."""
    id: str = Field(
        ..., description="The document identifier (product, page, etc.)"
    )

    name: str = Field(
        ..., description="The title / name of the product or document returned from search"
    )

    reasoning: str = Field(
        ..., description="Why was this document returned at the given relevance level?"
    )

    search_plan: SearchPlan = Field(
        ..., description="The search parameters used to retrieve this document"
    )

    rank: int = Field(
        ..., description="The rank of this document in the search results (1 is best)"
    )

    relevance: Literal['exact', 'partial', 'irrelevant'] = Field(
        ..., description="The relevance of this document to the search query"
    )


class SearchResults(BaseModel):
    """The results of a search query, ordered by relevance."""
    query: str = Field(
        ..., description="The original search query passed by the user"
    )

    search_plans: list[SearchPlan] = Field(
        ..., description="The list of search plans used to retrieve results"
    )

    results: list[SearchResult] = Field(
        ..., description="The list of search results"
    )


class SearchClient(ABC):
    """Use an MCP server to search for products"""

    @abstractmethod
    def search(self, prompt: str) -> SearchResults:
        pass
