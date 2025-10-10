from abc import ABC, abstractmethod
from typing import Literal
from pydantic import BaseModel, Field


RelevanceLevels = Literal['🤩', '🙂', '😐', '😭']


class SearchResult(BaseModel):
    """A single search result row."""
    id: str = Field(
        ..., description="The document identifier (product, page, etc.)"
    )

    name: str = Field(
        ..., description="The title of the product or document returned from search"
    )

    rank: int = Field(
        ..., description="The rank of this document in the search results (1 is best)"
    )

    relevance: RelevanceLevels = Field(
        ..., description="The relevance of this document to the search query"
    )

    @property
    def score(self) -> float:
        """A numeric score for the relevance of this result."""
        return 1 / self.rank


class SearchResults(BaseModel):
    """The results of a search query, ordered by relevance."""
    query: str = Field(
        ..., description="Original search query passed by the user"
    )

    intent_explained: str = Field(
        ..., description="Summary of the user's intent based on the search query"
    )

    results: list[SearchResult] = Field(
        ..., description="The ranked search results"
    )

    self_evaluation: RelevanceLevels = Field(
        ..., description="Evaluation of overall search results quality."
    )


class SearchClient(ABC):
    """Use an MCP server to search for products"""

    @abstractmethod
    def search(self, prompt: str) -> SearchResults:
        pass
