from pydantic import BaseModel, Field
from typing import List


class Query(BaseModel):
    """
    Base model for search queries, containing common query attributes.
    """
    keywords: List[str] = Field(
        default_factory=list,
        description="List of keywords extracted from the query"
    )


class UnderstoodQuery(Query):
    """
    Structured representation of a search query for furniture e-commerce.
    Inherits keywords from the base Query model.
    """
    material: str = Field(
        default="",
        description="Material extracted from the query, or empty string if none found"
    )
    color: str = Field(
        default="",
        description="Color mentioned in the query, or empty string if none found"
    )
    furniture_type: str = Field(
        default="",
        description="Type of furniture mentioned in the query"
    )
    room: str = Field(
        default="",
        description="Room where the furniture would be placed, if mentioned"
    )
    dimensions: List[str] = Field(
        default_factory=list,
        description="Any dimensions mentioned in the query"
    )