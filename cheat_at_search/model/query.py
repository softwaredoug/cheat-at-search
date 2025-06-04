from pydantic import BaseModel, Field
from typing import List, Literal
from cheat_at_search.model.category_list import Categories, CategoriesReversed, SubCategories, FullyQualifiedCategories


class Query(BaseModel):
    """
    Base model for search queries, containing common query attributes.
    """
    keywords: str = Field(
        ...,
        description="The original search query keywords sent in as input"
    )


class SynonymMapping(BaseModel):
    """
    Model for mapping phrases in the query to equivalent phrases or synonyms.
    """
    phrase: str = Field(
        ...,
        description="The original phrase from the query"
    )
    synonyms: List[str] = Field(
        ...,
        description="List of synonyms or equivalent phrases for the original phrase"
    )


class QueryWithSynonyms(Query):
    """
    Extended model for search queries that includes synonyms for keywords.
    Inherits from the base Query model.
    """
    synonyms: List[SynonymMapping] = Field(
        ...,
        description="Mapping of phrases in the query to equivalent phrases or synonyms"
    )


class SpellingCorrectedQuery(Query):
    """
    Model for search queries with spelling corrections applied.
    Inherits keywords from the base Query model.
    """
    corrected_keywords: str = Field(
        ...,
        description="Identical to original query string, but with spelling corrections applied"
    )


class BucketedQuery(Query):
    """
    Extended model for search queries that includes synonyms for keywords.
    Inherits from the base Query model.
    """
    information_need: Literal["navigation", "exploration"] = Field(
        default_factory=str,
        description="Information need of the query, either 'navigation' (go to specific product) or 'exploration' (browse products)"
    )


class QueryCategory(Query):
    """
    Structured representation of a search query for furniture e-commerce.
    Inherits keywords from the base Query model and adds category and sub-category.
    """
    category: Categories = Field(
        description="Category of the product, if identified. Use 'No Category Fits' if ambiguous or no category in list fits"
    )
    sub_category: SubCategories = Field(
        description="Sub-category of the product, if identified. Use 'No SubCategory Fits' if ambiguous or no sub-category in list fits"
    )


class QueryCategoryReversed(Query):
    """
    Structured representation of a search query for furniture e-commerce.
    Inherits keywords from the base Query model and adds category and sub-category.
    """
    category: CategoriesReversed = Field(
        description="Category of the product, if identified. Use 'No Category Fits' if ambiguous or no category in list fits"
    )
    sub_category: SubCategories = Field(
        description="Sub-category of the product, if identified. Use 'No SubCategory Fits' if ambiguous or no sub-category in list fits"
    )


class QueryCategoryFullyQualified(Query):
    """
    Fully qualified search query that includes all structured information.
    """
    full_category: FullyQualifiedCategories = Field(
        description="Fully qualified category of the product, if identified. Use 'No Category Fits' if ambiguous or no category in list fits. Options ordered from most common product categories to least common"
    )

    @property
    def category(self) -> str:
        return self.full_category.split('/')[0].strip()

    @property
    def sub_category(self) -> str:
        parts = self.full_category.split('/')
        return parts[1].strip() if len(parts) > 1 else 'No SubCategory Fits'


class StructuredQuery(BaseModel):
    """
    Structured representation of a search query for furniture e-commerce.
    Inherits keywords from the base Query model.
    """
    search_terms: str = Field(
        default="",
        description="A rebuilt / better search query to use to search the product catalog"
    )
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
