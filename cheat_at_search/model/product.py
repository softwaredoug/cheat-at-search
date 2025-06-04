from pydantic import BaseModel, Field
from typing import List, Optional, Literal

from cheat_at_search.model.category_list import Categories, SubCategories, FullyQualifiedCategories


class ProductCategory(BaseModel):
    """
    Base model for product categories in the furniture e-commerce space.
    Contains common category attributes.
    """
    category: Categories = Field(
        description="Category of the product, if identified. Use 'No Category Fits' if ambiguous or no category in list fits"
    )
    sub_category: SubCategories = Field(
        description="Sub-category of the product, if identified. Use 'No SubCategory Fits' if ambiguous or no sub-category in list fits"
    )


class ProductCategoryFullyQualified(BaseModel):
    """
    Base model for product categories in the furniture e-commerce space.
    Contains common category attributes.
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


class Product(BaseModel):
    """
    Base model for products in the furniture e-commerce space.
    Contains common product attributes.
    """
    id: str = Field(
        ...,
        description="Unique identifier for the product"
    )
    name: str = Field(
        ...,
        description="Name of the product"
    )
    description: str = Field(
        ...,
        description="Description of the product"
    )
    category: str = Field(
        ...,
        description="Category of the product (e.g., chair, table)"
    )
    classification: str = Field(
        ...,
        description="Classification of the product (e.g., modern, vintage)"
    )


class EnrichedProduct(Product):
    """
    Enriched representation of a product with additional attributes.
    Inherits from the base Product model.
    """
    material: Optional[str] = Field(
        None,
        description="Material of the product, if available"
    )
    color: Optional[str] = Field(
        None,
        description="Color of the product, if available"
    )
    furniture_type: str = Field(
        default="",
        description="Type of furniture mentioned in the query"
    )
    room: str = Field(
        default="",
        description="Room where the furniture would be placed, if mentioned"
    )
    dimensions: Optional[List[str]] = Field(
        None,
        description="Dimensions of the product, if available"
    )


RoomType = Literal["living room", "bedroom", "kitchen", "dining room", "office", "bathroom", "unknown",
                   "outdoor", "hallway", "entryway", "garage", "basement"]


class ProductRoom(BaseModel):
    """
    Classification of product to a room
    """
    room: RoomType = Field(
        default="unknown",
        description="Room for this furniture product (use unknown if applies to any room or unknown room type)"
    )
