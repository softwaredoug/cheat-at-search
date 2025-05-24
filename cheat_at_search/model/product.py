from pydantic import BaseModel, Field
from typing import List, Optional, Literal


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
