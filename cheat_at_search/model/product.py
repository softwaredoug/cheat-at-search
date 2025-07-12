from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from cheat_at_search.cache import StoredLruCache
import numpy as np

from cheat_at_search.model.category_list import Categories, SubCategories, FullyQualifiedCategories
from sentence_transformers import SentenceTransformer


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


model = SentenceTransformer('all-MiniLM-L6-v2')


@StoredLruCache(maxsize=50000)
def encode(text):
    return model.encode(text)


class BrandedTerms(BaseModel):
    """
    Represents a classification of a product.

    In this case, hallucinated, something the model is making up that looks like one
    of our classifications
    """
    branded_terms: list[str] = Field(
        description="Any branded terms (product lines, brand names, etc) mentioned"
    )


ItemTypes = Literal[
    "area rug", "accent pillow", "bed", "cocktail table", "floor & wall tile",
    "entertainment center", "kitchen mat", "sectional", "sofa", "patio sofa",
    "doormat", "furniture cushion", "wall clock", "garden statue",
    "kitchen island", "garment rack", "mattress pad",
    "loveseat", "armchair", "recliner", "coffee table", "end table",
    "tv stand", "media console", "bookshelf", "bed frame", "mattress", "nightstand",
    "dresser", "wardrobe", "chest of drawers", "dining table", "dining chair", "bar stool",
    "sideboard", "buffet", "bench", "office chair", "desk", "filing cabinet", "bookcase",
    "patio chair", "patio table", "outdoor sofa", "umbrella", "grill", "toolbox",
    "door knob", "door lock", "deadbolt", "light switch", "outlet", "extension cord",
    "smart bulb", "ceiling fan", "floor lamp", "table lamp", "chandelier", "rug",
    "curtains", "blinds", "shower curtain", "mirror", "wall art", "picture frame",
    "clock", "candle holder", "vase", "planter", "kitchen faucet", "sink", "toilet",
    "bathroom faucet", "chaise lounge",
    "shower head", "plunger", "broom", "dustpan", "mop", "bucket", "vacuum", "trash can",
    "recycling bin", "laundry basket", "ironing board", "drying rack", "cutlery", "slow cooker",
    "frying pan", "saucepan", "mixing bowl", "cutting board", "storage bin", "shelving unit",
    "no item type matches", "unknown", "ottoman", "comforter", "chair cushion", "refrigerator",
    "greenhouse", "crown molding", "vanity", "flag", "potted plant", "basket", "podium",
    "blanket", "anti-fatigue mat", "serving tray"
]


class ItemType(BaseModel):
    """
    Represents a classification of a product.

    In this case, hallucinated, something the model is making up that looks like one
    of our classifications
    """
    @property
    def similarity(self):
        """Compare item_type to item_type_unconstrained"""
        return np.dot(encode(self.item_type), encode(self.item_type_unconstrained))

    @property
    def item_type_same(self):
        """Check if item_type matches item_type_unconstrained"""
        THRESHOLD = 0.5
        if self.similarity < THRESHOLD:
            return "no item type matches"
        return self.item_type

    item_type: ItemTypes = Field(
        ...,
        description="The type of item this product is from the provided list. Use 'no item type matches' if no item type matches the item"
    )
    item_type_unconstrained: str = Field(
        ...,
        description="The type of item this product is, ie dining table, bed, etc"
    )


class Material(BaseModel):
    """
    Represents the material of a product.
    """
    materials: list[str] = Field(
        ...,
        description="Materials of the product, if available"
    )
