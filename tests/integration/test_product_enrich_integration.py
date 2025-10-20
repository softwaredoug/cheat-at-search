from cheat_at_search.wands_data import products
from cheat_at_search.enrich import AutoEnricher, ProductEnricher
import pytest
import tempfile
from cheat_at_search.data_dir import mount
from pydantic import BaseModel, Field
from typing import Literal, get_args


models_to_test = [
    "openai/gpt-4.1-nano",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1",
    "google/gemini-2.5-flash-lite",
]


@pytest.fixture(scope="module")
def mounted_data_dir():
    """
    Fixture to mount the data directory before running tests.
    """
    data_dir = tempfile.mkdtemp()
    print(f"Mounting data directory at: {data_dir}")
    mount(use_gdrive=False, manual_path=data_dir, load_keys=False)
    return True


Rooms = Literal[
    'Living Room',
    'Dining Room',
    'Outdoor',
    'Bedroom',
    'Dining Room',
    'Kitchen',
    'Office',
    'Bathroom',
    'No Room Fits'
]

rooms_as_list = list(get_args(Rooms))


class Room(BaseModel):
    """
    Represents the room this furniture product goes in.
    """
    room: Literal[Rooms] = Field(
        description="The room this product belongs to"
    )


@pytest.mark.parametrize("model", models_to_test)
def test_product_enrichment_one(model, mounted_data_dir):
    room_enricher = AutoEnricher(
        model=model,
        system_prompt="You are a helpful furniture, hardware, and home-goods ecommerce shopping assistant that understands furniture products",
        response_model=Room
    )

    def get_room_prompt(product) -> str:
        prompt = f"""
    I am going to give you a furniture e-commerce product.

    You tell me which of the listed room it belongs to. Or if ambiguous, could fit in multiple rooms, or unclear, return 'No Room Fits'

    Default to 'No Room Fits' unless given compelling evidence.

    Rugs can go in any room - they should get 'No Room Fits'
    Hardware can go in any room - they should get 'No Room Fits'
    Most decor can go in any room - they should get 'No Room Fits'
    If multiple rooms are mentioned - they should get 'No Room Fits'

    Product Name: {product['product_name']}
    Description: {product['product_description']}
            """
        return prompt

    product_enricher = ProductEnricher(
        enricher=room_enricher,
        prompt_fn=get_room_prompt,
    )

    room = product_enricher.enrich_one(products.iloc[0].to_dict())
    assert room.room in rooms_as_list, f"Unexpected room: {room.room}"


@pytest.mark.parametrize("df_size", [20, 23, 50])
@pytest.mark.parametrize("model", models_to_test)
def test_product_enrichment(model, df_size, mounted_data_dir):
    room_enricher = AutoEnricher(
        model=model,
        system_prompt="You are a helpful furniture, hardware, and home-goods ecommerce shopping assistant that understands furniture products",
        response_model=Room
    )

    def get_room_prompt(product) -> str:
        prompt = f"""
    I am going to give you a furniture e-commerce product.

    You tell me which of the listed room it belongs to. Or if ambiguous, could fit in multiple rooms, or unclear, return 'No Room Fits'

    Default to 'No Room Fits' unless given compelling evidence.

    Rugs can go in any room - they should get 'No Room Fits'
    Hardware can go in any room - they should get 'No Room Fits'
    Most decor can go in any room - they should get 'No Room Fits'
    If multiple rooms are mentioned - they should get 'No Room Fits'

    Product Name: {product['product_name']}
    Description: {product['product_description']}
            """
        return prompt

    product_enricher = ProductEnricher(
        enricher=room_enricher,
        prompt_fn=get_room_prompt,
    )

    enriched_products = product_enricher.enrich_all(products[:df_size], workers=2, batch_size=5)
    existing_rooms = enriched_products['room'].dropna().unique().tolist()
    assert len(enriched_products['room'].dropna()) == df_size, "Some products were not enriched with a room"
    for actual_room in existing_rooms:
        assert actual_room in rooms_as_list, f"Unexpected room: {actual_room}"
