import tempfile
from typing import Literal, get_args
from unittest.mock import patch

from pydantic import BaseModel, Field

from cheat_at_search.data_dir import mount
from cheat_at_search.enrich import AutoEnricher, DataframeEnricher
from cheat_at_search.wands_data import products

from .openai_responses_mock import MockOpenAIResponses, responses_api_response


Rooms = Literal[
    "Living Room",
    "Dining Room",
    "Outdoor",
    "Bedroom",
    "Dining Room",
    "Kitchen",
    "Office",
    "Bathroom",
    "No Room Fits",
]

rooms_as_list = list(get_args(Rooms))


class Room(BaseModel):
    """Represents the room this furniture product goes in."""

    room: Literal[Rooms] = Field(description="The room this product belongs to")


def configure_mock_openai_enricher(mock_key_for_provider, mock_openai):
    data_dir = tempfile.mkdtemp()
    mount(use_gdrive=False, manual_path=data_dir, load_keys=False)
    mock_key_for_provider.return_value = "test-openai-key"
    mock_responses = MockOpenAIResponses()
    mock_openai.return_value = mock_responses.client
    return mock_responses


def get_room_prompt(row) -> str:
    return f"""
I am going to give you a furniture e-commerce product.

You tell me which of the listed room it belongs to. Or if ambiguous, could fit in multiple rooms, or unclear, return 'No Room Fits'

Default to 'No Room Fits' unless given compelling evidence.

Rugs can go in any room - they should get 'No Room Fits'
Hardware can go in any room - they should get 'No Room Fits'
Most decor can go in any room - they should get 'No Room Fits'
If multiple rooms are mentioned - they should get 'No Room Fits'

Product Name: {row['product_name']}
Description: {row['product_description']}
        """


def room_response_from_prompt(**kwargs):
    prompt = kwargs["input"][-1]["content"].lower()
    if "bed" in prompt or "mattress" in prompt:
        room = "Bedroom"
    elif "desk" in prompt or "office" in prompt:
        room = "Office"
    elif "dining" in prompt or "table" in prompt:
        room = "Dining Room"
    elif "outdoor" in prompt or "patio" in prompt:
        room = "Outdoor"
    elif "sofa" in prompt or "chair" in prompt:
        room = "Living Room"
    else:
        room = "No Room Fits"
    return responses_api_response(output_parsed=Room(room=room))


@patch("cheat_at_search.enrich.openai_enrich_client.OpenAI")
@patch("cheat_at_search.enrich.openai_enrich_client.key_for_provider")
def test_product_enrichment_one(mock_key_for_provider, mock_openai):
    mock_responses = configure_mock_openai_enricher(mock_key_for_provider, mock_openai)
    mock_responses.queue_parse_response(payload={"room": "Living Room"})
    room_enricher = AutoEnricher(
        model="openai/gpt-4.1-nano",
        system_prompt="You are a helpful furniture, hardware, and home-goods ecommerce shopping assistant that understands furniture products",
        response_model=Room,
    )
    dataframe_enricher = DataframeEnricher(
        enricher=room_enricher,
        prompt_fn=get_room_prompt,
    )

    room = dataframe_enricher.enrich_one(products.iloc[0].to_dict())

    assert room.room == "Living Room"
    assert room.room in rooms_as_list
    mock_openai.assert_called_once_with(api_key="test-openai-key")
    parse_kwargs = mock_responses.client.responses.parse.call_args.kwargs
    assert parse_kwargs["model"] == "gpt-4.1-nano"
    assert parse_kwargs["text_format"] is Room
    assert "Product Name:" in parse_kwargs["input"][-1]["content"]


@patch("cheat_at_search.enrich.openai_enrich_client.OpenAI")
@patch("cheat_at_search.enrich.openai_enrich_client.key_for_provider")
def test_product_enrichment_dataframe(mock_key_for_provider, mock_openai):
    mock_responses = configure_mock_openai_enricher(mock_key_for_provider, mock_openai)
    mock_responses.client.responses.parse.side_effect = room_response_from_prompt
    room_enricher = AutoEnricher(
        model="openai/gpt-4.1-nano",
        system_prompt="You are a helpful furniture, hardware, and home-goods ecommerce shopping assistant that understands furniture products",
        response_model=Room,
    )
    dataframe_enricher = DataframeEnricher(
        enricher=room_enricher,
        prompt_fn=get_room_prompt,
    )
    rows = products[:7]

    enriched_products = dataframe_enricher.enrich_all(rows, workers=1, batch_size=3)

    existing_rooms = enriched_products["room"].dropna().unique().tolist()
    assert len(enriched_products["room"].dropna()) == len(rows)
    for actual_room in existing_rooms:
        assert actual_room in rooms_as_list
    assert mock_responses.client.responses.parse.call_count == len(rows)
