import tempfile
from pathlib import Path
from typing import Literal
from unittest.mock import patch

from pydantic import BaseModel, Field

from cheat_at_search.data_dir import mount
from cheat_at_search.enrich import AutoEnricher

from .openai_responses_mock import MockOpenAIResponses, responses_api_response


class ColorEnrich(BaseModel):
    color: str = Field(..., description="The color of the product")


class ColorEnrichLiteral(BaseModel):
    color: Literal["red", "blue", "green"] = Field(
        ...,
        description="The color to enrich, must be one of red, blue, or green.",
    )


def configure_mock_openai_enricher(mock_key_for_provider, mock_openai):
    data_dir = tempfile.mkdtemp()
    mount(use_gdrive=False, manual_path=data_dir, load_keys=False)
    mock_key_for_provider.return_value = "test-openai-key"
    mock_responses = MockOpenAIResponses()
    mock_openai.return_value = mock_responses.client
    return mock_responses


def color_response_from_prompt(**kwargs):
    user_prompt = kwargs["input"][-1]["content"]
    color = user_prompt.split()[-2]
    output = kwargs["text_format"](color=color)
    return responses_api_response(output_parsed=output)


@patch("cheat_at_search.enrich.openai_enrich_client.OpenAI")
@patch("cheat_at_search.enrich.openai_enrich_client.key_for_provider")
def test_auto_enricher_openai_responses_parse_flow(mock_key_for_provider, mock_openai):
    mock_responses = configure_mock_openai_enricher(mock_key_for_provider, mock_openai)
    mock_responses.queue_parse_response(payload={"color": "blue"})

    enricher = AutoEnricher(
        model="openai/gpt-4.1-nano",
        system_prompt="Classify product colors.",
        response_model=ColorEnrich,
        temperature=0.25,
    )

    prompt = "What color is this product?\n\nblue sofa"

    result = enricher.enrich(prompt)

    assert result == ColorEnrich(color="blue")
    mock_openai.assert_called_once_with(api_key="test-openai-key")
    mock_responses.client.responses.parse.assert_called_once_with(
        model="gpt-4.1-nano",
        temperature=0.25,
        input=[
            {"role": "system", "content": "Classify product colors."},
            {"role": "user", "content": prompt},
        ],
        text_format=ColorEnrich,
    )


@patch("cheat_at_search.enrich.openai_enrich_client.OpenAI")
@patch("cheat_at_search.enrich.openai_enrich_client.key_for_provider")
def test_auto_enricher_debug_returns_metadata(mock_key_for_provider, mock_openai):
    mock_responses = configure_mock_openai_enricher(mock_key_for_provider, mock_openai)
    mock_responses.queue_parse_response(
        payload={"color": "blue"},
        input_tokens=12,
        output_tokens=4,
        response_id="resp_debug_123",
    )

    enricher = AutoEnricher(
        model="openai/gpt-4.1-nano",
        system_prompt="Classify product colors.",
        response_model=ColorEnrich,
    )

    debug_meta = enricher.debug("What color is this product?\n\nblue sofa")

    assert debug_meta.model == "gpt-4.1-nano"
    assert debug_meta.prompt_tokens == 12
    assert debug_meta.completion_tokens == 4
    assert debug_meta.response_id == "resp_debug_123"
    assert debug_meta.output == ColorEnrich(color="blue")


@patch("cheat_at_search.enrich.cached_enrich_client.json.load")
@patch("cheat_at_search.enrich.openai_enrich_client.OpenAI")
@patch("cheat_at_search.enrich.openai_enrich_client.key_for_provider")
def test_auto_enricher_recovers_from_corrupt_cache(
    mock_key_for_provider,
    mock_openai,
    mock_json_load,
):
    mock_json_load.side_effect = ValueError("corrupt cache")
    mock_responses = configure_mock_openai_enricher(mock_key_for_provider, mock_openai)
    mock_responses.queue_parse_response(payload={"color": "blue"})

    enricher = AutoEnricher(
        model="openai/gpt-4.1-nano",
        system_prompt="Classify product colors.",
        response_model=ColorEnrich,
    )

    cache_path = Path(enricher.cached_enricher.cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{\"stale\": \"data\"}")

    enricher.cached_enricher.load_cache()

    assert not cache_path.exists()
    assert enricher.enrich("What color is this product?\n\nblue sofa") == ColorEnrich(
        color="blue"
    )


@patch("cheat_at_search.enrich.openai_enrich_client.OpenAI")
@patch("cheat_at_search.enrich.openai_enrich_client.key_for_provider")
def test_auto_enricher_repeated_prompt_uses_cache(mock_key_for_provider, mock_openai):
    mock_responses = configure_mock_openai_enricher(mock_key_for_provider, mock_openai)
    mock_responses.queue_parse_response(payload={"color": "blue"})
    mock_responses.queue_parse_response(payload={"color": "red"})

    enricher = AutoEnricher(
        model="openai/gpt-4.1-nano",
        system_prompt="Classify product colors.",
        response_model=ColorEnrich,
    )

    prompt = "What color is this product?\n\nblue sofa"

    first_result = enricher.enrich(prompt)
    second_result = enricher.enrich(prompt)

    assert first_result == ColorEnrich(color="blue")
    assert second_result == first_result
    assert mock_responses.client.responses.parse.call_count == 1


@patch("cheat_at_search.enrich.openai_enrich_client.OpenAI")
@patch("cheat_at_search.enrich.openai_enrich_client.key_for_provider")
def test_auto_enricher_literal_respects_structured_output(mock_key_for_provider, mock_openai):
    mock_responses = configure_mock_openai_enricher(mock_key_for_provider, mock_openai)
    mock_responses.queue_parse_response(payload={"color": "red"})

    enricher = AutoEnricher(
        model="openai/gpt-4.1-nano",
        system_prompt="Classify product colors.",
        response_model=ColorEnrichLiteral,
    )

    result = enricher.enrich("What color is this product?\n\npurple sofa")

    assert result == ColorEnrichLiteral(color="red")
    parse_kwargs = mock_responses.client.responses.parse.call_args.kwargs
    assert parse_kwargs["text_format"] is ColorEnrichLiteral


@patch("cheat_at_search.enrich.openai_enrich_client.OpenAI")
@patch("cheat_at_search.enrich.openai_enrich_client.key_for_provider")
def test_auto_enricher_enrich_all_basic(mock_key_for_provider, mock_openai):
    mock_responses = configure_mock_openai_enricher(mock_key_for_provider, mock_openai)
    mock_responses.client.responses.parse.side_effect = color_response_from_prompt

    enricher = AutoEnricher(
        model="openai/gpt-4.1-nano",
        system_prompt="Classify product colors.",
        response_model=ColorEnrich,
    )

    prompts = [
        "What color is this product? blue sofa",
        "What color is this product? red chair",
        "What color is this product? green table",
    ]

    results = enricher.enrich_all(prompts, workers=1, batch_size=10)

    assert results == [
        ColorEnrich(color="blue"),
        ColorEnrich(color="red"),
        ColorEnrich(color="green"),
    ]
    assert mock_responses.client.responses.parse.call_count == len(prompts)


@patch("cheat_at_search.enrich.openai_enrich_client.OpenAI")
@patch("cheat_at_search.enrich.openai_enrich_client.key_for_provider")
def test_auto_enricher_enrich_all_handles_multiple_batches(mock_key_for_provider, mock_openai):
    mock_responses = configure_mock_openai_enricher(mock_key_for_provider, mock_openai)
    seen_prompts = []

    def record_and_respond(**kwargs):
        seen_prompts.append(kwargs["input"][-1]["content"])
        return color_response_from_prompt(**kwargs)

    mock_responses.client.responses.parse.side_effect = record_and_respond

    enricher = AutoEnricher(
        model="openai/gpt-4.1-nano",
        system_prompt="Classify product colors.",
        response_model=ColorEnrich,
    )

    prompts = [
        "What color is this product? blue sofa",
        "What color is this product? red chair",
        "What color is this product? green table",
        "What color is this product? yellow lamp",
        "What color is this product? purple rug",
    ]

    results = enricher.enrich_all(prompts, workers=1, batch_size=2)

    assert results == [
        ColorEnrich(color="blue"),
        ColorEnrich(color="red"),
        ColorEnrich(color="green"),
        ColorEnrich(color="yellow"),
        ColorEnrich(color="purple"),
    ]
    assert seen_prompts == prompts
    assert mock_responses.client.responses.parse.call_count == len(prompts)


@patch("cheat_at_search.enrich.openai_enrich_client.OpenAI")
@patch("cheat_at_search.enrich.openai_enrich_client.key_for_provider")
def test_auto_enricher_gpt5_uses_reasoning_and_verbosity(mock_key_for_provider, mock_openai):
    mock_responses = configure_mock_openai_enricher(mock_key_for_provider, mock_openai)
    mock_responses.queue_parse_response(payload={"color": "blue"})

    enricher = AutoEnricher(
        model="openai/gpt-5-nano",
        system_prompt="Classify product colors.",
        response_model=ColorEnrich,
    )

    result = enricher.enrich("What color is this product?\n\nblue sofa")

    assert result == ColorEnrich(color="blue")
    mock_responses.client.responses.parse.assert_called_once_with(
        model="gpt-5-nano",
        reasoning={"effort": "minimal"},
        input=[
            {"role": "system", "content": "Classify product colors."},
            {"role": "user", "content": "What color is this product?\n\nblue sofa"},
        ],
        text_format=ColorEnrich,
        text={"verbosity": "low"},
    )
