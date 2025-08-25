from cheat_at_search.data_dir import mount
import pytest
from pydantic import BaseModel, Field
from cheat_at_search.agent.enrich import AutoEnricher
import tempfile
from typing import Literal


models_to_test = [
    "openai/gpt-4.1-nano",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1",
    "anthropic/claude-sonnet-4-20250514"
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


class ColorEnrich(BaseModel):
    color: str = Field(
        ...,
        description="The color to enrich",
    )


class ColorEnrichLiteral(BaseModel):
    color: Literal['red', 'blue', 'green'] = Field(
        ...,
        description="The color to enrich, must be one of 'red', 'blue', or 'green'.",
    )


@pytest.mark.parametrize("model", models_to_test)
@pytest.mark.parametrize("response_model", [ColorEnrich, ColorEnrichLiteral])
def test_simple_enrichment(mounted_data_dir, response_model, model):

    enricher = AutoEnricher(
        model=model,
        system_prompt="You are a helpful AI assistant that classifies e-commerce products",
        response_model=response_model,
    )

    prompt = """
        What color is this product?

        blue sofa
    """

    result = enricher.enrich(prompt)
    assert result.color.lower() == "blue", f"Expected 'blue', got '{result.color}'"


@pytest.mark.parametrize("model", models_to_test)
@pytest.mark.parametrize("response_model", [ColorEnrich, ColorEnrichLiteral])
def test_debug(mounted_data_dir, response_model, model):

    enricher = AutoEnricher(
        model=model,
        system_prompt="You are a helpful AI assistant that classifies e-commerce products",
        response_model=response_model,
    )

    prompt = """
        What color is this product?

        blue sofa
    """

    debug_meta = enricher.debug(prompt)
    assert debug_meta.prompt_tokens > 0
    assert debug_meta.completion_tokens > 0
    result = debug_meta.output
    assert result.color.lower() == "blue", f"Expected 'blue', got '{result.color}'"


@pytest.mark.parametrize("model", models_to_test)
def test_simple_enrich_literal_respects_literals(mounted_data_dir, model):

    enricher = AutoEnricher(
        model=model,
        system_prompt="You are a helpful AI assistant that classifies e-commerce products",
        response_model=ColorEnrichLiteral
    )

    prompt = """
        What color is this product?

        purple sofa
    """

    result = enricher.enrich(prompt)
    assert result.color in ['red', 'blue', 'green'], f"Expected one of ['red', 'blue', 'green'], got '{result.color}'"
