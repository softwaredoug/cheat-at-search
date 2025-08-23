from cheat_at_search.data_dir import mount
import pytest
from pydantic import BaseModel, Field
from cheat_at_search.agent.enrich import AutoEnricher
import tempfile
from typing import Literal


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


@pytest.mark.parametrize("output_cls", [ColorEnrich, ColorEnrichLiteral])
def test_simple_enrichment(mounted_data_dir, output_cls):

    enricher = AutoEnricher(
        model='gpt-4.1-nano',
        system_prompt="You are a helpful AI assistant that very lightly spell-checks furniture e-commerce queries.",
        output_cls=output_cls,
    )

    prompt = """
        What color is this product?

        blue sofa
    """

    result = enricher.enrich(prompt)
    assert result.color == "blue", f"Expected 'blue', got '{result.color}'"


def test_simple_enrich_literal_respects_literals(mounted_data_dir):

    enricher = AutoEnricher(
        model='gpt-4.1-nano',
        system_prompt="You are a helpful AI assistant that very lightly spell-checks furniture e-commerce queries.",
        output_cls=ColorEnrichLiteral
    )

    prompt = """
        What color is this product?

        purple sofa
    """

    result = enricher.enrich(prompt)
    assert result.color in ['red', 'blue', 'green'], f"Expected one of ['red', 'blue', 'green'], got '{result.color}'"
