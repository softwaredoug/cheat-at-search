from cheat_at_search.data_dir import mount
import pytest
from pydantic import BaseModel, Field
from cheat_at_search.agent.enrich import AutoEnricher


@pytest.fixture(scope="session")
def mounted_data_dir():
    """
    Fixture to mount the data directory before running tests.
    """
    mount(use_gdrive=False, manual_path="data")
    return True


class ColorEnrich(BaseModel):
    color: str = Field(
        ...,
        description="The color to enrich",
    )


def test_simple_enrichment():

    enricher = AutoEnricher(
        model='gpt-4.1-nano',
        system_prompt="You are a helpful AI assistant that very lightly spell-checks furniture e-commerce queries.",
        output_cls=ColorEnrich
    )

    prompt = """
        What color is this product?

        blue sofa
    """

    result = enricher.enrich(prompt)
    assert result.color == "blue", f"Expected 'blue', got '{result.color}'"
