import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import BaseModel, Field

from cheat_at_search.data_dir import mount
from cheat_at_search.enrich import AutoEnricher


class ColorEnrich(BaseModel):
    color: str = Field(..., description="The color of the product")


def fake_openai_response(parsed_output):
    return SimpleNamespace(
        id="resp_test_123",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=3,
        ),
        output_parsed=parsed_output,
    )


@patch("cheat_at_search.enrich.openai_enrich_client.OpenAI")
@patch("cheat_at_search.enrich.openai_enrich_client.key_for_provider")
def test_auto_enricher_openai_responses_parse_flow(mock_key_for_provider, mock_openai):
    data_dir = tempfile.mkdtemp()
    mount(use_gdrive=False, manual_path=data_dir, load_keys=False)

    mock_key_for_provider.return_value = "test-openai-key"

    mock_client = Mock()
    mock_client.responses.parse.return_value = fake_openai_response(
        ColorEnrich(color="blue")
    )
    mock_openai.return_value = mock_client

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
    mock_client.responses.parse.assert_called_once_with(
        model="gpt-4.1-nano",
        temperature=0.25,
        input=[
            {"role": "system", "content": "Classify product colors."},
            {"role": "user", "content": prompt},
        ],
        text_format=ColorEnrich,
    )

    cached_result = enricher.enrich(prompt)

    assert cached_result == ColorEnrich(color="blue")
    assert mock_client.responses.parse.call_count == 1
