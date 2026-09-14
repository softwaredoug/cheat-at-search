from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel

from cheat_at_search.agent.openai_agent import (
    OpenAIAgent,
    _maybe_get_image_input,
)


class ImageResult(BaseModel):
    image_url: str


def test_maybe_get_image_input_from_dict():
    assert _maybe_get_image_input({"image_url": "https://example.com/image.png"}) == {
        "type": "input_image",
        "image_url": "https://example.com/image.png",
    }


def test_maybe_get_image_input_from_base_model():
    assert _maybe_get_image_input(ImageResult(image_url="https://example.com/image.png")) == {
        "type": "input_image",
        "image_url": "https://example.com/image.png",
    }


def test_maybe_get_image_input_ignores_missing_or_invalid_urls():
    assert _maybe_get_image_input({}) is None
    assert _maybe_get_image_input({"image_url": "  "}) is None
    assert _maybe_get_image_input({"image_url": None}) is None
    assert _maybe_get_image_input("https://example.com/image.png") is None


def _response(*output):
    return SimpleNamespace(
        output=list(output),
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )


def _function_call():
    return SimpleNamespace(
        type="function_call",
        name="_get_image",
        arguments="{}",
        call_id="call_1",
    )


def _get_image() -> dict:
    """Return an image result."""
    return {"image_url": "https://example.com/image.png"}


@patch("cheat_at_search.agent.openai_agent.OpenAI")
@patch("cheat_at_search.agent.openai_agent.key_for_provider", return_value="test-key")
def test_process_images_adds_image_context_without_changing_tool_output(
    mock_key_for_provider, mock_openai
):
    mock_openai.return_value.responses.create.side_effect = [
        _response(_function_call()),
        _response(),
    ]
    agent = OpenAIAgent(
        tools=[_get_image],
        model="openai/gpt-5",
        process_images=True,
    )

    _, inputs, _ = agent.chat(inputs=[])

    assert inputs[-2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"image_url":"https://example.com/image.png"}',
    }
    assert inputs[-1] == {
        "role": "user",
        "content": [
            {
                "type": "input_image",
                "image_url": "https://example.com/image.png",
            }
        ],
    }


@patch("cheat_at_search.agent.openai_agent.OpenAI")
@patch("cheat_at_search.agent.openai_agent.key_for_provider", return_value="test-key")
def test_process_images_defaults_to_disabled(mock_key_for_provider, mock_openai):
    mock_openai.return_value.responses.create.side_effect = [
        _response(_function_call()),
        _response(),
    ]
    agent = OpenAIAgent(tools=[_get_image], model="openai/gpt-5")

    _, inputs, _ = agent.chat(inputs=[])

    assert len(inputs) == 2
    assert inputs[-1]["type"] == "function_call_output"
    assert isinstance(inputs[-1]["output"], str)
