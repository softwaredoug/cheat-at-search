from typing import Annotated
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel, Field

from cheat_at_search.agent.openai_agent import (
    OpenAIAgent,
    _maybe_get_image_input,
)
from cheat_at_search.agent.sid_legacy_agent import SIDLegacyAgent


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

    _, inputs = agent.loop(inputs=[])

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

    _, inputs = agent.loop(inputs=[])

    assert len(inputs) == 2
    assert inputs[-1]["type"] == "function_call_output"
    assert isinstance(inputs[-1]["output"], str)


def _annotated_search(
    query: Annotated[str, "Terms describing what to search for"],
    limit: Annotated[int, Field(description="Maximum number of results", ge=1, le=10)] = 5,
) -> list[dict]:
    """Search documents for a query."""
    return []


def read(id: Annotated[str, "Stable document ID"]) -> dict:
    """Read a document by its stable ID."""
    return {"id": id, "title": "", "description": ""}


@patch("cheat_at_search.agent.openai_agent.OpenAI")
@patch("cheat_at_search.agent.openai_agent.key_for_provider", return_value="openai-test-key")
@patch("cheat_at_search.agent.sid_legacy_agent.OpenAI")
@patch("cheat_at_search.agent.sid_legacy_agent.key_for_provider", return_value="sid-test-key")
def test_sid_legacy_tools_are_accepted_by_openai_with_annotated_descriptions(
    mock_sid_key_for_provider,
    mock_sid_openai,
    mock_openai_key_for_provider,
    mock_openai,
):
    tools = [_annotated_search, read]
    sid_agent = SIDLegacyAgent(tools=tools, max_turns=2)
    openai_agent = OpenAIAgent(tools=tools, model="openai/gpt-5")

    assert set(sid_agent.tool_schemas) == set(openai_agent.search_tools)

    sid_arguments = {
        argument.name: argument.description
        for argument in sid_agent.tool_schemas["_annotated_search"].arguments
    }
    openai_properties = openai_agent.search_tools["_annotated_search"][1]["parameters"]["properties"]

    assert openai_properties["query"]["description"] == sid_arguments["query"]
    assert openai_properties["limit"]["description"] == sid_arguments["limit"]
    assert openai_properties["limit"]["minimum"] == 1
    assert openai_properties["limit"]["maximum"] == 10
