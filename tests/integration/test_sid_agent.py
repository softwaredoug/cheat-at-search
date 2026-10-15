import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cheat_at_search.agent.search_client import SearchResults
from cheat_at_search.agent.sid_agent import SIDAgent


def _tool_call(name, arguments, call_id):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _completion(*tool_calls):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=None,
                    tool_calls=list(tool_calls),
                )
            )
        ]
    )


def hello_search(query: str) -> list[dict]:
    """Search the hello-world catalog."""
    assert query == "hello world"
    return [
        {
            "id": "hello-1",
            "title": "Hello world",
            "content": "A friendly greeting.",
        }
    ]


@pytest.mark.integration
@patch("cheat_at_search.agent.sid_agent.OpenAI")
@patch("cheat_at_search.agent.sid_agent.key_for_provider", return_value="sid-test-key")
def test_sid_agent_hello_world_tool_loop(mock_key_for_provider, mock_openai):
    mock_client = mock_openai.return_value
    mock_client.chat.completions.create.side_effect = [
        _completion(_tool_call("hello_search", {"query": "hello world"}, "call-search")),
        _completion(
            _tool_call(
                "report_helpful_ids",
                {"ids": ["hello-1"]},
                "call-report",
            )
        ),
    ]

    agent = SIDAgent(tools=[hello_search], max_turns=3)
    inputs = [
        {
            "role": "system",
            "content": (
                "Find relevant documents using the search tool. "
                "When done, call report_helpful_ids with `ids`, a list of "
                "document IDs ranked most helpful first."
            ),
        },
        {"role": "user", "content": "Find a hello world greeting."},
    ]

    results = agent.loop(inputs)

    assert "report_helpful_ids" not in agent.tool_schemas
    assert isinstance(results, SearchResults)
    assert [(result.id, result.rank) for result in results.results] == [("hello-1", 1)]
    mock_key_for_provider.assert_called_once_with("sid")
    mock_openai.assert_called_once_with(
        base_url="https://api.sid-1.com/v1",
        api_key="sid-test-key",
    )

    create = mock_client.chat.completions.create
    assert create.call_count == 2
    first_request = create.call_args_list[0].kwargs
    assert first_request["model"] == "sid-1"
    assert first_request["tools"] == [
        {"type": "function", "function": {"name": "dummy"}}
    ]
    assert "hello_search" in first_request["messages"][0]["content"]

    second_messages = create.call_args_list[1].kwargs["messages"]
    assert second_messages[2]["role"] == "assistant"
    assert second_messages[2]["tool_calls"][0]["id"] == "call-search"
    assert second_messages[3] == {
        "role": "tool",
        "tool_call_id": "call-search",
        "content": (
            '<doc id="hello-1" title="Hello world">\n'
            "A friendly greeting.\n</doc>\n\n"
            "You have 2 out of 3 turns left."
        ),
    }
