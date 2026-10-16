import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cheat_at_search.agent.search_client import SearchResults
from cheat_at_search.agent.sid_agent import SIDAgent


def _tool_call(name, arguments, call_id):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _completion(*tool_calls, content=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=list(tool_calls),
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
    )


def hello_search(query: str, agent_state: dict) -> list[dict]:
    """Search the hello-world catalog."""
    assert query == "hello world"
    assert agent_state["user_id"] == "user-1"
    assert agent_state["trace_logger"] is not None
    return [
        {
            "id": "hello-1",
            "title": "Hello world",
            "description": "A friendly greeting.",
            "score": 0.95,
        }
    ]


def read(id: str) -> dict:
    """Read a document by its stable ID."""
    return {
        "id": id,
        "title": "Hello world",
        "description": "A friendly greeting.",
    }


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

    agent = SIDAgent(tools=[hello_search, read], max_turns=3)
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

    agent_state = {"user_id": "user-1"}
    active_logger = Mock()
    response, _, usage = agent.loop(
        inputs,
        agent_state=agent_state,
        return_usage=True,
        logger=active_logger,
    )
    results = response.output_parsed

    assert "report_helpful_ids" not in agent.tool_schemas
    assert [arg.name for arg in agent.tool_schemas["hello_search"].arguments] == [
        "query"
    ]
    assert agent_state["trace_logger"] is active_logger
    assert isinstance(response.output_parsed, SearchResults)
    assert results.ranked_results == ["hello-1"]
    assert usage == {
        "input_tokens": 20,
        "output_tokens": 4,
        "num_tool_calls": 2,
    }
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
    active_logger.info.assert_any_call(
        "System prompt: %s", first_request["messages"][0]["content"]
    )
    active_logger.info.assert_any_call(
        "User prompt: %s", "Find a hello world greeting."
    )
    assert "hello_search" in first_request["messages"][0]["content"]
    assert "<tool_call>" in first_request["messages"][0]["content"]
    assert (
        "Every assistant turn must contain one or more tool calls"
        in first_request["messages"][0]["content"]
    )
    assert '"name": "report_helpful_ids"' in first_request["messages"][0]["content"]

    second_messages = create.call_args_list[1].kwargs["messages"]
    assert second_messages[2]["role"] == "assistant"
    assert second_messages[2]["tool_calls"][0]["id"] == "call-search"
    assert second_messages[3] == {
        "role": "tool",
        "tool_call_id": "call-search",
        "content": (
            '<doc id="hello-1" title="Hello world" score="0.95">\n'
            "A friendly greeting.\n</doc>\n\n"
            "You have 2 out of 3 turns left."
        ),
    }


@patch("cheat_at_search.agent.sid_agent.OpenAI")
@patch("cheat_at_search.agent.sid_agent.key_for_provider", return_value="sid-test-key")
def test_sid_agent_logs_reasoning_tool_calls_and_outputs_in_order(
    mock_key_for_provider, mock_openai
):
    events = []

    def logged_search(query: str) -> list[dict]:
        """Search the test catalog."""
        events.append(("tool", "search", query))
        return [{"id": "doc-1", "title": "A result", "description": "Found it."}]

    def log_info(message, *args):
        events.append(("log", message % args if args else message))

    mock_client = mock_openai.return_value
    mock_client.chat.completions.create.side_effect = [
        _completion(
            _tool_call("logged_search", {"query": "hello"}, "call-search"),
            content="<think>Search for the greeting.</think>",
        ),
        _completion(
            _tool_call("report_helpful_ids", {"ids": ["doc-1"]}, "call-report"),
            content="<think>The result is relevant.</think>",
        ),
    ]
    active_logger = Mock()
    active_logger.info.side_effect = log_info
    agent = SIDAgent(tools=[logged_search, read], max_turns=2)

    agent.loop(
        [
            {"role": "system", "content": "Search and report useful IDs."},
            {"role": "user", "content": "Find a greeting."},
        ],
        logger=active_logger,
    )

    assert events[0][0] == "log"
    assert events[0][1].startswith(
        "System prompt: Search and report useful IDs."
    )
    assert "logged_search" in events[0][1]
    assert "<tool_call>" in events[0][1]
    assert events[1] == ("log", "User prompt: Find a greeting.")
    assert events[2:] == [
        ("log", "SID reasoning: <think>Search for the greeting.</think>"),
        ("log", 'Tool called: logged_search args={"query": "hello"}'),
        ("tool", "search", "hello"),
        (
            "log",
            'Tool response: <doc id="doc-1" title="A result">\nFound it.\n</doc>',
        ),
        ("log", "SID reasoning: <think>The result is relevant.</think>"),
        ("log", 'Tool called: report_helpful_ids args={"ids": ["doc-1"]}'),
        ("log", 'Tool response: ["doc-1"]'),
    ]
