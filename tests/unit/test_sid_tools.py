from types import SimpleNamespace
from typing import Annotated, Any, Dict, Literal, Sequence, cast
from unittest.mock import MagicMock, Mock

import pytest
from sid import DocumentCache

from cheat_at_search.agent.sid_tools import (
    REPORT_TOOL,
    build_sid2_tool_config,
    call_sid2_tool,
)


def filtered_search(
    query: Annotated[str, "What to find", "red hiking jacket"],
    limit: Annotated[int, "Maximum results", 5] = 5,
    mode: Annotated[Literal["exact", "broad"], "Search mode", "broad"] = "broad",
) -> list[dict]:
    """Search the product catalog."""
    return []


def basic_search(query: str, limit: int = 5) -> list[dict]:
    """Search documents with the default limit."""
    return []


def search_bm25(query: str) -> dict:
    """Run the backend BM25 search."""
    return {"backend_tool": "search_bm25", "query": query}


search_bm25.aliases = ["text_search", "bm25_search"]


def another_text_search(query: str) -> list[dict]:
    """Run another backend text search."""
    return []


another_text_search.aliases = ["text_search"]


def reserved_read_alias(query: str) -> list[dict]:
    """A search tool that conflicts with SID-2's built-in read tool."""
    return []


reserved_read_alias.aliases = ["read"]


def w_scenarios(scenarios: Dict[str, Dict[str, Any]]):
    """Decorate scenarios with readable pytest IDs, matching legacy tool tests."""
    return pytest.mark.parametrize(
        cast(Sequence[str], [key for key in scenarios.values()][0].keys()),
        [tuple(scenario.values()) for scenario in scenarios.values()],
        ids=list(scenarios.keys()),
    )


SCHEMA_SCENARIOS = {
    "annotated examples and literal filter": {
        "tool": filtered_search,
        "expected_name": "filtered_search",
        "expected_description": "Search the product catalog.",
        "expected_param_descriptions": {
            "query": "What to find",
            "limit": "Maximum results",
            "mode": "Search mode",
        },
        "expected_param_schema": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
            "mode": {"type": "string", "enum": ["exact", "broad"]},
        },
        "expected_example_arguments": {
            "query": "red hiking jacket",
            "limit": 5,
            "mode": "broad",
        },
    },
    "fallback examples": {
        "tool": basic_search,
        "expected_name": "basic_search",
        "expected_description": "Search documents with the default limit.",
        "expected_param_descriptions": {"query": "", "limit": ""},
        "expected_param_schema": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "expected_example_arguments": {"query": "example query", "limit": 5},
    },
}


@w_scenarios(SCHEMA_SCENARIOS)
def test_build_sid2_tool_config_uses_callable_metadata(
    tool,
    expected_name,
    expected_description,
    expected_param_descriptions,
    expected_param_schema,
    expected_example_arguments,
):
    tools, examples = build_sid2_tool_config([tool])

    search_schema = tools[0]
    assert search_schema["type"] == "function"
    assert search_schema["name"] == expected_name
    assert search_schema["description"] == expected_description
    assert search_schema["parameters"]["type"] == "object"
    properties = search_schema["parameters"]["properties"]
    assert {
        name: properties[name].get("description", "")
        for name in expected_param_descriptions
    } == expected_param_descriptions
    for name, expected_schema in expected_param_schema.items():
        assert {
            key: properties[name][key]
            for key in expected_schema
        } == expected_schema
    assert examples[0] == {
        "caption": f"Example call for {expected_name}: {expected_description}",
        "name": expected_name,
        "arguments": expected_example_arguments,
    }


ALIASES_SCENARIOS = {
    "backend name replaced by aliases": {
        "tool": search_bm25,
        "expected_names": [
            "text_search",
            "bm25_search",
            "read",
            "report_helpful_ids",
        ],
    },
    "no aliases uses function name": {
        "tool": basic_search,
        "expected_names": ["basic_search", "read", "report_helpful_ids"],
    },
}


@w_scenarios(ALIASES_SCENARIOS)
def test_build_sid2_tool_config_uses_function_aliases(tool, expected_names):
    tools, examples = build_sid2_tool_config([tool])

    assert [tool_spec["name"] for tool_spec in tools] == expected_names
    assert [example["name"] for example in examples] == expected_names
    if tool is search_bm25:
        assert all(
            tool_spec["description"] == "Run the backend BM25 search."
            for tool_spec in tools[:2]
        )


@w_scenarios(
    {
        "two tools share an alias": {
            "tools": [search_bm25, another_text_search],
            "error": "unique",
        },
        "alias collides with read": {
            "tools": [reserved_read_alias],
            "error": "reserves",
        },
    }
)
def test_build_sid2_tool_config_rejects_alias_collisions(tools, error):
    with pytest.raises(ValueError, match=error):
        build_sid2_tool_config(tools)


def test_build_sid2_tool_config_adds_read_and_fixed_report_tools():
    tools, examples = build_sid2_tool_config([filtered_search])

    assert [tool["name"] for tool in tools] == [
        "filtered_search",
        "read",
        "report_helpful_ids",
    ]
    assert tools[1]["parameters"]["properties"]["id"]["type"] == "string"
    assert tools[2]["parameters"]["properties"]["ids"]["items"]["type"] == "string"
    assert tools[2] == REPORT_TOOL
    assert [example["name"] for example in examples] == [
        "filtered_search",
        "read",
        "report_helpful_ids",
    ]


def reserved_tool(name):
    def tool(value: str) -> str:
        """A tool that collides with a built-in SID-2 tool."""
        return value

    tool.__name__ = name
    return tool


RESERVED_TOOL_SCENARIOS = {
    "read": {"tool": reserved_tool("read")},
    "report_helpful_ids": {"tool": reserved_tool("report_helpful_ids")},
}


@w_scenarios(RESERVED_TOOL_SCENARIOS)
def test_build_sid2_tool_config_rejects_reserved_tool_names(tool):
    with pytest.raises(ValueError, match="reserves"):
        build_sid2_tool_config([filtered_search, tool])


TOOL_CALL_SCENARIOS = {
    "malformed JSON": {
        "tool_name": "echo_tool",
        "arguments": "not-json",
        "expected": "invalid JSON arguments",
    },
    "non-object arguments": {
        "tool_name": "echo_tool",
        "arguments": "[]",
        "expected": "arguments must be a JSON object",
    },
    "invalid parameter type": {
        "tool_name": "echo_tool",
        "arguments": '{"query": 123}',
        "expected": "invalid arguments",
    },
    "unknown tool": {
        "tool_name": "missing_tool",
        "arguments": "{}",
        "expected": "not registered",
    },
}


def echo_tool(query: str, agent_state: dict) -> dict:
    """Echo a query with state to test tool dispatch."""
    return {"query": query, "tenant": agent_state["tenant"]}


@w_scenarios(TOOL_CALL_SCENARIOS)
def test_call_sid2_tool_handles_bad_or_unknown_calls(tool_name, arguments, expected):
    call = SimpleNamespace(name=tool_name, arguments=arguments)

    output = call_sid2_tool(
        call, [echo_tool], DocumentCache(), agent_state={"tenant": "test"}
    )

    assert expected in output


def test_call_sid2_tool_parses_arguments_and_injects_agent_state():
    call = SimpleNamespace(
        name="echo_tool",
        arguments='{"query": "hello"}',
    )

    output = call_sid2_tool(
        call,
        [echo_tool],
        DocumentCache(),
        agent_state={"tenant": "tenant-1"},
    )

    assert output == '{"query":"hello","tenant":"tenant-1"}'


def test_call_sid2_tool_dispatches_alias_to_original_callable():
    call = SimpleNamespace(
        name="text_search",
        arguments='{"query": "hello"}',
    )

    output = call_sid2_tool(call, [search_bm25], DocumentCache())

    assert output == '{"backend_tool":"search_bm25","query":"hello"}'


def test_call_sid2_tool_renders_search_documents_from_the_episode_cache():
    def search(query: str, snippet_size: int = 50) -> list[dict]:
        """Search the hello-world catalog."""
        return [
            {
                "id": "hello-1",
                "title": "Hello world",
                "content": "A friendly hello world greeting.",
            }
        ]

    call = SimpleNamespace(
        name="search",
        arguments='{"query": "hello world", "snippet_size": 12}',
    )

    mock_cache = MagicMock()
    mock_cache.__contains__.return_value = False
    view = mock_cache.apply_snippet.return_value
    view.render_xml.return_value = '<doc id="123456">hello world</doc>'

    output = call_sid2_tool(call, [search], mock_cache)

    assert output == '<doc id="123456">hello world</doc>'
    mock_cache.add_document.assert_called_once_with(
        "hello-1",
        {
            "title": "Hello world",
            "content": "A friendly hello world greeting.",
        },
    )
    mock_cache.apply_snippet.assert_called_once_with(
        "hello-1",
        snippet_field="content",
        query="hello world",
        snippet_size=12,
    )
    mock_cache.update_seen.assert_called_once_with(view)


def test_call_sid2_tool_reads_cached_document_by_ranged_id():
    mock_cache = Mock()
    mock_cache.contains_model_facing_id.return_value = True
    mock_cache.to_data_id.return_value = "backend-1"
    mock_cache.resolve_char_range.return_value = (5, 15)
    view = mock_cache.get_single_span_document_view.return_value
    view.render_xml.return_value = '<doc id="123456#5:15">hello world</doc>'
    call = SimpleNamespace(
        name="read",
        arguments='{"id": "123456#5:15"}',
    )

    output = call_sid2_tool(call, [filtered_search], mock_cache)

    assert output == '<doc id="123456#5:15">hello world</doc>'
    mock_cache.resolve_char_range.assert_called_once_with(
        "backend-1", "content", (5, 15)
    )
    mock_cache.get_single_span_document_view.assert_called_once_with(
        "backend-1",
        snippet_field="content",
        snippet_display_span=(5, 15),
    )
    mock_cache.update_seen.assert_called_once_with(view)


def test_call_sid2_tool_treats_report_as_a_terminal_call():
    call = SimpleNamespace(
        name="report_helpful_ids",
        arguments='{"ids": ["123456"]}',
    )

    with pytest.raises(ValueError, match="terminal"):
        call_sid2_tool(call, [echo_tool], DocumentCache())
