import json
from typing import Any, Dict, Literal, Sequence, cast, Annotated
import inspect

import pytest
from pydantic import Field

from cheat_at_search.agent.sid_legacy_tool import (
    _format_tool_result,
    call_tool,
    format_sid_tool_instructions,
    format_tool_schema,
    tool_schema,
)


def w_scenarios(scenarios: Dict[str, Dict[str, Any]]):
    """Decorate for parametrizing tests that names the scenarios and params."""
    return pytest.mark.parametrize(
        cast(Sequence[str], [key for key in scenarios.values()][0].keys()),
        [tuple(scenario.values()) for scenario in scenarios.values()],
        ids=list(scenarios.keys()),
    )


def search_tool(query: str) -> dict:
    """Return a hello-world greeting."""
    return {"query": query}


def complex_tool(query: Annotated[str, "The search query"],
                 count: Annotated[int, Field(..., description="The number of results to return")],
                 flag: bool) -> dict:
    """Return a complex response with multiple parameters."""
    return {"query": query, "count": count, "flag": flag}


def stateful_tool(query: str, agent_state: dict) -> dict:
    """Return the query with host-provided tenant state."""
    return {"query": query, "tenant": agent_state["tenant"]}


def fielded_search_tool(
    keywords: Annotated[str, "Search query keywords."],
    fields: Annotated[list[str], "Fields to search."],
    operator: Literal["and", "or", "phrase"] = "or",
    top_k: int = 5,
) -> list[dict]:
    """Search weighted fields using BM25."""
    return []


def explicit_example_tool(
    keywords: Annotated[str, "Search query keywords.", "portable red chair"],
    top_k: Annotated[int, "Maximum results.", 17],
    fields: Annotated[list[str], "Fields to search.", ["title^4.0"]],
) -> list[dict]:
    """Search with explicitly documented argument examples."""
    return []


SCENARIOS = {
    "hello-world tool": {
        "tool": search_tool,
        "expected_params": {"query": [str, ""]},
        "tool_call": {"query": "test"},
        "expected_tool_result": '{"query": "test"}',
        "agent_state": None,
    },
    "complex tool": {
        "tool": complex_tool,
        "expected_params": {
            "query": [str, "The search query"],
            "count": [int, "The number of results to return"],
            "flag": [bool, ""]
        },
        "tool_call": {"query": "test", "count": 5, "flag": True},
        "expected_tool_result": '{"query": "test", "count": 5, "flag": true}',
        "agent_state": None,
    },
    "stateful tool": {
        "tool": stateful_tool,
        "expected_params": {"query": [str, ""]},
        "tool_call": {
            "query": "test",
            "agent_state": {"tenant": "model-controlled"},
        },
        "expected_tool_result": '{"query": "test", "tenant": "trusted"}',
        "agent_state": {"tenant": "trusted"},
    },
}


@w_scenarios(SCENARIOS)
def test_tool_schema_hello_world(
    tool, expected_params, tool_call, expected_tool_result, agent_state
):
    expected_docstring = inspect.getdoc(tool) or ""
    expected_argument_names = list(expected_params)
    expected_name = tool.__name__
    schema = tool_schema(tool)

    assert schema.name == expected_name
    assert schema.docstring == expected_docstring
    assert [argument.name for argument in schema.arguments] == expected_argument_names
    assert {
        argument.name: [argument.type, argument.description]
        for argument in schema.arguments
    } == expected_params
    assert schema.function is tool

    call_result = call_tool(tool_call, schema, agent_state=agent_state)
    assert call_result == expected_tool_result
    formatted = format_tool_schema(schema)
    assert expected_docstring in formatted
    assert expected_name in formatted.splitlines()[0]


def test_format_tool_result_formats_documents_as_xml():
    result = [
        {
            "id": "hello-1",
            "title": "Hello & world",
            "description": "A <friendly> greeting.",
            "score": 0.9,
        }
    ]

    assert _format_tool_result(result) == (
        '<doc id="hello-1" title="Hello &amp; world" score="0.9">\n'
        "A &lt;friendly&gt; greeting.\n</doc>"
    )


def test_format_tool_result_explains_empty_search_results():
    assert _format_tool_result([]) == (
        "No documents matched this query. Try different keywords, synonyms, "
        "or a broader search."
    )


def test_format_tool_result_serializes_non_document_results_as_json():
    assert _format_tool_result({"ok": True}) == '{"ok": true}'


def test_agent_state_is_hidden_from_sid_and_injected_when_calling_tool():
    observed_state = {}

    def stateful_search(query: str, agent_state: int) -> dict:
        """Search using agent-owned state."""
        observed_state["value"] = agent_state
        return {"query": query, "state": agent_state}

    schema = tool_schema(stateful_search)

    assert [argument.name for argument in schema.arguments] == ["query"]
    result = call_tool(
        {"query": "blue lamp", "agent_state": 999},
        schema,
        agent_state=42,
    )

    assert observed_state["value"] == 42
    assert json.loads(result) == {"query": "blue lamp", "state": 42}


@pytest.mark.parametrize(
    "argument_name",
    ["keywords", "query", "question", "product_description"],
)
def test_format_tool_schema_uses_example_query_for_query_arguments(argument_name):
    def search(**kwargs):
        """Search for matching documents."""
        return []

    search.__signature__ = inspect.Signature(
        [
            inspect.Parameter(
                argument_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=str,
            )
        ]
    )
    search.__annotations__ = {argument_name: str, "return": list}

    formatted = format_tool_schema(tool_schema(search))

    assert '"%s": "example query"' % argument_name in formatted
    assert "<tool_call>" in formatted


def test_format_tool_schema_generates_typed_examples_and_literal_choice():
    formatted = format_tool_schema(tool_schema(fielded_search_tool))
    example_call = formatted.split("<tool_call>\n", 1)[1].split("\n</tool_call>", 1)[0]
    arguments = json.loads(example_call)["arguments"]

    assert arguments["keywords"] == "example query"
    assert arguments["fields"] == ["title^9.3", "description^4.1"]
    assert arguments["top_k"] == 5
    assert arguments["operator"] in {
        "and",
        "or",
        "phrase",
    }


def test_format_tool_schema_prefers_explicit_annotated_examples():
    schema = tool_schema(explicit_example_tool)
    formatted = format_tool_schema(schema)
    example_call = formatted.split("<tool_call>\n", 1)[1].split("\n</tool_call>", 1)[0]
    arguments = json.loads(example_call)["arguments"]

    assert schema.arguments[0].description == "Search query keywords."
    assert schema.arguments[0].example == "portable red chair"
    assert arguments == {
        "keywords": "portable red chair",
        "top_k": 17,
        "fields": ["title^4.0"],
    }


def test_format_sid_tool_instructions_explains_call_protocol_and_final_ids():
    formatted = format_sid_tool_instructions()
    report_call = formatted.rsplit("<tool_call>\n", 1)[1].split("\n</tool_call>", 1)[0]

    assert "Every assistant turn must contain one or more tool calls" in formatted
    assert json.loads(report_call) == {
        "name": "report_helpful_ids",
        "arguments": {"ids": ["document_id_from_search"]},
    }
