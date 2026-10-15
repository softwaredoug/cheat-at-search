from typing import Any, Dict, Sequence, cast, Annotated
import inspect

import pytest
from pydantic import Field

from cheat_at_search.agent.sid_tool import tool_schema, call_tool, format_tool_schema


def w_scenarios(scenarios: Dict[str, Dict[str, Any]]):
    """Decorate for parametrizing tests that names the scenarios and params."""
    return pytest.mark.parametrize(
        cast(Sequence[str], [key for key in scenarios.values()][0].keys()),
        [tuple(scenario.values()) for scenario in scenarios.values()],
        ids=list(scenarios.keys()),
    )


def search_tool(query: str) -> str:
    """Return a hello-world greeting."""
    return {"query": query}


def complex_tool(query: Annotated[str, "The search query"],
                 count: Annotated[int, Field(..., description="The number of results to return")],
                 flag: bool) -> dict:
    """Return a complex response with multiple parameters."""
    return {"query": query, "count": count, "flag": flag}


SCENARIOS = {
    "hello-world tool": {
        "tool": search_tool,
        "expected_params": {"query": [str, ""]},
        "tool_call": {"query": "test"}
    },
    "complex tool": {
        "tool": complex_tool,
        "expected_params": {
            "query": [str, "The search query"],
            "count": [int, "The number of results to return"],
            "flag": [bool, ""]
        },
        "tool_call": {"query": "test", "count": 5, "flag": True},
    }
}


@w_scenarios(SCENARIOS)
def test_tool_schema_hello_world(
    tool, expected_params, tool_call
):
    expected_docstring = inspect.getdoc(tool) or ""
    expected_argument_names = [param.name for param in inspect.signature(tool).parameters.values()]
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

    call_result = call_tool(tool_call, schema)
    assert call_result == tool_call
    formatted = format_tool_schema(schema)
    assert expected_docstring in formatted
    assert expected_name in formatted.splitlines()[0]
