"""Tool schemas and cache-backed tools for SID-2."""

import json
from collections.abc import Callable, Iterable
from typing import Annotated, Any

from pydantic import ValidationError
from sid import DocumentCache, InvalidCharacterRange, parse_rendered_model_facing_id
from sid.document_cache import SNIPPET_SIZE_DEFAULT

from cheat_at_search.agent.pydantize import make_tool_adapter
from cheat_at_search.agent.sid_legacy_tool import (
    _example_argument_value,
    tool_schema as legacy_tool_schema,
)


SNIPPET_FIELD = "content"


cache = DocumentCache(range_mode="lenient")


def read(
    id: Annotated[
        str,
        "A model-facing document ID or ranged reference, such as 123456#800:1600.",
        "document_id_from_search",
    ],
) -> str:
    """Read up to 10,000 characters from a document returned by search."""

    try:
        model_id, char_range = parse_rendered_model_facing_id(id)

        if not cache.contains_model_facing_id(model_id):
            return "Unknown document ID. Use search or text_search first."

        data_id = cache.to_data_id(model_id)

        if char_range is not None:
            char_range = cache.resolve_char_range(
                data_id, SNIPPET_FIELD, char_range
            )
    except (ValueError, InvalidCharacterRange) as error:
        return f"Error executing read: {error}"

    if char_range is not None:
        start, end = char_range
        char_range = (start, min(end, start + 10_000))
    elif len(cache.get_document(data_id)[SNIPPET_FIELD]) > 10_000:
        char_range = (0, 10_000)

    view = cache.get_single_span_document_view(
        data_id,
        snippet_field=SNIPPET_FIELD,
        snippet_display_span=char_range,
    )
    output = view.render_xml()
    cache.update_seen(view)
    return output


REPORT_TOOL = {
    "type": "function",
    "name": "report_helpful_ids",
    "description": (
        "Report the most helpful document ids from the search results. This is required in "
        "the last step. Each id may be reported at most once; repeated ids are penalized. "
        "Ids may be whole documents ('123456') or ranged references ('123456#800:1600'); a "
        "ranged reference counts as its whole document, so there is no need to strip the "
        "'#start:end' part."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "description": "The ids of the most helpful documents from the search results.",
                "items": {"type": "string"},
            },
        },
        "required": ["ids"],
    },
}

REPORT_TOOL_EXAMPLE = {
    "caption": "Report the most helpful document IDs, ranked best first:",
    "name": "report_helpful_ids",
    "arguments": {"ids": ["document_id_from_search"]},
}


def resolve_reported_ids(references: list[str]) -> list[str]:
    """Map valid, unique model-facing report IDs back to backend IDs."""
    resolved = []
    seen_model_ids = set()
    for reference in references:
        try:
            model_id, _ = parse_rendered_model_facing_id(reference)
        except (TypeError, ValueError):
            continue
        if model_id in seen_model_ids or not cache.contains_model_facing_id(model_id):
            continue
        seen_model_ids.add(model_id)
        resolved.append(cache.to_data_id(model_id))
    return resolved


def _tool_example(tool: Callable[..., Any]) -> dict[str, Any]:
    """Build one SID-2 few-shot tool-call example from a callable's annotations."""
    schema = legacy_tool_schema(tool)
    arguments = {
        argument.name: (
            argument.example
            if argument.has_example
            else _example_argument_value(argument.name, argument.type)
        )
        for argument in schema.arguments
    }
    return {
        "caption": f"Example call for {schema.name}: {schema.docstring}",
        "name": schema.name,
        "arguments": arguments,
    }


def _response_tool_schema(tool: Callable[..., Any]) -> dict[str, Any]:
    """Build a Responses function schema, retaining SID legacy metadata."""
    schema = legacy_tool_schema(tool)
    _, response_schema, _ = make_tool_adapter(tool)

    # The legacy parser understands both string descriptions and example values
    # in Annotated metadata. Use its parsed descriptions for consistent output.
    response_schema["name"] = schema.name
    response_schema["description"] = schema.docstring
    properties = response_schema["parameters"].get("properties", {})
    for argument in schema.arguments:
        if argument.description and argument.name in properties:
            properties[argument.name]["description"] = argument.description
    return response_schema


def build_sid2_tool_config(
    search_tools: Iterable[Callable[..., Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build SID-2's API ``tools`` and ``tool_examples`` from search callables.

    Tool names, descriptions, parameter descriptions, types, and annotated
    examples are derived from each callable. SID-2's cache-backed ``read`` and
    fixed ``report_helpful_ids`` tools are appended automatically.
    """
    search_tools = list(search_tools)
    if not search_tools:
        raise ValueError("At least one SID-2 search tool is required.")

    names = [getattr(tool, "__name__", None) for tool in search_tools]
    if any(not callable(tool) for tool in search_tools):
        raise TypeError("SID-2 tools must be callable.")
    reserved_names = {"read", "report_helpful_ids"}
    collisions = sorted(reserved_names.intersection(names))
    if collisions:
        raise ValueError("SID-2 reserves these tool names: " + ", ".join(collisions))
    if len(names) != len(set(names)):
        raise ValueError("SID-2 tool names must be unique.")

    functions = [*search_tools, read]
    tools = [_response_tool_schema(tool) for tool in functions]
    tools.append(REPORT_TOOL)

    examples = [_tool_example(tool) for tool in functions]
    examples.append(REPORT_TOOL_EXAMPLE)
    return tools, examples


def _render_search_results(results: list[dict], arguments: dict[str, Any]) -> str:
    """Cache full backend documents and render query-focused XML snippets."""
    if not results:
        return "No results found. Try a different query."

    query_names = {
        "query",
        "keywords",
        "question",
        "search_query",
        "product_description",
    }
    query = next(
        (
            value
            for name, value in arguments.items()
            if isinstance(value, str)
            and (
                name.lower() in query_names
                or name.lower().endswith(("_query", "_keywords"))
            )
        ),
        "",
    )
    snippet_size = arguments.get("snippet_size", SNIPPET_SIZE_DEFAULT)
    if (
        not isinstance(snippet_size, int)
        or isinstance(snippet_size, bool)
        or snippet_size <= 0
    ):
        return "Error executing search: snippet_size must be a positive integer."

    rendered = []
    for result in results:
        if not isinstance(result, dict):
            raise ValueError(
                f"Search results must be document objects, got {result!r}."
            )
        fields = dict(result)
        data_id = fields.pop("id", None)
        content = fields.get(SNIPPET_FIELD)
        if not isinstance(data_id, str) or not data_id:
            raise ValueError(
                f"Search result must have a non-empty string 'id': {result!r}."
            )
        if not isinstance(content, str) or not content:
            raise ValueError(
                f"Search result {data_id!r} must have non-empty {SNIPPET_FIELD!r} text."
            )

        if data_id not in cache:
            cache.add_document(data_id, fields)
        view = cache.apply_snippet(
            data_id,
            snippet_field=SNIPPET_FIELD,
            query=query,
            snippet_size=snippet_size,
        )
        rendered.append(view.render_xml())
        cache.update_seen(view)

    return "\n".join(rendered)


def call_sid2_tool(
    tool_call: Any,
    tools: Iterable[Callable[..., Any]],
    agent_state: dict | None = None,
) -> str:
    """Parse a Responses function call, validate its parameters, and execute it.

    Search results are registered in the SID episode cache and returned as
    XML snippets. ``report_helpful_ids`` is terminal and must be handled by the
    caller rather than executed as an ordinary tool.
    """
    if isinstance(tool_call, dict):
        name = tool_call.get("name")
        raw_arguments = tool_call.get("arguments")
    else:
        name = getattr(tool_call, "name", None)
        raw_arguments = getattr(tool_call, "arguments", None)

    if not isinstance(name, str) or not name:
        return "Error executing tool: function call is missing a tool name."
    if name == "report_helpful_ids":
        raise ValueError("report_helpful_ids is a terminal tool call and is not executed.")

    functions = list(tools)
    names = [getattr(tool, "__name__", None) for tool in functions]
    if any(not callable(tool) for tool in functions):
        raise TypeError("SID-2 tools must be callable.")
    if len(names) != len(set(names)):
        raise ValueError("SID-2 tool names must be unique.")
    if "read" in names:
        raise ValueError("SID-2 reserves the 'read' tool name.")
    functions.append(read)
    function_by_name = {tool.__name__: tool for tool in functions}
    tool = function_by_name.get(name)
    if tool is None:
        return f"Error executing {name}: tool is not registered."

    try:
        arguments = (
            json.loads(raw_arguments)
            if isinstance(raw_arguments, str)
            else raw_arguments
        )
    except json.JSONDecodeError as error:
        return f"Error executing {name}: invalid JSON arguments: {error}"
    if not isinstance(arguments, dict):
        return f"Error executing {name}: arguments must be a JSON object."

    if agent_state is None:
        agent_state = {}
    _, _, call_from_tool = make_tool_adapter(tool)
    try:
        result, json_result = call_from_tool(arguments, agent_state=agent_state)
    except ValidationError as error:
        return f"Error executing {name}: invalid arguments: {error}"

    if isinstance(result, str):
        return result
    if name == "read":
        return json_result
    if isinstance(result, list) and all(isinstance(document, dict) for document in result):
        return _render_search_results(result, arguments)
    return json_result
