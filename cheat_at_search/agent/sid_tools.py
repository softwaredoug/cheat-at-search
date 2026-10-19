"""Tool schemas and cache-backed tools for SID-2."""

import json
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import ValidationError
from sid import DocumentCache
from sid.document_cache import SNIPPET_SIZE_DEFAULT

from cheat_at_search.agent.pydantize import make_tool_adapter
from cheat_at_search.agent.sid_builtins import (
    REPORT_TOOL,
    REPORT_TOOL_EXAMPLE,
    SNIPPET_FIELD,
    read,
)
from cheat_at_search.agent.sid_legacy_tool import (
    _example_argument_value,
    tool_schema as legacy_tool_schema,
)
from cheat_at_search.passage_fn import default_passage_fn


def _tool_names(tool: Callable[..., Any]) -> list[str]:
    """Return the provider-facing names declared on a callable.

    First see if the function has `aliases` (ie SID likes text_search / search)
    Second, if no SID alias is found (ie this isn't called `text_search` or something)
     then go ahead and just use the function name as the tool.
    """
    aliases = getattr(tool, "aliases", None)
    if aliases is None or aliases == [] or aliases == ():
        return [tool.__name__]
    if isinstance(aliases, str) or not isinstance(aliases, (list, tuple)):
        raise TypeError(f"{tool.__name__}.aliases must be a list or tuple of names.")
    if not all(isinstance(name, str) and name for name in aliases):
        raise ValueError(f"{tool.__name__}.aliases must contain non-empty strings.")
    if len(aliases) != len(set(aliases)):
        raise ValueError(f"{tool.__name__}.aliases must not contain duplicates.")
    return list(aliases)


def _tool_example(tool: Callable[..., Any], name: str) -> dict[str, Any]:
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
        "caption": f"Example call for {name}: {schema.docstring}",
        "name": name,
        "arguments": arguments,
    }


def _response_tool_schema(tool: Callable[..., Any], name: str) -> dict[str, Any]:
    """Build a Responses function schema, retaining SID legacy metadata."""
    schema = legacy_tool_schema(tool)
    _, response_schema, _ = make_tool_adapter(tool)

    # The legacy parser understands both string descriptions and example values
    # in Annotated metadata. Use its parsed descriptions for consistent output.
    response_schema["name"] = name
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

    Tool descriptions, parameter descriptions, types, and annotated examples
    are derived from each callable. A callable's non-empty ``aliases`` list
    supplies its SID-facing tool names; otherwise its ``__name__`` is used.
    SID-2's cache-backed ``read`` and fixed ``report_helpful_ids`` tools are
    appended automatically.

    Python's typing.Annotated metadata provides parameter descriptions and
    example values when provided, ie

    def search(query: Annotated[str, "Terms describing the greeting to find.", "delicious taco"]) -> list[dict]:
    """
    search_tools = list(search_tools)
    if not search_tools:
        raise ValueError("At least one SID-2 search tool is required.")

    if any(not callable(tool) for tool in search_tools):
        raise TypeError("SID-2 tools must be callable.")

    exposed_tools = [
        (name, tool)
        for tool in search_tools
        for name in _tool_names(tool)
    ]
    names = [name for name, _ in exposed_tools]
    reserved_names = {"read", "report_helpful_ids"}
    collisions = sorted(reserved_names.intersection(names))
    if collisions:
        raise ValueError("SID-2 reserves these tool names: " + ", ".join(collisions))
    if len(names) != len(set(names)):
        raise ValueError("SID-2 tool names must be unique.")

    exposed_tools.append(("read", read))
    tools = [_response_tool_schema(tool, name) for name, tool in exposed_tools]
    tools.append(REPORT_TOOL)

    examples = [_tool_example(tool, name) for name, tool in exposed_tools]
    examples.append(REPORT_TOOL_EXAMPLE)
    return tools, examples


def _query_from_arguments(arguments: dict[str, Any]) -> str:
    """Map between common query argument names to get the query."""
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
    return query


def _render_search_results(
    results: list[dict],
    arguments: dict[str, Any],
    cache: DocumentCache,
    passage_fn: Callable[[Any], str],
) -> str:
    """Cache full backend documents and render query-focused XML snippets.

    Its expected that search tools for agents return a list of dicts of flat
    attributes like title and description. (IE thats usually what would be sent
    back to OpenAI)

    SID expects a specific XML response format, ie

    <doc id="hello-world" title="France capital">
        France is a nation in Europe. Paris is the capital of France.
    </doc>

    It also expects a cachable snippet inside that XML, based on the
    relevance to the query, ie for the query "What is the capital of France?"
    the DocumentCache would represent this as:

    <doc id="hello-world" title="France capital">
        Paris is the capital of France.
    </doc>

    This requires us intercepting the query from the tool arguments
    and then using that to generate a snippet using SID's document
    cache.

    """
    if not results:
        return "No results found. Try a different query."

    query = _query_from_arguments(arguments)

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
        if not isinstance(data_id, str) or not data_id:
            raise ValueError(
                f"Search result must have a non-empty string 'id': {result!r}."
            )
        fields.pop("score", None)
        content = passage_fn(fields)
        if not isinstance(content, str) or not content:
            raise ValueError(
                f"passage_fn must return non-empty text for search result {data_id!r}."
            )
        fields.pop("description", None)
        fields.pop(SNIPPET_FIELD, None)
        fields[SNIPPET_FIELD] = content

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
    document_cache: DocumentCache,
    passage_fn: Callable[[Any], str] = default_passage_fn,
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
    if any(not callable(tool) for tool in functions):
        raise TypeError("SID-2 tools must be callable.")
    exposed_tools = [
        (name, tool)
        for tool in functions
        for name in _tool_names(tool)
    ]
    names = [name for name, _ in exposed_tools]
    if len(names) != len(set(names)):
        raise ValueError("SID-2 tool names must be unique.")
    if "read" in names:
        raise ValueError("SID-2 reserves the 'read' tool name.")
    exposed_tools.append(("read", read))
    function_by_name = {name: tool for name, tool in exposed_tools}
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
    tool_agent_state = agent_state
    if name == "read":
        tool_agent_state = {"_sid_document_cache": document_cache}
    try:
        result, json_result = call_from_tool(
            arguments, agent_state=tool_agent_state
        )
    except ValidationError as error:
        return f"Error executing {name}: invalid arguments: {error}"

    if isinstance(result, str):
        return result
    if name == "read":
        return json_result
    if isinstance(result, list) and all(isinstance(document, dict) for document in result):
        return _render_search_results(
            result, arguments, document_cache, passage_fn
        )
    return json_result
