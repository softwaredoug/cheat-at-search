"""Describe SID tool for the SID agent."""
import inspect
import random
from pydantic import BaseModel, PrivateAttr
from pydantic.fields import FieldInfo
from typing import Any, Callable, Literal, Union, get_origin, get_args, Annotated, get_type_hints
from types import UnionType
from xml.sax.saxutils import escape, quoteattr
import json

from cheat_at_search.agent.pydantize import make_tool_adapter


class SIDArgument(BaseModel):
    name: str
    description: str
    type: Any      # really basic types and literal
    example: Any = None
    has_example: bool = False


class SIDTool(BaseModel):
    """Callable SID model."""
    name: str
    docstring: str
    arguments: list[SIDArgument]

    function: Callable[..., Any]
    _call_from_tool: Callable[..., Any] | None = PrivateAttr(default=None)

    @property
    def short_description(self):
        return {"type": "function", "function": {"name": self.name}}


def _format_tool_result(result) -> str:
    """Format document lists as SID XML; serialize other tool results as JSON."""
    if isinstance(result, list) and all(isinstance(document, dict) for document in result):
        if not result:
            return "No documents matched this query. Try different keywords, synonyms, or a broader search."

        formatted_documents = []
        for document in result:
            attributes = ["id", "title"] + [
                key for key in document if key not in {"id", "title", "description"}
            ]
            attributes_text = " ".join(
                f"{name}={quoteattr(str(document[name]))}"
                for name in attributes
                if name in document and document[name] is not None
            )
            description = escape(str(document["description"]))
            formatted_documents.append(
                f"<doc {attributes_text}>\n{description}\n</doc>"
            )
        return "\n".join(formatted_documents)
    return json.dumps(result)


def tool_schema(tool: Callable) -> SIDTool:
    """Generate a SIDTool schema from a callable."""
    name = tool.__name__
    docstring = inspect.getdoc(tool) or ""
    sig = inspect.signature(tool)
    _, _, call_from_tool = make_tool_adapter(tool)

    hints = get_type_hints(tool, include_extras=True)
    arguments: list[SIDArgument] = []
    for param in sig.parameters.values():
        if param.name == "agent_state":
            continue
        annotation = hints.get(param.name, param.annotation)
        description = ""
        example = None
        has_example = False

        if get_origin(annotation) is Annotated:
            annotation, *metadata = get_args(annotation)
            for item in metadata:
                if isinstance(item, FieldInfo):
                    if item.description and not description:
                        description = item.description
                    if item.examples and not has_example:
                        example = item.examples[0]
                        has_example = True
                elif isinstance(item, str) and not description:
                    description = item
                elif not has_example:
                    example = item
                    has_example = True
        arguments.append(
            SIDArgument(
                name=param.name,
                description=description,
                type=annotation,
                example=example,
                has_example=has_example,
            )
        )
    schema = SIDTool(name=name, docstring=docstring, arguments=arguments, function=tool)
    schema._call_from_tool = call_from_tool
    return schema


def format_tool_schema(tool: SIDTool) -> str:
    """Format a SIDTool schema and a representative SID tool call."""
    args_per_line: list[str] = []
    example_arguments: dict[str, Any] = {}
    for arg in tool.arguments:
        args_per_line.append(f"     - {arg.name}: {arg.description} (type: {arg.type})")
        example_arguments[arg.name] = (
            arg.example
            if arg.has_example
            else _example_argument_value(arg.name, arg.type)
        )
    example = {
        "name": tool.name,
        "arguments": example_arguments,
    }
    return "\n".join(
        [f" - {tool.name}: {tool.docstring}",
         "   - Arguments:"]
        + args_per_line
        + [
            "   - Example call:",
            "<tool_call>",
            json.dumps(example, indent=2, ensure_ascii=False),
            "</tool_call>",
        ]
    )


def _example_argument_value(name: str, annotation: Any) -> Any:
    """Return a representative JSON value for a common tool argument."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Annotated:
        return _example_argument_value(name, args[0])
    if origin is Literal:
        return random.choice(args)
    if origin in {Union, UnionType}:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if not non_none_args:
            return None
        return _example_argument_value(name, non_none_args[0])

    normalized_name = name.lower()
    if annotation is str:
        if (
            normalized_name in {"query", "keywords", "question", "product_description"}
            or normalized_name.endswith(("_query", "_keywords"))
        ):
            return "example query"
        if normalized_name in {"id", "doc_id", "document_id"} or normalized_name.endswith("_id"):
            return "document_id_from_search"
        if normalized_name in {"category", "product_category"}:
            return "Furniture"
        if normalized_name == "path":
            return "."
        if normalized_name == "glob":
            return "**/*.txt"
        if normalized_name == "pattern":
            return "example"
        if normalized_name == "command":
            return "pwd"
        return "example"
    if annotation is bool:
        return True
    if annotation is int:
        if normalized_name == "timeout":
            return 30
        if normalized_name == "max_alternatives":
            return 3
        if normalized_name in {
            "top_k",
            "limit",
            "max_results",
            "num_results",
            "count",
            "n",
        }:
            return 5
        return 1
    if annotation is float:
        if normalized_name == "b":
            return 0.75
        if normalized_name == "k1":
            return 1.2
        return 1.0
    if origin is list:
        if normalized_name in {"fields"}:
            return ["title^9.3", "description^4.1"]
        if normalized_name in {"product_categories", "categories"}:
            return ["Furniture"]
        if normalized_name == "feature_names":
            return ["color"]
        if normalized_name in {"ids", "doc_ids", "document_ids"}:
            return ["document_id_from_search"]
        item_type = args[0] if args else str
        return [_example_argument_value(name, item_type)]
    if origin is dict:
        return {}
    if annotation is Any or annotation is inspect.Signature.empty:
        return "example"
    return "example"


def format_sid_tool_instructions() -> str:
    """Format SID's tool-call protocol and built-in final reporting tool."""
    report_call = {
        "name": "report_helpful_ids",
        "arguments": {"ids": ["document_id_from_search"]},
    }
    return "\n".join(
        [
            "Every assistant turn must contain one or more tool calls.",
            "Use this exact format; do not write tool calls as Python code or markdown code blocks:",
            "<tool_call>",
            '{"name": "tool_name", "arguments": {"argument_name": "value"}}',
            "</tool_call>",
            "When finished, call report_helpful_ids with document IDs returned by the search tools,",
            "ranked most helpful first. This is the final tool call:",
            "<tool_call>",
            json.dumps(report_call, indent=2),
            "</tool_call>",
        ]
    )


def call_tool(
    args: dict[str, Any],
    tool: SIDTool,
    agent_state: dict | None = None,
) -> Any:
    """Call a SIDTool with the provided arguments."""
    if tool._call_from_tool is None:
        raise RuntimeError(f"SID tool {tool.name} has no argument adapter.")
    result, _ = tool._call_from_tool(args, agent_state=agent_state)
    if isinstance(result, str) and result.startswith("Tool error:"):
        return result
    return _format_tool_result(result)
