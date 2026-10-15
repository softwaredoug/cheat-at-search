"""Describe SID tool for the SID agent."""
import inspect
from pydantic import BaseModel, PrivateAttr
from pydantic.fields import FieldInfo
from typing import Any, Callable, get_origin, get_args, Annotated, get_type_hints
from xml.sax.saxutils import escape, quoteattr
import json

from cheat_at_search.agent.pydantize import make_tool_adapter


class SIDArgument(BaseModel):
    name: str
    description: str
    type: Any      # really basic types and literal


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

        if get_origin(annotation) is Annotated:
            annotation, *metadata = get_args(annotation)
            description = next(
                (
                    item if isinstance(item, str) else item.description
                    for item in metadata
                    if isinstance(item, str)
                    or (isinstance(item, FieldInfo) and item.description)
                ),
                "",
            )
        arguments.append(
            SIDArgument(
                name=param.name,
                description=description,
                type=annotation,
            )
        )
    schema = SIDTool(name=name, docstring=docstring, arguments=arguments, function=tool)
    schema._call_from_tool = call_from_tool
    return schema


def format_tool_schema(tool: SIDTool) -> str:
    """Format a SIDTool schema for display."""
    args_per_line: list[str] = []
    for arg in tool.arguments:
        args_per_line.append(f"     - {arg.name}: {arg.description} (type: {arg.type})")
    return "\n".join(
        [f" - {tool.name}: {tool.docstring}",
         "   - Arguments:"] + args_per_line
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
