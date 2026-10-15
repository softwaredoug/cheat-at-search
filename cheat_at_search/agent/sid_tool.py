"""Describe SID tool for the SID agent."""
import inspect
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from typing import Any, Callable, get_origin, Literal, get_args, Annotated, get_type_hints
from xml.sax.saxutils import escape, quoteattr
import json


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


def _is_valid_sid_type(annotation: Any) -> bool:
    """Check if the annotation is a valid SID type."""
    # For simplicity, we consider basic types and literals as valid SID types.
    # Or lists of these things
    valid_types = (int, float, str, bool)
    if annotation in valid_types or get_origin(annotation) is Literal:
        return True
    if hasattr(annotation, "__origin__") and annotation.__origin__ is list:
        return _is_valid_sid_type(annotation.__args__[0])
    return False


def tool_schema(tool: Callable) -> SIDTool:
    """Generate a SIDTool schema from a callable."""
    name = tool.__name__
    docstring = inspect.getdoc(tool) or ""
    sig = inspect.signature(tool)

    hints = get_type_hints(tool, include_extras=True)
    arguments: list[SIDArgument] = []
    for param in sig.parameters.values():
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
    return SIDTool(name=name, docstring=docstring, arguments=arguments, function=tool)


def format_tool_schema(tool: SIDTool) -> str:
    """Format a SIDTool schema for display."""
    args_per_line: list[str] = []
    for arg in tool.arguments:
        args_per_line.append(f"     - {arg.name}: {arg.description} (type: {arg.type})")
    return "\n".join(
        [f" - {tool.name}: {tool.docstring}",
         "   - Arguments:"] + args_per_line
    )


def call_tool(args: dict[str, Any], tool: SIDTool) -> Any:
    """Call a SIDTool with the provided arguments."""
    # Validate arguments
    for arg in tool.arguments:
        if arg.name not in args:
            raise ValueError(f"Missing argument: {arg.name}")
        if not _is_valid_sid_type(type(args[arg.name])):
            raise TypeError(f"Invalid type for argument {arg.name}: {type(args[arg.name])}")

    # Call the function
    resp = tool.function(**args)
    return _format_tool_result(resp)
