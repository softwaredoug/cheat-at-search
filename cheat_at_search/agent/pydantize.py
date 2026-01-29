"""Take any function, and create a pydantic model wrapper for easy agentic use."""
from pydantic import BaseModel, create_model
from typing import Any, get_type_hints
from pydantic.type_adapter import TypeAdapter
import inspect


def make_tool_adapter(
    func,
    agent_state_param: str = "agent_state",
):
    """
    Returns (ArgsModel, tool_spec, call_from_tool)
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    fields: dict[str, tuple[type, Any]] = {}
    ordered_params: list[tuple[str, inspect.Parameter, bool]] = []

    for pname, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        is_agent_state = pname == agent_state_param
        if is_agent_state:
            ordered_params.append((pname, p, True))
            continue
        ann = hints.get(pname, Any)
        default = ... if p.default is inspect._empty else p.default
        fields[pname] = (ann, default)
        ordered_params.append((pname, p, False))

    ArgsModel: type[BaseModel] = create_model(f"{func.__name__.capitalize()}Args", **fields)  # type: ignore
    if not func.__doc__:
        raise ValueError(f"Function {func.__name__} must have a docstring for tool description.")
    tool_spec = {
        "type": "function",
        "name": func.__name__,
        "description": (func.__doc__ or f"Call {func.__name__}").strip(),
        "parameters": ArgsModel.model_json_schema(),
    }

    ret_ann = hints.get("return", Any)
    ret_adapter = TypeAdapter(ret_ann)   # works for BaseModel, containers, unions, tuples, etc.

    def call_from_tool(d: dict, agent_state: dict | None = None):
        m = ArgsModel.model_validate(d)

        posargs = []
        kwargs = {}
        for name, p, is_agent_state in ordered_params:
            if is_agent_state:
                val = agent_state
                if val is None:
                    val = {}
                if p.kind == inspect.Parameter.POSITIONAL_ONLY:
                    posargs.append(val)
                else:
                    kwargs[name] = val
                continue
            val = getattr(m, name)
            if p.kind == inspect.Parameter.POSITIONAL_ONLY:
                posargs.append(val)
            else:
                kwargs[name] = val

        result = func(*posargs, **kwargs)

        py_result = ret_adapter.dump_python(result)
        json_text = ret_adapter.dump_json(result).decode()
        return py_result, json_text

    return ArgsModel, tool_spec, call_from_tool


class FooBar(BaseModel):
    a: int
    b: str = "default"


class BarFoo(BaseModel):
    x: str
    y: float = 1.0


def do_foo(a: FooBar, b: int) -> BarFoo:
    return f"{a.a}-{a.b}-{b}"


if __name__ == "__main__":
    ArgsModel, tool_spec, call_from_tool = make_tool_adapter(do_foo)
    print(ArgsModel.model_json_schema(mode='serialization'))
    args = ArgsModel(a={"a": 10}, b=5)
    result = call_from_tool(args)
    print(result)
