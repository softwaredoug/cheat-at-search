"""Test pydantizing any function for an agent to call it."""
from cheat_at_search.agent.pydantize import make_tool_adapter
from pydantic import BaseModel


def test_pydantize():

    class FooBar(BaseModel):
        a: int
        b: str = "default"

    class BarFoo(BaseModel):
        x: str
        y: float = 1.0

    def do_foo(a: FooBar, b: int) -> BarFoo:
        """Return a composed BarFoo for testing."""
        return BarFoo(x=f"{a.a}-{a.b}-{b}", y=b * 1.5)

    ArgsModel, tool_spec, call_from_tool = make_tool_adapter(do_foo)
    expected_return = do_foo(FooBar(a=10), 5)
    args = ArgsModel(a={"a": 10}, b=5)
    dict_result, _ = call_from_tool(args)
    py_result = BarFoo.model_validate(dict_result)
    assert py_result == expected_return


def test_pydantize_no_args():

    class BarFoo(BaseModel):
        x: str
        y: float = 1.0

    def do_foo() -> BarFoo:
        """Return a fixed BarFoo for testing."""
        return BarFoo(x="no args", y=3.14)

    ArgsModel, tool_spec, call_from_tool = make_tool_adapter(do_foo)
    expected_return = do_foo()
    args = ArgsModel(a={"a": 10}, b=5)
    dict_result, _ = call_from_tool(args)
    py_result = BarFoo.model_validate(dict_result)
    assert py_result == expected_return


def test_pydantize_agent_state():

    class BarFoo(BaseModel):
        x: str
        y: float = 1.0

    def do_foo(a: int, agent_state: dict) -> BarFoo:
        """Return a BarFoo that includes agent state."""
        return BarFoo(x=f"{a}-{agent_state['suffix']}", y=agent_state["scale"])

    ArgsModel, tool_spec, call_from_tool = make_tool_adapter(do_foo)
    assert "agent_state" not in ArgsModel.model_fields
    assert "agent_state" not in tool_spec["parameters"]["properties"]

    agent_state = {"suffix": "ok", "scale": 2.5}
    args = ArgsModel(a=7)
    dict_result, _ = call_from_tool(args, agent_state=agent_state)
    py_result = BarFoo.model_validate(dict_result)
    assert py_result == BarFoo(x="7-ok", y=2.5)


def test_pydantize_agent_state_keyword_only():

    class BarFoo(BaseModel):
        x: str
        y: float = 1.0

    def do_foo(a: int, *, agent_state: dict) -> BarFoo:
        """Return a BarFoo using keyword-only agent state."""
        return BarFoo(x=f"{a}-{agent_state['suffix']}", y=agent_state["scale"])

    ArgsModel, tool_spec, call_from_tool = make_tool_adapter(do_foo)
    assert "agent_state" not in ArgsModel.model_fields
    assert "agent_state" not in tool_spec["parameters"]["properties"]

    agent_state = {"suffix": "kw", "scale": 1.25}
    args = ArgsModel(a=3)
    dict_result, _ = call_from_tool(args, agent_state=agent_state)
    py_result = BarFoo.model_validate(dict_result)
    assert py_result == BarFoo(x="3-kw", y=1.25)


def test_pydantize_agent_state_annotation_ignored():

    class BarFoo(BaseModel):
        x: str
        y: float = 1.0

    def do_foo(a: int, agent_state: int) -> BarFoo:
        """Return a BarFoo even with a non-dict agent_state annotation."""
        return BarFoo(x=f"{a}-{agent_state}", y=0.0)

    ArgsModel, tool_spec, call_from_tool = make_tool_adapter(do_foo)
    assert "agent_state" not in ArgsModel.model_fields
    assert "agent_state" not in tool_spec["parameters"]["properties"]

    args = ArgsModel(a=9)
    dict_result, _ = call_from_tool(args, agent_state=5)
    py_result = BarFoo.model_validate(dict_result)
    assert py_result == BarFoo(x="9-5", y=0.0)
