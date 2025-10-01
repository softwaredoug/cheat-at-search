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
        return BarFoo(x="no args", y=3.14)

    ArgsModel, tool_spec, call_from_tool = make_tool_adapter(do_foo)
    expected_return = do_foo()
    args = ArgsModel(a={"a": 10}, b=5)
    dict_result, _ = call_from_tool(args)
    py_result = BarFoo.model_validate(dict_result)
    assert py_result == expected_return
