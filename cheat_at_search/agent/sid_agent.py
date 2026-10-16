from cheat_at_search.agent.search_client import Agent, SearchResults
from cheat_at_search.data_dir import key_for_provider
from cheat_at_search.agent.sid_tool import (
    call_tool,
    format_sid_tool_instructions,
    format_tool_schema,
    tool_schema,
)
from cheat_at_search.logger import log_to_stdout
from openai import OpenAI
from pydantic import BaseModel
from hashlib import md5
import json
from time import sleep


TOOL_HEADER = "## Available Search Tools"
SID_STOP_FUNCTION = "report_helpful_ids"
logger = log_to_stdout("sid_search_client")


class SIDChatResponse(BaseModel):
    output_parsed: SearchResults


def _ids_to_search_results(ids: list[str]) -> SearchResults:
    """Convert a list of IDs to SearchResults."""
    return SearchResults(ranked_results=ids)


def _assistant_message_to_dict(message) -> dict:
    """Serialize an SDK assistant message for the next Chat Completions turn."""
    tool_calls = [
        {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments,
            },
        }
        for call in (message.tool_calls or [])
    ]
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": tool_calls,
    }


def _call_sid(client, model, inputs, tools, num_attempts=6):

    last_exception = None
    for attempt in range(num_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=inputs,
                tools=tools,
            )
            return response
        except Exception as e:
            logger.warning(
                "SIDAI call failed on attempt %d/%d: %s",
                attempt + 1,
                num_attempts,
                e,
            )
            last_exception = e
            sleep(3 ** attempt)
    raise last_exception or RuntimeError("SIDAI call failed after retries.")


class SIDAgent(Agent):
    def __init__(
        self,
        tools,
        max_turns: int,
        model="sid-1"
    ):
        self.model = model
        self.api_key = key_for_provider("sid")
        self.client = OpenAI(
            base_url="https://api.sid-1.com/v1",
            api_key=self.api_key,
        )
        self.max_turns = max_turns
        self.last_usage = None
        self.tool_schemas = {}
        names = set()
        for tool in tools:
            schema = tool_schema(tool)
            self.tool_schemas[schema.name] = schema
            names.add(schema.name)

        if 'read' not in names:
            raise ValueError("A 'read' tool that provides a result by ID must be provided to the SIDAgent.")

        if len(tools) == 0:
            raise ValueError("At least one tool must be provided to the SIDAgent.")

    def config_hash(self) -> str:
        """Return a hash representing the configuration of the agent."""
        tool_specs = [
            {
                "name": schema.name,
                "description": schema.docstring,
                "arguments": [
                    {
                        "name": argument.name,
                        "description": argument.description,
                        "type": str(argument.type),
                        "has_example": argument.has_example,
                        "example": argument.example,
                    }
                    for argument in schema.arguments
                ],
            }
            for schema in self.tool_schemas.values()
        ]
        tool_specs.sort(key=lambda spec: spec["name"])

        payload = {
            "provider": "sid",
            "model": self.model,
            "max_turns": self.max_turns,
            "tools": tool_specs,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return md5(serialized.encode("utf-8")).hexdigest()

    def _turn_budget_msg(self, turn_no: int) -> str:
        turns_left = self.max_turns - (turn_no + 1)
        if turns_left >= 1:
            notice = f"\n\nYou have {turns_left} out of {self.max_turns} turns left."
            if turns_left == 1:
                notice += " You must call `report_helpful_ids` in the next turn."
            return notice
        return ""

    def loop(
        self,
        inputs: list[dict] | None = None,
        agent_state: dict | None = None,
        return_usage: bool = False,
        logger=None,
    ):
        """Run the SID tool loop and return the response, messages, and usage."""
        active_logger = logger or globals()["logger"]
        if agent_state is not None:
            agent_state["trace_logger"] = active_logger
        if inputs is None:
            inputs = []

        # Check for system prompt in inputs
        if len(inputs) < 2:
            raise ValueError("At least two inputs are required: a system prompt and a user prompt (search query).")

        if inputs[0]["role"] != "system":
            raise ValueError("First input must be a system prompt.")

        system_prompt = inputs[0]["content"]

        if TOOL_HEADER not in system_prompt:
            tools_formatted = [format_tool_schema(tool) for tool in self.tool_schemas.values()]
            system_prompt += (
                "\n\n"
                + TOOL_HEADER
                + ":\n"
                + "\n".join(tools_formatted)
                + "\n\n"
                + format_sid_tool_instructions()
            )
            inputs[0]["content"] = system_prompt

        if inputs[1]["role"] != "user":
            raise ValueError("Second input must be a user prompt.")

        tools = [{"type": "function", "function": {"name": "dummy"}}]
        usage = {"input_tokens": 0, "output_tokens": 0, "num_tool_calls": 0}

        for turn in range(self.max_turns):
            response = _call_sid(self.client, self.model, inputs, tools)
            response_usage = getattr(response, "usage", None)
            usage["input_tokens"] += (
                getattr(response_usage, "prompt_tokens", None)
                or getattr(response_usage, "input_tokens", 0)
                or 0
            )
            usage["output_tokens"] += (
                getattr(response_usage, "completion_tokens", None)
                or getattr(response_usage, "output_tokens", 0)
                or 0
            )
            assistant_message = response.choices[0].message
            if assistant_message.content:
                active_logger.info("SID reasoning: %s", assistant_message.content)
            tool_calls = assistant_message.tool_calls or []
            inputs.append(_assistant_message_to_dict(assistant_message))

            if not tool_calls:
                # No tool calls seems to indicate some kind of error
                # per Sid
                raise ValueError(f"No tool calls returned from SID. Assistant message: {assistant_message}")

            for tool_call in tool_calls:
                usage["num_tool_calls"] += 1
                name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                call_id = tool_call.id
                active_logger.info(
                    "Tool called: %s args=%s",
                    name,
                    json.dumps(tool_args, sort_keys=True),
                )
                if name == SID_STOP_FUNCTION:
                    result = _ids_to_search_results(tool_args["ids"])
                    active_logger.info(
                        "Tool response: %s", json.dumps(tool_args["ids"])
                    )
                    self.last_usage = usage
                    return SIDChatResponse(output_parsed=result), inputs, usage
                if name not in self.tool_schemas:
                    raise ValueError(f"Tool {name} not found in registered tools.")
                resp = call_tool(
                    tool_args,
                    self.tool_schemas[name],
                    agent_state=agent_state,
                )
                active_logger.info("Tool response: %s", resp)
                inputs.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": resp,
                })

            if inputs[-1]["role"] == "tool":
                inputs[-1]["content"] += self._turn_budget_msg(turn)

        self.last_usage = usage
        return SIDChatResponse(output_parsed=_ids_to_search_results([])), inputs, usage
