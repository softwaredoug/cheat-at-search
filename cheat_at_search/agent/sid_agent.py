from cheat_at_search.agent.search_client import Agent, SearchResults, SearchResult
from cheat_at_search.data_dir import key_for_provider
from cheat_at_search.agent.sid_tool import tool_schema, call_tool, format_tool_schema
from openai import OpenAI
from hashlib import md5
import json


TOOL_HEADER = "## Available Search Tools"
SID_STOP_FUNCTION = "report_helpful_ids"


def _ids_to_search_results(ids: list[str]) -> SearchResults:
    """Convert a list of IDs to SearchResults."""
    results = []
    for rank_no, _id in enumerate(ids):
        search_result = SearchResult(id=_id, rank=rank_no + 1)
        results.append(search_result)

    return SearchResults(results=results)


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
        self.tool_schemas = {}
        for tool in tools:
            schema = tool_schema(tool)
            self.tool_schemas[schema.name] = schema

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

    def loop(self, inputs: list[dict]):
        """Implement agentic loop over calls."""
        # Check for system prompt in inputs
        if len(inputs) < 2:
            raise ValueError("At least two inputs are required: a system prompt and a user prompt (search query).")

        if inputs[0]["role"] != "system":
            raise ValueError("First input must be a system prompt.")

        system_prompt = inputs[0]["content"]

        if TOOL_HEADER not in system_prompt:
            tools_formatted = [format_tool_schema(tool) for tool in self.tool_schemas.values()]
            system_prompt += "\n\n" + TOOL_HEADER + ":\n" + "\n".join(tools_formatted)
            inputs[0]["content"] = system_prompt

        if inputs[1]["role"] != "user":
            raise ValueError("Second input must be a user prompt.")

        tools = [{"type": "function", "function": {"name": "dummy"}}]

        for turn in range(self.max_turns):

            response = self.client.chat.completions.create(
                model=self.model,
                messages=inputs,
                tools=tools,
            )
            assistant_message = response.choices[0].message
            tool_calls = assistant_message.tool_calls or []
            inputs.append(_assistant_message_to_dict(assistant_message))

            if not tool_calls:
                # No tool calls, return the assistant's message
                return assistant_message["content"]

            for tool_call in tool_calls:
                name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                call_id = tool_call.id
                if name == SID_STOP_FUNCTION:
                    # Stop function called, return the assistant's message
                    return _ids_to_search_results(tool_args["ids"])
                resp = call_tool(tool_args, self.tool_schemas[name])
                inputs.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": resp,
                })
