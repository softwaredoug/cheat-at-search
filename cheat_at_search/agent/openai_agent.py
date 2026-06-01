from cheat_at_search.agent.search_client import Agent, SearchResults
from cheat_at_search.data_dir import key_for_provider
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.agent.pydantize import make_tool_adapter
from openai import BadRequestError, OpenAI
from typing import Optional
from hashlib import md5
import json
import textwrap


logger = log_to_stdout("openai_search_client")


class OpenAIAgent(Agent):
    def __init__(
        self,
        tools,
        model: str,
        max_tokens: Optional[int] = None,
        response_model=None,
        reasoning_level: str = "medium",
        summary: bool = True,
    ):
        self.search_tools = {tool.__name__: make_tool_adapter(tool) for tool in tools}

        self.provider = model.split("/")[0]
        self.model = model.split("/")[-1]
        self.openai_key = key_for_provider("openai")
        if self.provider != "openai":
            raise ValueError(
                f"Provider {self.provider} is not supported. This client only supports OpenAI."
            )
        self.response_model = response_model
        self.openai = OpenAI(api_key=self.openai_key)
        self.last_usage = None
        self.max_tokens = max_tokens
        self.reasoning_level = reasoning_level
        self.summary = summary

    def config_hash(self) -> str:
        tool_specs = []
        for tool in self.search_tools.values():
            tool_spec = tool[1]
            tool_specs.append(
                {
                    "name": tool_spec.get("name"),
                    "description": tool_spec.get("description"),
                    "parameters": tool_spec.get("parameters"),
                }
            )
        tool_specs = sorted(tool_specs, key=lambda spec: spec.get("name") or "")

        response_model = None
        if self.response_model is not None:
            response_model = (
                f"{self.response_model.__module__}.{self.response_model.__qualname__}"
            )

        payload = {
            "provider": self.provider,
            "model": f"{self.provider}/{self.model}",
            "response_model": response_model,
            "max_tokens": self.max_tokens,
            "reasoning_level": self.reasoning_level,
            "tools": tool_specs,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return md5(serialized.encode("utf-8")).hexdigest()

    def chat(
        self,
        inputs=None,
        agent_state: Optional[dict] = None,
        return_usage=False,
        logger=None,
    ) -> SearchResults:
        """Chat, handle any response."""
        active_logger = logger or globals()["logger"]
        if agent_state is not None:
            agent_state["trace_logger"] = active_logger
        tools = []
        for tool in self.search_tools.values():
            tool_spec = tool[1]
            tools.append(tool_spec)
        usage = {"input_tokens": 0, "output_tokens": 0, "num_tool_calls": 0}
        try:
            tool_calls_found = True
            while tool_calls_found:
                reasoning = {
                    "effort": self.reasoning_level,
                    "summary": "auto" if self.summary else "none",
                }
                resp = self._call_responses_with_retry(
                    inputs=inputs,
                    tools=tools,
                    reasoning=reasoning,
                    active_logger=active_logger,
                )
                # Iterate over tool calls
                inputs += resp.output

                usage["input_tokens"] += resp.usage.input_tokens
                usage["output_tokens"] += resp.usage.output_tokens

                total_tokens = usage["input_tokens"] + usage["output_tokens"]
                if self.summary:
                    active_logger.info("--")
                    active_logger.info("InpTok: %s", resp.usage.input_tokens)
                    active_logger.info("OutTok: %s", resp.usage.output_tokens)
                    for item in resp.output:
                        if item.type == "reasoning":
                            active_logger.info("Reasoning:")
                            for summary_item in item.summary:
                                active_logger.info(
                                    "%s\n",
                                    textwrap.fill(summary_item.text, 80),
                                )
                            item.summary = []

                active_logger.debug("Usage: ", resp.usage)
                if self.max_tokens and total_tokens >= self.max_tokens:
                    active_logger.info(
                        f"Reached max tokens limit of {self.max_tokens}. Stopping further tool calls."
                    )
                    break

                tool_calls_found = False

                for item in resp.output:
                    if item.type == "function_call":
                        tool_calls_found = True
                        usage["num_tool_calls"] += 1
                        tool_name = item.name
                        if tool_name not in self.search_tools:
                            raise ValueError(
                                f"Tool {tool_name} not found in registered tools."
                            )
                        tool_calls_found = True
                        tool = self.search_tools[tool_name]
                        ToolArgsModel = tool[0]
                        tool_fn = tool[2]

                        arg_preview = item.arguments or ""
                        if len(arg_preview) > 60:
                            arg_preview = f"{arg_preview[:57]}..."
                        active_logger.info(
                            "Tool called: %s args=%s",
                            tool_name,
                            arg_preview,
                        )

                        fn_args: ToolArgsModel = ToolArgsModel.model_validate_json(
                            item.arguments
                        )
                        py_resp, json_resp = tool_fn(fn_args, agent_state=agent_state)
                        resp_preview = json_resp or ""
                        if len(resp_preview) > 1000:
                            resp_preview = f"{resp_preview[:997]}..."
                            resp_preview = f"{resp_preview} (total {len(json_resp)} chars)"
                        active_logger.info("Tool response: %s", resp_preview)
                        # 4. Provide function call results to the model
                        inputs.append(
                            {
                                "type": "function_call_output",
                                "call_id": item.call_id,
                                "output": json_resp,
                            }
                        )
            if return_usage:
                resp.usage = usage
            return resp, inputs, usage
        except Exception as e:
            active_logger.error("Error calling MCP search tool: %s", e)
            raise e

    def _call_responses_with_retry(self, inputs, tools, reasoning, active_logger):
        attempts = 2
        for attempt in range(1, attempts + 1):
            try:
                if self.response_model:
                    return self.openai.responses.parse(
                        model=self.model,
                        input=inputs,
                        tools=tools,
                        reasoning=reasoning,
                        text_format=self.response_model,
                    )
                return self.openai.responses.create(
                    model=self.model,
                    input=inputs,
                    tools=tools,
                    reasoning=reasoning,
                )
            except BadRequestError as exc:
                if getattr(exc, "status_code", None) == 400:
                    raise
                if attempt == attempts:
                    raise
                active_logger.warning(
                    "OpenAI responses call failed (%s). Retrying...",
                    exc,
                )
            except Exception as exc:
                if attempt == attempts:
                    raise
                active_logger.warning(
                    "OpenAI responses call failed (%s). Retrying...",
                    exc,
                )

    def loop(
        self,
        inputs=None,
        agent_state=None,
        return_usage=False,
        logger=None,
    ) -> SearchResults:
        """Issue a 'search' and expect structured output response."""
        assert self.response_model is not None, (
            "response_model must be set for structured search results."
        )
        resp, _, usage = self.chat(
            inputs=inputs,
            agent_state=agent_state,
            logger=logger,
        )
        self.last_usage = resp.usage
        if return_usage:
            return resp.output_parsed, usage
        return resp.output_parsed
