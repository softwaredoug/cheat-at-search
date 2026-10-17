"""Agent scaffold for SID-2's Responses API and episode protocol."""

import json
from time import sleep
from typing import Any, Literal
from uuid import uuid4

from openai import OpenAI
from pydantic import BaseModel
from sid import DocumentCache

from cheat_at_search.agent.search_client import (
    Agent,
    SearchResults,
    initialize_resource_counters,
    record_resource,
)
from cheat_at_search.agent.sid_tools import (
    build_sid2_tool_config,
    call_sid2_tool,
    resolve_reported_ids,
)
from cheat_at_search.data_dir import key_for_provider
from cheat_at_search.logger import log_to_stdout

logger = log_to_stdout("sid2_search_client")


def _log_preview(value: Any, max_chars: int = 1200) -> str:
    """Keep verbose tool calls and observations bounded in the trace."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars - 3]}... (total {len(text)} chars)"


def _log_reasoning_summaries(response: Any, active_logger) -> None:
    """Log available reasoning summaries without exposing encrypted content."""
    for item in getattr(response, "output", []):
        if getattr(item, "type", None) != "reasoning":
            continue
        for summary in getattr(item, "summary", None) or []:
            text = (
                summary.get("text")
                if isinstance(summary, dict)
                else getattr(summary, "text", None)
            )
            if text:
                active_logger.info("Reasoning summary: %s", _log_preview(text))


def _call_sid(client, inputs, request, num_attempts=5):
    for attempt in range(num_attempts):
        try:
            return client.responses.create(input=inputs, **request)
        except Exception as exc:
            if attempt + 1 == num_attempts:
                raise
            logger.warning(
                "SID-2 request failed on attempt %d/%d: %s",
                attempt + 1,
                num_attempts,
                exc,
            )
            sleep(3 ** attempt)  # Exponential backoff


class SIDChatResponse(BaseModel):
    output_parsed: SearchResults


class SIDAgent(Agent):
    """SID-2 Responses client with schema generation and an agentic tool loop.

    Callers provide backend search tools. SID-2 ``read`` and
    ``report_helpful_ids`` tools are added automatically.
    """

    def __init__(
        self,
        tools,
        *,
        model: str = "sid-2",
        max_turns: int = 10,
        reward: str = "ndcg",
        reasoning_effort: Literal["low", "medium", "high"] = "low",
    ) -> None:
        """Configure the SID-2 model, search tools, and episode limits."""

        self.reward = reward
        self.model = model
        self.max_turns = max_turns
        self.reasoning_effort = reasoning_effort
        self.search_tools = list(tools)
        self.tools, self.tool_examples = build_sid2_tool_config(self.search_tools)
        self.api_key = key_for_provider("sid")
        self.client = OpenAI(
            base_url="https://api.sid-1.com/v1",
            api_key=self.api_key,
        )

    def _request_config(self, instructions: str, session_id: str) -> dict[str, Any]:
        """Build the fixed Responses request parameters for one SID-2 episode."""
        return {
            "model": self.model,
            "max_output_tokens": 4096,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "instructions": instructions,
            "tools": self.tools,
            "reasoning": {"effort": self.reasoning_effort},
            "extra_body": {
                "sid": {
                    "tool_examples": self.tool_examples,
                    "reward": self.reward,
                }
            },
            "extra_headers": {"x-session-affinity": session_id},
        }

    def _process_report_helpful_ids(
        self,
        function_calls: list[Any],
        has_searched: bool,
        document_cache: DocumentCache,
    ) -> SIDChatResponse | str | None:
        """Validate a terminal report call and map its IDs to backend IDs."""
        report_calls = [
            item for item in function_calls if item.name == "report_helpful_ids"
        ]
        if not report_calls:
            return None

        report_call = report_calls[0]

        def error(message: str) -> str:
            return f"Error executing report_helpful_ids: {message}"

        if len(function_calls) != 1:
            return error(
                "report_helpful_ids must be called by itself, without other tool calls."
            )
        if not has_searched:
            return error("Cannot report in the first turn. Must make a search first.")

        try:
            report_arguments = json.loads(report_call.arguments)
        except (TypeError, json.JSONDecodeError) as exc:
            return error(f"Invalid arguments: {exc}")

        report_ids = (
            report_arguments.get("ids")
            if isinstance(report_arguments, dict)
            else None
        )
        if (
            not isinstance(report_ids, list)
            or not report_ids
            or not all(isinstance(reference, str) for reference in report_ids)
        ):
            return error("report_helpful_ids requires an ids list of strings.")

        return SIDChatResponse(
            output_parsed=SearchResults(
                ranked_results=resolve_reported_ids(report_ids, document_cache)
            )
        )

    def loop(
        self,
        inputs=None,
        agent_state: dict | None = None,
        logger=None,
    ) -> tuple[Any, list]:
        """Run an episode until SID-2 reports IDs or the turn limit is reached."""
        active_logger = logger or globals()["logger"]
        if agent_state is None:
            agent_state = {}
        initialize_resource_counters(agent_state)
        agent_state["trace_logger"] = active_logger
        if inputs is None:
            raise ValueError("SIDAgent requires a system prompt in the inputs list.")

        inputs = list(inputs)
        if not inputs:
            raise ValueError("SIDAgent requires a system prompt in the inputs list.")

        system_message = inputs[0]
        if not isinstance(system_message, dict) or system_message.get("role") != "system":
            raise ValueError("The first input must be a system prompt with role 'system'.")
        if len(inputs) < 2:
            raise ValueError(
                "SIDAgent requires a system prompt followed by a user prompt."
            )
        instructions = system_message.get("content")
        if not isinstance(instructions, str):
            raise ValueError("The system prompt content must be a string.")
        if not isinstance(inputs[1], dict) or inputs[1].get("role") != "user":
            raise ValueError("The second input must be a user prompt with role 'user'.")

        active_logger.info("System prompt: %s", instructions)
        for message in inputs[1:]:
            if isinstance(message, dict) and message.get("role") == "user":
                active_logger.info("User prompt: %s", message.get("content", ""))

        request = self._request_config(instructions, uuid4().hex)
        document_cache = DocumentCache(range_mode="lenient")
        has_searched = False
        search_tool_names = {tool.__name__ for tool in self.search_tools}

        for turn in range(1, self.max_turns + 1):
            active_logger.info(
                "SID-2 turn %d/%d (%d remaining)",
                turn,
                self.max_turns,
                self.max_turns - turn,
            )
            response = _call_sid(self.client, inputs[1:], request)
            inputs.extend(response.output)

            _log_reasoning_summaries(response, active_logger)
            for item in response.output:
                if getattr(item, "type", None) == "message":
                    text = getattr(response, "output_text", None)
                    if text:
                        active_logger.info("Assistant message: %s", _log_preview(text))

            response_usage = getattr(response, "usage", None)
            input_details = getattr(response_usage, "input_tokens_details", None)
            output_details = getattr(response_usage, "output_tokens_details", None)
            record_resource(
                agent_state,
                "input_tokens",
                getattr(response_usage, "input_tokens", 0) or 0,
            )
            record_resource(
                agent_state,
                "output_tokens",
                getattr(response_usage, "output_tokens", 0) or 0,
            )
            active_logger.info(
                "Token usage: input=%s cached_input=%s output=%s reasoning=%s",
                getattr(response_usage, "input_tokens", 0) or 0,
                getattr(input_details, "cached_tokens", 0) or 0,
                getattr(response_usage, "output_tokens", 0) or 0,
                getattr(output_details, "reasoning_tokens", 0) or 0,
            )

            function_calls = [
                item for item in response.output if item.type == "function_call"
            ]
            record_resource(agent_state, "num_tool_calls", len(function_calls))
            active_logger.debug("SID-2 usage: %s", response_usage)

            # An incomplete response may end in the middle of a tool call. Do
            # not execute a truncated call or continue the episode.
            if getattr(response, "status", None) == "incomplete":
                active_logger.warning(
                    "SID-2 response was incomplete; ending episode without a report."
                )
                break

            for item in function_calls:
                active_logger.info(
                    "Tool called: %s args=%s",
                    item.name,
                    _log_preview(item.arguments or ""),
                )

            report_result = self._process_report_helpful_ids(
                function_calls, has_searched, document_cache
            )
            if isinstance(report_result, SIDChatResponse):
                active_logger.info(
                    "Tool response: %s",
                    _log_preview(report_result.output_parsed.ranked_results),
                )
                return report_result, inputs

            for item in response.output:
                if item.type != "function_call":
                    continue
                if item.name == "report_helpful_ids":
                    observation = report_result
                else:
                    has_searched = has_searched or item.name in search_tool_names
                    observation = call_sid2_tool(
                        item,
                        self.search_tools,
                        document_cache=document_cache,
                        agent_state=agent_state,
                    )
                active_logger.info(
                    "Tool response: %s",
                    _log_preview(observation),
                )
                inputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": observation,
                    }
                )

            if turn < self.max_turns:
                remaining = self.max_turns - turn
                budget_message = (
                    f"You have {remaining} turns remaining out of "
                    f"{self.max_turns} total."
                )
                inputs.append({"role": "user", "content": budget_message})
                active_logger.info("User prompt: %s", budget_message)
                if remaining == 1:
                    final_turn_message = (
                        "This is your last turn: you must call "
                        "report_helpful_ids now. Anything you have found "
                        "but not reported is lost when this turn ends."
                    )
                    inputs.append({"role": "user", "content": final_turn_message})
                    active_logger.info("User prompt: %s", final_turn_message)

        active_logger.warning(
            "SID-2 reached max_turns=%d without reporting helpful IDs.",
            self.max_turns,
        )
        return SIDChatResponse(output_parsed=SearchResults(ranked_results=[])), inputs
