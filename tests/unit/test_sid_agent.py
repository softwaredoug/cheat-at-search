from types import SimpleNamespace
from typing import Annotated, Any, Dict, Literal, Sequence, cast
from unittest.mock import Mock, patch

import pytest

from cheat_at_search.agent.sid_agent import SIDAgent, SIDChatResponse


def filtered_search(
    query: Annotated[str, "What to find", "red hiking jacket"],
    limit: Annotated[int, "Maximum results", 5] = 5,
    mode: Annotated[Literal["exact", "broad"], "Search mode", "broad"] = "broad",
) -> list[dict]:
    """Search the product catalog."""
    return []


def w_scenarios(scenarios: Dict[str, Dict[str, Any]]):
    """Decorate scenarios with readable pytest IDs."""
    return pytest.mark.parametrize(
        cast(Sequence[str], [key for key in scenarios.values()][0].keys()),
        [tuple(scenario.values()) for scenario in scenarios.values()],
        ids=list(scenarios.keys()),
    )


@patch("cheat_at_search.agent.sid_agent.OpenAI")
@patch("cheat_at_search.agent.sid_agent.key_for_provider", return_value="sid-test-key")
@patch("cheat_at_search.agent.sid_agent.DocumentCache")
def test_sid_agent_report_returns_legacy_search_results_shape(
    mock_document_cache, mock_key_for_provider, mock_openai
):
    def search(query: str) -> list[dict]:
        """Search a hello-world test corpus."""
        return [
            {
                "id": "backend-doc-1",
                "title": "Hello world",
                "content": "A friendly hello world greeting.",
            }
        ]

    episode_cache = mock_document_cache.return_value
    episode_cache.__contains__.return_value = False
    episode_cache.apply_snippet.return_value.render_xml.return_value = (
        '<doc id="123456">A friendly hello world greeting.</doc>'
    )
    episode_cache.contains_model_facing_id.return_value = True
    episode_cache.to_data_id.return_value = "backend-doc-1"
    search_response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                name="search",
                arguments='{"query": "hello world"}',
                call_id="call-search",
            )
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=2),
    )
    report_response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                name="report_helpful_ids",
                arguments='{"ids": ["123456#0:10", "123456"]}',
                call_id="call-report",
            )
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=2),
    )
    mock_openai.return_value.responses.create.side_effect = [
        search_response,
        report_response,
    ]
    agent = SIDAgent(tools=[search], max_turns=2)
    state = {}
    active_logger = Mock()
    inputs = [
        {"role": "system", "content": "Search and report helpful documents."},
        {"role": "user", "content": "Find the hello world greeting."},
    ]

    final_response, inputs = agent.loop(
        inputs, agent_state=state, logger=active_logger
    )

    mock_document_cache.assert_called_once_with(range_mode="lenient")
    episode_cache.add_document.assert_called_once_with(
        "backend-doc-1",
        {"title": "Hello world", "content": "A friendly hello world greeting."},
    )
    assert isinstance(final_response, SIDChatResponse)
    assert final_response.output_parsed.ranked_results == ["backend-doc-1"]
    assert inputs[-1] is report_response.output[0]
    request_headers = [
        call.kwargs["extra_headers"]
        for call in mock_openai.return_value.responses.create.call_args_list
    ]
    assert request_headers[0] == request_headers[1]
    assert state["resource.num_tool_calls"] == 2
    active_logger.info.assert_any_call(
        "Tool called: %s args=%s",
        "report_helpful_ids",
        '{"ids": ["123456#0:10", "123456"]}',
    )
    active_logger.info.assert_any_call("Tool response: %s", '["backend-doc-1"]')
    active_logger.info.assert_any_call(
        "User prompt: %s", "You have 1 turns remaining out of 2 total."
    )
    active_logger.info.assert_any_call(
        "User prompt: %s",
        "This is your last turn: you must call report_helpful_ids now. "
        "Anything you have found but not reported is lost when this turn ends.",
    )


@patch("cheat_at_search.agent.sid_agent.OpenAI")
@patch("cheat_at_search.agent.sid_agent.key_for_provider", return_value="sid-test-key")
def test_sid_agent_request_config_uses_generated_tools_and_examples(
    mock_key_for_provider, mock_openai
):
    agent = SIDAgent(
        tools=[filtered_search],
        reward="ndcg",
        reasoning_effort="medium",
    )

    request = agent._request_config("Find product documents.", "episode-1")

    assert request["model"] == "sid-2"
    assert request["instructions"] == "Find product documents."
    assert request["tools"] == agent.tools
    assert [tool["name"] for tool in request["tools"]] == [
        "filtered_search",
        "read",
        "report_helpful_ids",
    ]
    assert request["reasoning"] == {"effort": "medium"}
    assert request["extra_body"]["sid"] == {
        "tool_examples": agent.tool_examples,
        "reward": "ndcg",
    }
    assert request["extra_headers"] == {"x-session-affinity": "episode-1"}
    mock_key_for_provider.assert_called_once_with("sid")
    mock_openai.assert_called_once_with(
        base_url="https://api.sid-1.com/v1",
        api_key="sid-test-key",
    )


@patch("cheat_at_search.agent.sid_agent.OpenAI")
@patch("cheat_at_search.agent.sid_agent.key_for_provider", return_value="sid-test-key")
def test_sid_agent_loop_sends_user_history_and_returns_first_response(
    mock_key_for_provider, mock_openai
):
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="reasoning",
                summary=[SimpleNamespace(text="Search for a greeting.")],
            ),
            SimpleNamespace(
                type="function_call",
                name="filtered_search",
                arguments='{"query": "red hiking jacket"}',
                call_id="call-search-1",
            ),
        ],
        usage=SimpleNamespace(input_tokens=23, output_tokens=4),
    )
    mock_openai.return_value.responses.create.return_value = response
    agent = SIDAgent(tools=[filtered_search], max_turns=1)
    inputs = [
        {"role": "system", "content": "Find relevant product documents."},
        {"role": "user", "content": "Find a red hiking jacket."},
    ]

    agent_state = {}
    active_logger = Mock()
    returned_response, updated_inputs = agent.loop(
        inputs, agent_state=agent_state, logger=active_logger
    )

    mock_openai.return_value.responses.create.assert_called_once()
    request = mock_openai.return_value.responses.create.call_args.kwargs
    assert request["input"] == [inputs[1]]
    assert request["instructions"] == inputs[0]["content"]
    assert request["tools"] == agent.tools
    assert request["extra_body"]["sid"]["tool_examples"] == agent.tool_examples
    assert request["extra_body"]["sid"]["reward"] == "ndcg"
    assert request["extra_headers"]["x-session-affinity"]
    assert isinstance(returned_response, SIDChatResponse)
    assert returned_response.output_parsed.ranked_results == []
    assert updated_inputs == [
        *inputs,
        *response.output,
        {
            "type": "function_call_output",
            "call_id": "call-search-1",
            "output": "No results found. Try a different query.",
        },
    ]
    assert {
        key: value
        for key, value in agent_state.items()
        if key.startswith("resource.")
    } == {
        "resource.input_tokens": 23,
        "resource.output_tokens": 4,
        "resource.num_tool_calls": 1,
    }
    assert response.usage.input_tokens == 23
    assert response.usage.output_tokens == 4
    assert agent_state["trace_logger"] is active_logger
    active_logger.info.assert_any_call("System prompt: %s", inputs[0]["content"])
    active_logger.info.assert_any_call("User prompt: %s", inputs[1]["content"])
    active_logger.info.assert_any_call("SID-2 turn %d/%d (%d remaining)", 1, 1, 0)
    active_logger.info.assert_any_call("Reasoning summary: %s", "Search for a greeting.")
    active_logger.info.assert_any_call(
        "Tool called: %s args=%s",
        "filtered_search",
        '{"query": "red hiking jacket"}',
    )
    active_logger.info.assert_any_call(
        "Tool response: %s", "No results found. Try a different query."
    )
    active_logger.info.assert_any_call(
        "Token usage: input=%s cached_input=%s output=%s reasoning=%s",
        23,
        0,
        4,
        0,
    )


@patch("cheat_at_search.agent.sid_agent.OpenAI")
@patch("cheat_at_search.agent.sid_agent.key_for_provider", return_value="sid-test-key")
@patch("cheat_at_search.agent.sid_agent.DocumentCache")
def test_sid_agent_creates_a_document_cache_for_each_episode(
    mock_document_cache, mock_key_for_provider, mock_openai
):
    episode_caches = [Mock(), Mock()]
    mock_document_cache.side_effect = episode_caches
    mock_openai.return_value.responses.create.return_value = SimpleNamespace(
        output=[],
        usage=SimpleNamespace(input_tokens=0, output_tokens=0),
    )
    agent = SIDAgent(tools=[filtered_search], max_turns=1)
    inputs = [
        {"role": "system", "content": "Find relevant product documents."},
        {"role": "user", "content": "Find a red hiking jacket."},
    ]

    agent.loop(inputs)
    agent.loop(inputs)

    assert mock_document_cache.call_count == 2
    assert all(
        call.kwargs == {"range_mode": "lenient"}
        for call in mock_document_cache.call_args_list
    )
    assert episode_caches[0] is not episode_caches[1]


INVALID_LOOP_INPUT_SCENARIOS = {
    "no messages": {"inputs": [], "error": "system prompt in the inputs list"},
    "first message is user": {
        "inputs": [{"role": "user", "content": "Find something."}],
        "error": "first input must be a system prompt",
    },
    "second message is not user": {
        "inputs": [
            {"role": "system", "content": "Search."},
            {"role": "assistant", "content": "Not a user query."},
        ],
        "error": "second input must be a user prompt",
    },
    "system content is not text": {
        "inputs": [
            {"role": "system", "content": [{"type": "input_text", "text": "Search."}]},
            {"role": "user", "content": "Find something."},
        ],
        "error": "system prompt content must be a string",
    },
}


@w_scenarios(INVALID_LOOP_INPUT_SCENARIOS)
def test_sid_agent_loop_validates_system_and_user_inputs(inputs, error):
    agent = object.__new__(SIDAgent)

    with pytest.raises(ValueError, match=error):
        agent.loop(inputs)
