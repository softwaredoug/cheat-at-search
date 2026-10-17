import os
from typing import Annotated

import pytest
from openai import PermissionDeniedError

from cheat_at_search.agent.sid_agent import SIDAgent, SIDChatResponse


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("SID_API_KEY"),
    reason="Set SID_API_KEY to run the live SID-2 integration test.",
)
def test_sid_agent_requests_hello_world_search():
    searches = []

    def search(
        query: Annotated[str, "Terms describing the greeting to find.", "hello world"],
    ) -> list[dict]:
        """Search the hello-world greeting catalog."""
        searches.append(query)
        return [
            {
                "id": "hello-world",
                "title": "Hello world",
                "content": "A friendly hello world greeting.",
            }
        ]

    agent = SIDAgent(tools=[search], reasoning_effort="low")
    agent_state = {}
    inputs = [
        {
            "role": "system",
            "content": """You are an expert research assistant searching a small catalog of greeting documents.

Steps:
1. Reflect on what information is needed to answer the question and use the provided search tools to find documents. Each document has an id.
2. Repeat step 1 until all documents necessary and sufficient to answer the question have been found. Take as many turns and searches as needed – you can make multiple searches per turn! Most questions will require multiple turns. Many will need more.
3. Use the report_helpful_ids tool to report the most helpful document ids. List the most helpful document ids first (important!).

The episode ends once report_helpful_ids is called.""",
        },
        {"role": "user", "content": "Find the hello world greeting."},
    ]
    try:
        response, inputs = agent.loop(inputs=inputs, agent_state=agent_state)
    except PermissionDeniedError as exc:
        if "sid-2" in str(exc).lower():
            pytest.skip("Configured SID_API_KEY is not authorized for sid-2.")
        raise

    assert isinstance(response, SIDChatResponse)
    assert response.output_parsed.ranked_results == ["hello-world"]
    assert len(response.output_parsed.ranked_results) > 0
    assert searches
    assert any("hello" in query.lower() for query in searches)
    assert agent_state["resource.num_tool_calls"] >= 1
    tool_outputs = [
        item
        for item in inputs
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    assert tool_outputs
    assert any('<doc id="' in item["output"] for item in tool_outputs)
