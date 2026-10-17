import os
from typing import Annotated, Literal

import pytest

from cheat_at_search.agent.openai_agent import OpenAIAgent


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run the live OpenAI agent test.",
)
def test_openai_agent_accepts_annotated_scalar_literal_and_list_arguments():
    calls = []

    def annotated_search(
        query: Annotated[str, "Search query.", "annotation integration test"],
        fields: Annotated[
            list[str],
            "Fields to search.",
            ["title", "description"],
        ],
        limit: Annotated[int, "Maximum results.", 3] = 3,
        mode: Annotated[
            Literal["title", "description"], "Search mode.", "title"
        ] = "title",
    ) -> dict[str, str]:
        """Run an annotated search for the integration test."""
        calls.append(
            {
                "query": query,
                "limit": limit,
                "mode": mode,
                "fields": fields,
            }
        )
        return {"status": "ok"}

    agent = OpenAIAgent(
        tools=[annotated_search],
        model=f"openai/{os.getenv('OPENAI_AGENT_TEST_MODEL', 'gpt-5-mini')}",
        reasoning_level="low",
    )
    agent_state = {}
    response, _ = agent.loop(
        inputs=[
            {
                "role": "system",
                "content": (
                    "Call annotated_search exactly once with query "
                    "'annotation integration test', limit 3, mode 'title', and "
                    "fields ['title', 'description']. Then stop."
                ),
            },
            {"role": "user", "content": "Run the annotation integration check."},
        ],
        agent_state=agent_state,
    )

    assert calls == [
        {
            "query": "annotation integration test",
            "limit": 3,
            "mode": "title",
            "fields": ["title", "description"],
        }
    ]
    assert agent_state["resource.num_tool_calls"] == 1
    assert response is not None
