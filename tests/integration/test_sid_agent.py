import os

import pytest
from openai import PermissionDeniedError

from cheat_at_search.agent.sid_agent import SIDAgent


SID_DOCUMENT = {
    "id": "sid-live-integration-document",
    "title": "SID backend integration test document",
    "description": (
        "This document exists only in the in-memory test search tool. "
        "It verifies that SID can call a backend tool and report its result."
    ),
}


def search(query: str) -> list[dict]:
    """Search the in-memory SID integration-test corpus."""
    return [SID_DOCUMENT]


def read(id: str) -> dict:
    """Read a document by its stable ID from the SID integration-test corpus."""
    return SID_DOCUMENT


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("SID_API_KEY"),
    reason="Set SID_API_KEY to run the live SID backend test.",
)
def test_sid_agent_runs_against_live_sid_backend():
    system_prompt = """
You are validating a search integration. Find the single document that answers
the user's request, using the provided search tools.

Call search with a query that finds the document. After reviewing the tool
results, call report_helpful_ids with only the stable IDs returned by search,
most helpful first. Do not invent IDs.

The read tool accepts one document id and returns its full contents:
<tool_call>
{"name": "read", "arguments": {"id": "document_id_from_search"}}
</tool_call>

To search, provide a query:
<tool_call>
{"name": "search", "arguments": {"query": "SID backend integration-test document"}}
</tool_call>

To finish, report the IDs returned by search:
<tool_call>
{"name": "report_helpful_ids", "arguments": {"ids": ["document_id_from_search"]}}
</tool_call>
"""
    agent = SIDAgent(tools=[search, read], max_turns=4)

    try:
        response, _, usage = agent.loop(
            inputs=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Find the SID backend integration-test document.",
                },
            ],
            return_usage=True,
        )
    except PermissionDeniedError as exc:
        if "You do not have access to model sid-1" in str(exc):
            pytest.skip("Configured SID_API_KEY is not authorized for sid-1.")
        raise

    assert response.output_parsed.ranked_results == [SID_DOCUMENT["id"]]
    assert usage["num_tool_calls"] >= 2
