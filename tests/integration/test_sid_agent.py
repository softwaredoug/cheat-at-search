import os
from typing import Annotated, Literal

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


def _make_annotated_search(calls):
    def search(
        keywords: Annotated[
            str,
            "Search query keywords.",
            "SID backend integration-test document",
        ],
        top_k: Annotated[int, "Maximum number of results.", 5] = 5,
        mode: Annotated[
            Literal["exact", "broad"], "Search mode.", "exact"
        ] = "exact",
    ) -> list[dict]:
        """Search using annotated scalar and Literal arguments."""
        calls.append({"keywords": keywords, "top_k": top_k, "mode": mode})
        return [SID_DOCUMENT]

    return search


def _make_annotated_fields_search(calls):
    def search(
        keywords: Annotated[
            str,
            "Search query keywords.",
            "SID backend integration-test document",
        ],
        fields: Annotated[
            list[str],
            "Fields to search.",
            ["title", "description"],
        ] = ["title", "description"],
    ) -> list[dict]:
        """Search selected text fields using annotated list arguments."""
        calls.append({"keywords": keywords, "fields": fields})
        return [SID_DOCUMENT]

    return search


def _run_live_sid(search_tool, system_prompt):
    agent = SIDAgent(tools=[search_tool, read], max_turns=4)
    try:
        return agent.loop(
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


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("SID_API_KEY"),
    reason="Set SID_API_KEY to run the live SID backend test.",
)
def test_sid_agent_accepts_annotated_scalar_and_literal_arguments():
    calls = []
    response, _, usage = _run_live_sid(
        _make_annotated_search(calls),
        system_prompt=(
            "Call the search tool once with keywords describing the requested document, "
            "top_k set to 5, and mode set to exact. Then report the document IDs it returns."
        ),
    )

    assert response.output_parsed.ranked_results == [SID_DOCUMENT["id"]]
    assert usage["num_tool_calls"] >= 2
    assert calls
    assert isinstance(calls[0]["keywords"], str)
    assert isinstance(calls[0]["top_k"], int)
    assert calls[0]["mode"] in {"exact", "broad"}


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("SID_API_KEY"),
    reason="Set SID_API_KEY to run the live SID backend test.",
)
def test_sid_agent_accepts_annotated_list_argument_examples():
    calls = []
    response, _, usage = _run_live_sid(
        _make_annotated_fields_search(calls),
        system_prompt=(
            "Call the search tool with keywords describing the requested document and "
            "fields set to [\"title\", \"description\"]. Then report the document IDs it returns."
        ),
    )

    assert response.output_parsed.ranked_results == [SID_DOCUMENT["id"]]
    assert usage["num_tool_calls"] >= 2
    assert calls
    assert calls[0]["fields"] == ["title", "description"]
