from cheat_at_search.wands_data import products
from cheat_at_search.agent.openai_agent import OpenAIAgent
from cheat_at_search.agent.search_client import SearchResults
from cheat_at_search.agent.strategy import ReasoningSearchStrategy
from cheat_at_search.agent.harness import Harness
from cheat_at_search.tokenizers import snowball_tokenizer
from typing import List, Dict, Literal
from unittest.mock import patch, Mock
import httpx
import pytest
from openai import BadRequestError
from searcharray import SearchArray
import numpy as np
from pydantic import BaseModel, Field

from .openai_responses_mock import (
    MockOpenAIResponses,
    function_call,
    responses_api_response,
)

products["product_name_snowball"] = SearchArray.index(
    products["product_name"], tokenizer=snowball_tokenizer
)

products["description_snowball"] = SearchArray.index(
    products["product_description"], tokenizer=snowball_tokenizer
)


def configure_mock_openai(mock_key_for_provider, mock_openai):
    mock_key_for_provider.return_value = "test-openai-key"
    mock_responses = MockOpenAIResponses()
    mock_openai.return_value = mock_responses.client
    return mock_responses


@patch("cheat_at_search.agent.openai_agent.OpenAI")
@patch("cheat_at_search.agent.openai_agent.key_for_provider")
def test_responses_retry_on_exception(mock_key_for_provider, mock_openai):
    mock_agent_openai = configure_mock_openai(mock_key_for_provider, mock_openai)
    success_response = responses_api_response()
    mock_agent_openai.client.responses.create = Mock(
        side_effect=[Exception("boom"), success_response]
    )

    search_client = OpenAIAgent(
        tools=[],
        model="openai/gpt-5",
    )

    resp, _, _ = search_client.chat(inputs=[])

    assert resp is success_response
    assert mock_agent_openai.client.responses.create.call_count == 2


@patch("cheat_at_search.agent.openai_agent.OpenAI")
@patch("cheat_at_search.agent.openai_agent.key_for_provider")
def test_responses_no_retry_on_bad_request_400(
    mock_key_for_provider, mock_openai
):
    mock_agent_openai = configure_mock_openai(mock_key_for_provider, mock_openai)
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    bad_request = BadRequestError(
        "bad request",
        response=httpx.Response(400, request=request),
        body=None,
    )
    mock_agent_openai.client.responses.create = Mock(side_effect=bad_request)

    search_client = OpenAIAgent(
        tools=[],
        model="openai/gpt-5",
    )

    with pytest.raises(BadRequestError):
        search_client.chat(inputs=[])

    assert mock_agent_openai.client.responses.create.call_count == 1


def search_products(query: str, top_k: int = 5) -> List[Dict]:
    """
    Search for furniture products.

    This is direct keyword search, no synonyms, only BM25 scoring on product name and description and
    basic snowball tokenization of query and document.

    Args:
        query: The search query string.
        top_k: The number of top results to return.

    Returns:
        A list of dictionaries containing product information.
    """
    print("Searching for:", query, "top_k:", top_k)
    query_tokens = snowball_tokenizer(query)
    scores = np.zeros(len(products))
    for token in query_tokens:
        scores += products["product_name_snowball"].array.score(token) * 10
        scores += products["description_snowball"].array.score(token)

    top_k_indices = np.argsort(scores)[-top_k:][::-1]
    scores = scores[top_k_indices]
    top_products = products.iloc[top_k_indices]
    top_products["score"] = scores

    # Serialize back in JSON
    try:
        results = []
        print("Getting results")
        for id, row in top_products.iterrows():
            results.append(
                {
                    "id": id,
                    "product_name": row["product_name"],
                    "product_description": row["product_description"],
                    "score": row["score"],
                }
            )

        return results
    except Exception as e:
        print("!!!")
        print("Error serializing results:", e)
        raise e


def alt_search_products(query: str, top_k: int = 5) -> List[Dict]:
    """
    Search for furniture products

    Args:
        query: The search query string.
        top_k: The number of top results to return.

    Returns:
        A list of dictionaries containing product information.
    """
    print("Searching for:", query, "top_k:", top_k)
    query_tokens = snowball_tokenizer(query)
    scores = np.zeros(len(products))
    for token in query_tokens:
        scores += products["product_name_snowball"].array.score(token) * 10

    top_k_indices = np.argsort(scores)[-top_k:][::-1]
    scores = scores[top_k_indices]
    top_products = products.iloc[top_k_indices]
    top_products["score"] = scores

    # Serialize back in JSON
    try:
        results = []
        print("Getting results")
        for id, row in top_products.iterrows():
            results.append(
                {
                    "id": id,
                    "product_name": row["product_name"],
                    "product_description": row["product_description"],
                    "score": row["score"],
                }
            )

        return results
    except Exception as e:
        print("!!!")
        print("Error serializing results:", e)
        raise e


class SearchInteraction(BaseModel):
    user_query: str = Field(..., description="The original user search query")
    search_tool_name: str = Field(..., description="The name of the search tool used")
    search_tool_query: str = Field(
        ..., description="The actual search query sent to the search tool"
    )
    quality: Literal["good", "meh", "bad"] = Field(
        ..., description="The quality of the results returned by the search tool"
    )
    reasoning: str = Field(..., description="The reasoning for the quality rating")


saved_search_interactions = {}


def save_queries_used(search_interactions: List[SearchInteraction]) -> None:
    """Store how you used tools and the quality of their results (so you can later retrieve for future occurrences of this user query)

    Args:
        search_interactions: A list of SearchInteraction objects representing the interactions to save.

    """
    saved_queries = list(saved_search_interactions.keys())
    for interaction in search_interactions:
        if interaction.user_query not in search_interactions:
            saved_search_interactions[interaction.user_query] = []
        saved_search_interactions[interaction.user_query].append(interaction)
    print("Saved interactions for queries:", saved_queries)


def get_past_queries(original_user_query: str) -> List[SearchInteraction]:
    """Get the past queries used for a given user query.

    Args:
        original_user_query: The original user search query the user sent you.
    """
    if original_user_query in saved_search_interactions:
        return saved_search_interactions[original_user_query]
    return []


@patch("cheat_at_search.agent.openai_agent.OpenAI")
@patch("cheat_at_search.agent.openai_agent.key_for_provider")
def test_calling_search_tool(mock_key_for_provider, mock_openai):
    # thread, public_url = serve_tools(fns=[search_products])
    # time.sleep(1)
    mock_agent_openai = configure_mock_openai(mock_key_for_provider, mock_openai)

    mock_agent_openai.queue_parse_response(
        output=[
            function_call(
                "search_products",
                {"query": "oversized sofa", "top_k": 5},
            )
        ]
    )
    mock_agent_openai.queue_parse_response(
        payload={
            "results": [
                {"id": "0", "rank": 1},
                {"id": "1", "rank": 2},
            ]
        }
    )

    search_client = OpenAIAgent(
        tools=[search_products],
        model="openai/gpt-5",
        response_model=SearchResults,
    )
    prompt = """
        Reason carefully to find furniture products that match the following description, returning top 10 best results.

        It's OK to repeatedly try different queries until you find the best ones.

        a couch for my really big butt

    """
    inputs = [
        {
            "role": "system",
            "content": "You are a helpful assistant that helps people find furniture products.",
        },
        {"role": "user", "content": prompt},
    ]
    resp, final_inputs, usage = search_client.chat(inputs=inputs, return_usage=True)
    results = resp.output_parsed
    assert [result.id for result in results.results] == ["0", "1"]
    assert usage["num_tool_calls"] == 1
    assert final_inputs[-1]["type"] == "function_call_output"
    assert "product_name" in final_inputs[-1]["output"]
    mock_openai.assert_called_once_with(api_key="test-openai-key")
    assert mock_agent_openai.client.responses.parse.call_count == 2
    first_parse_kwargs = mock_agent_openai.client.responses.parse.call_args_list[0].kwargs
    assert first_parse_kwargs["model"] == "gpt-5"
    assert first_parse_kwargs["text_format"] is SearchResults
    assert first_parse_kwargs["tools"][0]["name"] == "search_products"


class PreferredSearchTool(BaseModel):
    tool_name: Literal[
        "search_products",
        "alt_search_products",
    ] = Field(..., description="The function name of the preferred search tool")
    reason: str = Field(..., description="The reason for preferring this search tool")


@patch("cheat_at_search.agent.openai_agent.OpenAI")
@patch("cheat_at_search.agent.openai_agent.key_for_provider")
def test_analyze_best_search_backend(mock_key_for_provider, mock_openai):
    mock_agent_openai = configure_mock_openai(mock_key_for_provider, mock_openai)
    mock_agent_openai.queue_parse_response(
        payload={
            "tool_name": "search_products",
            "reason": "The product description search is more useful for this request.",
        }
    )

    system_prompt = """
        You are a helpful assistant that analyzes the best search tool for finding furniture products.
    """

    prompt = """
        Reason carefully to find furniture products that match the following description, returning top 10 best results using the provided tool.

        It's OK to repeatedly try different queries.

        Based on your experience, return which tool made it easier to find this query:

        a couch for my really big butt

    """

    search_client = OpenAIAgent(
        tools=[search_products, alt_search_products],
        model="openai/gpt-5",
        response_model=PreferredSearchTool,
    )
    inputs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    resp, _, _ = search_client.chat(inputs=inputs, return_usage=True)
    preferred_tool = resp.output_parsed
    assert preferred_tool.tool_name in ["search_products", "alt_search_products"]
    parse_kwargs = mock_agent_openai.client.responses.parse.call_args.kwargs
    assert parse_kwargs["text_format"] is PreferredSearchTool
    assert [tool["name"] for tool in parse_kwargs["tools"]] == [
        "search_products",
        "alt_search_products",
    ]


@patch("cheat_at_search.agent.openai_agent.OpenAI")
@patch("cheat_at_search.agent.openai_agent.key_for_provider")
def test_reasoning_search_strategy(mock_key_for_provider, mock_openai):
    mock_agent_openai = configure_mock_openai(mock_key_for_provider, mock_openai)
    system_prompt = """
        You are a helpful assistant that helps people find furniture products.
    """

    prompt = """
        Reason carefully to find furniture products that match the following description, returning top 10 best results.

        It's OK to repeatedly try different queries until you find the best ones.

    """

    queries = [
        "a couch for my really big butt",
        "a small chair for my tiny apartment",
        "a bed for a kid who loves space",
        "a table for a fancy dinner party",
        "a lamp for reading at night",
    ]
    for _query in queries:
        mock_agent_openai.queue_parse_response(
            payload={
                "results": [
                    {"id": str(idx), "rank": idx + 1}
                    for idx in range(5)
                ]
            }
        )

    search_client = OpenAIAgent(
        tools=[search_products],
        model="openai/gpt-5",
        response_model=SearchResults,
    )
    harness = Harness(search_client)
    strategy = ReasoningSearchStrategy(
        products,
        harness=harness,
        prompt=prompt,
        system_prompt=system_prompt,
        cache=False,
    )
    for query in queries:
        top_k, scores = strategy.search(query, k=5)
        assert len(top_k) == 5
        assert len(scores) == 5
    assert mock_agent_openai.client.responses.parse.call_count == len(queries)


@patch("cheat_at_search.agent.openai_agent.OpenAI")
@patch("cheat_at_search.agent.openai_agent.key_for_provider")
def test_agent_tool_error_is_returned_as_string(mock_key_for_provider, mock_openai):
    mock_agent_openai = configure_mock_openai(mock_key_for_provider, mock_openai)

    def failing_tool(query: str, top_k: int = 5) -> list[dict]:
        """Always fails for testing."""
        raise RuntimeError("tool failure")

    mock_agent_openai.queue_parse_response(
        output=[
            function_call(
                "failing_tool",
                {"query": "oversized sofa", "top_k": 5},
            )
        ]
    )
    mock_agent_openai.queue_parse_response(
        payload={
            "results": [
                {"id": "0", "rank": 1},
                {"id": "1", "rank": 2},
            ]
        }
    )

    search_client = OpenAIAgent(
        tools=[failing_tool],
        model="openai/gpt-5",
        response_model=SearchResults,
    )

    inputs = [
        {
            "role": "system",
            "content": "You are a helpful assistant that helps people find furniture products.",
        },
        {"role": "user", "content": "Find a couch"},
    ]

    resp, final_inputs, usage = search_client.chat(inputs=inputs, return_usage=True)

    assert [result.id for result in resp.output_parsed.results] == ["0", "1"]
    assert usage["num_tool_calls"] == 1
    assert final_inputs[-1]["type"] == "function_call_output"
    assert "tool failure" in final_inputs[-1]["output"]
