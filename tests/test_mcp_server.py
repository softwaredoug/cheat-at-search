from cheat_at_search.data_dir import key_for_provider
from cheat_at_search.wands_data import products
from cheat_at_search.agent.mcp import serve_tools
from cheat_at_search.agent.openai_search_client import OpenAISearchClient, OpenAIChatAdapter
from cheat_at_search.agent.search_client import ReasoningSearchStrategy
from cheat_at_search.tokenizers import snowball_tokenizer
from typing import List, Dict, Literal
from openai import OpenAI
from searcharray import SearchArray
import numpy as np
from pydantic import BaseModel, Field
import time
import pytest


openai_key = key_for_provider("openai")

client = OpenAI(
    api_key=openai_key,
)

products['product_name_snowball'] = SearchArray.index(products['product_name'],
                                                      tokenizer=snowball_tokenizer)

products['description_snowball'] = SearchArray.index(products['product_description'],
                                                     tokenizer=snowball_tokenizer)


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
        scores += products['product_name_snowball'].array.score(token) * 10
        scores += products['description_snowball'].array.score(token)

    top_k_indices = np.argsort(scores)[-top_k:][::-1]
    scores = scores[top_k_indices]
    top_products = products.iloc[top_k_indices]
    top_products['score'] = scores

    # Serialize back in JSON
    try:
        results = []
        print("Getting results")
        for id, row in top_products.iterrows():
            results.append({
                'id': id,
                'product_name': row['product_name'],
                'product_description': row['product_description'],
                'score': row['score']
            })

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
        scores += products['product_name_snowball'].array.score(token) * 10

    top_k_indices = np.argsort(scores)[-top_k:][::-1]
    scores = scores[top_k_indices]
    top_products = products.iloc[top_k_indices]
    top_products['score'] = scores

    # Serialize back in JSON
    try:
        results = []
        print("Getting results")
        for id, row in top_products.iterrows():
            results.append({
                'id': id,
                'product_name': row['product_name'],
                'product_description': row['product_description'],
                'score': row['score']
            })

        return results
    except Exception as e:
        print("!!!")
        print("Error serializing results:", e)
        raise e


def test_serving_search_tool():
    thread, _ = serve_tools(fns=[search_products],
                            port=8000)
    time.sleep(1)
    thread.join(1)


def test_calling_search_tool():
    thread, public_url = serve_tools(fns=[search_products],
                                     port=8000)
    time.sleep(1)

    search_client = OpenAISearchClient(mcp_url=public_url, model="openai/gpt-5",
                                       system_prompt="You are a helpful assistant that helps people find furniture products.")
    prompt = """
        Reason carefully to find furniture products that match the following description, returning top 10 best results.

        It's OK to repeatedly try different queries until you find the best ones.

        a couch for my really big butt

    """
    results = search_client.search(prompt)
    assert len(results.results) > 0


class PreferredSearchTool(BaseModel):
    tool_name : Literal[
        'search_products',
        'alt_search_products',
    ] = Field(..., description="The function name of the preferred search tool")
    reason: str = Field(..., description="The reason for preferring this search tool")


def test_analyze_best_search_backend():
    thread, public_url = serve_tools(fns=[search_products, alt_search_products],
                                     port=8000)
    time.sleep(1)

    system_prompt = """
        You are a helpful assistant that analyzes the best search tool for finding furniture products.
    """

    prompt = """
        Reason carefully to find furniture products that match the following description, returning top 10 best results using the provided tool.

        It's OK to repeatedly try different queries.

        Based on your experience, return which tool made it easier to find this query:

        a couch for my really big butt

    """

    search_client = OpenAISearchClient(mcp_url=public_url, model="openai/gpt-5",
                                       system_prompt=system_prompt,
                                       response_model=PreferredSearchTool)
    preferred_tool = search_client.search(prompt)
    assert preferred_tool.tool_name in ["search_products", "alt_search_products"]


def test_reasoning_search_strategy():
    thread, public_url = serve_tools(fns=[search_products],
                                     port=8000)
    time.sleep(1)

    system_prompt = """
        You are a helpful assistant that helps people find furniture products.
    """

    prompt = """
        Reason carefully to find furniture products that match the following description, returning top 10 best results.

        It's OK to repeatedly try different queries until you find the best ones.

    """

    search_client = OpenAISearchClient(mcp_url=public_url, model="openai/gpt-5",
                                       system_prompt=system_prompt)
    strategy = ReasoningSearchStrategy(products, search_client=search_client, prompt=prompt)
    queries = [
        "a couch for my really big butt",
        "a small chair for my tiny apartment",
        "a bed for a kid who loves space",
        "a table for a fancy dinner party",
        "a lamp for reading at night"
    ]
    for query in queries:
        top_k, scores = strategy.search(query, k=5)
        assert len(top_k) == 5
        assert len(scores) == 5


# @pytest.mark.skip(reason="Interactive test")
def test_interactive_chat():
    thread, public_url = serve_tools(fns=[search_products, alt_search_products],
                                     port=8000)
    time.sleep(1)

    system_prompt = """
        You are an interactive chat client that can search furniture products for the user using the provided tools. You can also introspect
        on the search backends answering any of the user's search queries.

        Some messages will be short e-commerce search queries, so you'll need to reason about their intent and what might make them happy.

        Other messages will be conversational messages about the search backends and how you used them.

    """

    search_client = OpenAISearchClient(mcp_url=public_url, model="openai/gpt-5",
                                       system_prompt=system_prompt)
    chat_adapter = OpenAIChatAdapter(search_client)

    while True:
        message = input("User: ")
        if message in ['reset']:
            chat_adapter.reset()
            print("Chat reset.")
            continue
        if message in ['exit', 'quit']:
            break
        response = chat_adapter.chat(message)
        print("Assistant:", response)
