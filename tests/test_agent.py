from cheat_at_search.data_dir import key_for_provider
from cheat_at_search.wands_data import products
from cheat_at_search.agent.openai_agent import OpenAIAgent
from cheat_at_search.agent.strategy import ReasoningSearchStrategy
from cheat_at_search.tokenizers import snowball_tokenizer
from typing import List, Dict, Literal
from openai import OpenAI
from searcharray import SearchArray
import numpy as np
from pydantic import BaseModel, Field


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


class SearchInteraction(BaseModel):
    user_query: str = Field(..., description="The original user search query")
    search_tool_name: str = Field(..., description="The name of the search tool used")
    search_tool_query: str = Field(..., description="The actual search query sent to the search tool")
    quality: Literal['good', 'meh', 'bad'] = Field(..., description="The quality of the results returned by the search tool")
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


def test_calling_search_tool():
    # thread, public_url = serve_tools(fns=[search_products])
    # time.sleep(1)

    search_client = OpenAIAgent(tools=[search_products], model="openai/gpt-5",
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
    system_prompt = """
        You are a helpful assistant that analyzes the best search tool for finding furniture products.
    """

    prompt = """
        Reason carefully to find furniture products that match the following description, returning top 10 best results using the provided tool.

        It's OK to repeatedly try different queries.

        Based on your experience, return which tool made it easier to find this query:

        a couch for my really big butt

    """

    search_client = OpenAIAgent(tools=[search_products, alt_search_products],
                                model="openai/gpt-5",
                                system_prompt=system_prompt,
                                response_model=PreferredSearchTool)
    preferred_tool = search_client.search(prompt)
    assert preferred_tool.tool_name in ["search_products", "alt_search_products"]


def test_reasoning_search_strategy():
    system_prompt = """
        You are a helpful assistant that helps people find furniture products.
    """

    prompt = """
        Reason carefully to find furniture products that match the following description, returning top 10 best results.

        It's OK to repeatedly try different queries until you find the best ones.

    """

    search_client = OpenAIAgent(tools=[search_products],
                                model="openai/gpt-5",
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
