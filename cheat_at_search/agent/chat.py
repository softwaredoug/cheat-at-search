from cheat_at_search.wands_data import products
from cheat_at_search.agent.mcp import serve_tools
from cheat_at_search.agent.openai_search_client import OpenAISearchClient, OpenAIChatAdapter
from cheat_at_search.data_dir import ensure_data_subdir
from cheat_at_search.tokenizers import snowball_tokenizer
from typing import List, Dict, Literal
from searcharray import SearchArray
import numpy as np
from pydantic import BaseModel, Field
import time
import pickle

from sentence_transformers import SentenceTransformer


class SearchInteraction(BaseModel):
    user_query: str = Field(..., description="The original user search query")
    search_tool_name: str = Field(..., description="The name of the search tool used")
    search_tool_query: str = Field(..., description="The actual search query sent to the search tool")
    quality: Literal['good', 'meh', 'bad'] = Field(..., description="The quality of the results returned by the search tool")
    reasoning: str = Field(..., description="The reasoning for the quality rating")


class PastQueriesResponse(BaseModel):
    interaction: SearchInteraction = Field(..., description="A list of past search interactions for the given user query")
    similarity_score: float = Field(..., description="A similarity score between the original user query and the past interaction's user query")


cached_interactions_dir = ensure_data_subdir("cached_interactions")

saved_search_interactions = {}

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def _index_search_queries():
    embeds = []
    unique_queries = set()
    for query, interactions in saved_search_interactions.items():
        print(f"Query: {query}")
        unique_queries.add(query)
        for interaction in interactions:
            print(f"  Tool: {interaction.search_tool_name}, Tool Query: {interaction.search_tool_query}, Quality: {interaction.quality}, Reasoning: {interaction.reasoning}")

    unique_queries = list(unique_queries)
    for query in unique_queries:
        embed = model.encode(query)
        embed /= np.linalg.norm(embed)
        embeds.append(embed)
    embeds = np.array(embeds)
    unique_queries = np.array(unique_queries)
    return unique_queries, embeds


try:
    with open(cached_interactions_dir / "saved_search_interactions.pkl", "rb") as f:
        saved_search_interactions = pickle.load(f)
        queries, query_embeddings = _index_search_queries()
except FileNotFoundError:
    print("No saved interactions file found, starting fresh.")


def save_interaction(search_tool_interaction: SearchInteraction) -> None:
    """
    Store a single interaction with a search backend, including the user query, the tool query used, and the quality of results.

    Args:
        search_tool_interactions: The interaction details to save.

    """
    if search_tool_interaction.user_query not in saved_search_interactions:
        saved_search_interactions[search_tool_interaction.user_query] = []
    else:
        saved_search_interactions[search_tool_interaction.user_query].append(search_tool_interaction)
    print("Saved interaction:", search_tool_interaction)
    with open(cached_interactions_dir / "saved_search_interactions.pkl", "wb") as f:
        pickle.dump(saved_search_interactions, f)


def get_past_queries(original_user_query: str) -> List[PastQueriesResponse]:
    """Get the past queries used for a given user query.

    Args:
        original_user_query: The original user search query the user sent you.
    """
    print("Getting past queries for:", original_user_query)
    threshold = 0.8
    embedded = model.encode(original_user_query)
    embedded /= np.linalg.norm(embedded)
    sims = np.dot(query_embeddings, embedded)
    above_thresh = np.where(sims > threshold)[0]
    matched_queries = queries[above_thresh]
    sims = sims[above_thresh]

    past_queries_resp: List[PastQueriesResponse] = []
    for query, sim in zip(matched_queries, sims):
        for interaction in saved_search_interactions[query]:
            print(f"Matched query: {query}, similarity: {sim}, interaction: {interaction}")
            past_queries_resp.append(PastQueriesResponse(interaction=interaction,
                                                         similarity_score=float(sim)))
    return past_queries_resp


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


def chat():
    thread, public_url = serve_tools(fns=[search_products,
                                          save_interaction, get_past_queries])
    time.sleep(1)

    system_prompt = """
        You take user search queries and use a search tool to find furniture products.

        Look at the search tools you have, their limitations, how they work, etc when forming your plan.

        Before searching you MUST use the "get_past_queries" to get similar, past queries
        the user has made

        Remember every tool usage you make. After searching with a tool, evaluate the results,
        then save the interaction (immediately after tool usage) with the "save_queries_used" tool
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


if __name__ == "__main__":
    chat()
