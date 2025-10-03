"""Tools for tracking search tool usage history."""
from cheat_at_search.data_dir import ensure_data_subdir
from pydantic import BaseModel, Field
import pickle
from typing import Literal, Optional, List
from sentence_transformers import SentenceTransformer
import numpy as np


class SearchInteraction(BaseModel):
    user_query: str = Field(..., description="The original user search query")
    search_tool_name: str = Field(..., description="The name of the search tool used")
    search_tool_query: str = Field(..., description="The actual search keywords sent to the search tool")
    search_tool_category: Optional[str] = Field(None, description="The category filter sent to the search tool, if any")
    quality: Literal['good', 'meh', 'bad'] = Field(..., description="The quality of the results returned by the search tool")
    reasoning: str = Field(..., description="The reasoning for the quality rating")


class PastQueriesResponse(BaseModel):
    interaction: SearchInteraction = Field(..., description="A list of past search interactions for the given user query")
    similarity_score: float = Field(..., description="A similarity score between the original user query and the past interaction's user query")


cached_interactions_dir = ensure_data_subdir("cached_interactions")

saved_search_interactions = {}

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def index():
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


saved_queries = np.array([])
query_embeddings = np.array([])
try:
    with open(cached_interactions_dir / "saved_search_interactions.pkl", "rb") as f:
        saved_search_interactions = pickle.load(f)
        saved_queries, query_embeddings = index()
except FileNotFoundError:
    print("No saved interactions file found, starting fresh.")


def save_queries(search_tool_interactions: list[SearchInteraction]) -> None:
    """
    Store all queries sent to the search backend, including the user query, the tool query used, and the quality of results.

    Args:
        search_tool_interactions: The interaction details to save.

    """
    for search_tool_interaction in search_tool_interactions:
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
    if len(saved_search_interactions) == 0:
        print("No saved interactions, returning empty.")
        return []
    print("Getting past queries for:", original_user_query)
    threshold = 0.8
    embedded = model.encode(original_user_query)
    embedded /= np.linalg.norm(embedded)
    try:
        if embedded.shape != (query_embeddings.shape[1],):
            print("Embedding shape mismatch, returning empty.")
            return []
    except IndexError:
        return []
    sims = np.dot(query_embeddings, embedded)
    above_thresh = np.where(sims > threshold)[0]
    matched_queries = saved_queries[above_thresh]
    sims = sims[above_thresh]

    past_queries_resp: List[PastQueriesResponse] = []
    for query, sim in zip(matched_queries, sims):
        for interaction in saved_search_interactions[query]:
            print(f"Matched query: {query}, similarity: {sim}, interaction: {interaction}")
            past_queries_resp.append(PastQueriesResponse(interaction=interaction,
                                                         similarity_score=float(sim)))
    return past_queries_resp
