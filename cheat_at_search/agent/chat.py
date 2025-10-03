from cheat_at_search.wands_data import enriched_products, queries as wands_queries, labeled_query_products
from cheat_at_search.agent.strategy import ReasoningSearchStrategy
from cheat_at_search.strategy.strategy import SearchStrategy
from cheat_at_search.agent.openai_search_client import OpenAISearchClient, OpenAIChatAdapter
from cheat_at_search.data_dir import ensure_data_subdir
from cheat_at_search.search import run_strategy
from cheat_at_search.strategy import BM25Search
from cheat_at_search.tokenizers import snowball_tokenizer
from typing import List, Dict, Literal, Optional
from searcharray import SearchArray
import numpy as np
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import sys
from time import perf_counter

from sentence_transformers import SentenceTransformer


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


saved_queries = np.array([])
query_embeddings = np.array([])
try:
    with open(cached_interactions_dir / "saved_search_interactions.pkl", "rb") as f:
        saved_search_interactions = pickle.load(f)
        saved_queries, query_embeddings = _index_search_queries()
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


class HumanEvaluation(BaseModel):
    """A human judgment of a search result for a given user query."""
    user_query: str = Field(..., description="The original user search query")
    human_label: str = Field(..., description="The human judgment label for the search results")
    product_name: str = Field(..., description="The name of the product judged")
    product_description: str = Field(..., description="The description of the product judged")


def get_human_judgments(user_query: str) -> List[HumanEvaluation]:
    """Get a sample of human judgments for a given user query (the ground truth you're evaluated against).

       It's ok to use this, its not cheating!

       Returns list of human evaluations
    """
    K = 10
    labeled = labeled_query_products.loc[labeled_query_products['query'] == user_query]
    if len(labeled) == 0:
        return []
    relevant = labeled[labeled['label'] == 'Exact']
    irrelevant = labeled[labeled['label'] == 'Irrelevant']
    # Get 3 relevant
    relevant = relevant.sample(min(3, len(relevant)), random_state=42)
    # Get 3 irrelevant
    irrelevant = irrelevant.sample(min(3, len(irrelevant)), random_state=42)
    # Get the rest Partial
    partial = labeled[labeled['label'] == 'Partial']
    partial = partial.sample(min(K - len(relevant) - len(irrelevant), len(partial)), random_state=42)

    labeled = pd.concat([relevant, irrelevant, partial]).sample(frac=1, random_state=42)

    results: List[HumanEvaluation] = []
    for item in labeled.to_dict(orient='records'):
        results.append(HumanEvaluation(user_query=user_query,
                                       human_label=item['label'],
                                       product_name=item['product_name'],
                                       product_description=item['product_description']))

    return results


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


enriched_products['product_name_snowball'] = SearchArray.index(enriched_products['product_name'],
                                                               tokenizer=snowball_tokenizer)

enriched_products['description_snowball'] = SearchArray.index(enriched_products['product_description'],
                                                              tokenizer=snowball_tokenizer)


enriched_products['category_snowball'] = SearchArray.index(enriched_products['category'],
                                                           tokenizer=snowball_tokenizer)


Categories = Literal['Furniture', 'Kitchen & Tabletop', 'Browse By Brand',
                     'Home Improvement', 'Décor & Pillows', 'Outdoor',
                     'Storage & Organization', 'Bed & Bath', 'Baby & Kids',
                     'Pet', 'Lighting', 'Rugs', 'School Furniture and Supplies',
                     'Commercial Business Furniture', 'Holiday Décor', 'Fountains',
                     'Contractor', 'Appliances', 'Sale', 'Reception Area',
                     'Foodservice', 'Institutional Furniture Parts & Accessories',
                     'Landscaping Screens & Bridges', 'Shop Product Type', 'Clips',
                     'Slicers, Peelers And Graters', 'Bed Accessories',
                     'Accommodations', 'Buffet Accessories', 'Specialty Serving',
                     'Display Cases', 'Key Organizers', 'Ergonomic Accessories',
                     'Slow Cookers', 'Bath Rugs & Mats', 'Furniture Cushions',
                     'Early Education', 'Learning Resources',
                     'Physical Education Equipment', 'Faux Plants and Trees',
                     'Desk Parts', 'Serving Dishes & Platters', 'Water Filter Pitchers',
                     'Shower Curtain Rods', 'Table Accessories',
                     'Sandboxes & Sand Toys', 'Meeting & Collaborative Spaces',
                     'Desktop Organizers & Desk Pads',
                     'Napkin Rings, Place Card Holders & Food Markers',
                     'Partition & Panel Hardware Accessories', 'Cash Handling', 'Hooks',
                     'Novelty Lighting', 'Protection Plans',
                     'Stages, Risers and Accessories']


def search_products(keywords: str,
                    category: Optional[Categories] = None,
                    top_k: int = 5) -> List[Dict]:
    """
    Search for furniture products with the given keywords and filters

    This is direct keyword search along with optional category filtering.

    Args:
        keywords: The search query string.
        category: category to filter products by.
        top_k: The number of top results to return.

    Returns:
        Search results as a list of dictionaries with 'id', 'product_name', 'product_description', and 'score' keys.

    """
    print("Searching for:", keywords, "top_k:", top_k)
    query_tokens = snowball_tokenizer(keywords)
    scores = np.zeros(len(enriched_products))
    for token in query_tokens:
        scores += enriched_products['product_name_snowball'].array.score(token) * 10
        scores += enriched_products['description_snowball'].array.score(token)

    # Filter by category
    if category:
        print("Filtering by category:", category)
        cat_tokenized = snowball_tokenizer(category)
        category_mask = enriched_products['category_snowball'].array.score(cat_tokenized) > 0
        scores = scores * category_mask

    top_k_indices = np.argsort(scores)[-top_k:][::-1]
    scores = scores[top_k_indices]
    top_products = enriched_products.iloc[top_k_indices].copy()
    top_products.loc[:, 'score'] = scores

    results = []

    for id, row in top_products.iterrows():
        results.append({
            'id': id,
            'product_name': row['product_name'],
            'product_description': row['product_description'],
            'category': row['category'],
            'score': row['score']
        })
    print(f"Keywords {keywords} -- Found {len(results)} results")
    return results


def chat():
    system_prompt = """
        You take user's search query, think of different queries, and use batch search tool to find furniture products.

        Search for furniture products using the following steps:

        1. Look at the search tool you have, its limitations, how it work, etc when forming your plan.

        2. Before searching, use the "get_human_judgments" tool to get the ground truth human judgments for this user query. If anything shows up, use that interpret user intent and evaluate relevance of results you find.

        3. Before searching, use the "get_past_queries" to get similar, past queries the user has made to
        gain insight on how to best search for this user query using the available tool

        4. Issue searches in one call to "search_products" tool.

        5. Evaluate the results you get back from the search tool.

        6. Save the results quality of each query (immediately after "search_products" usage) with the "save_queries" tool

        7. Iterate as needed, reformulating queries, until you have enough good results to present to the user.

        8. Present the best results to the user, citing the product name and description.

        Outside of searches, respond to questions about your behavior (in these cases, you should not use a tool).
    """

    search_client = OpenAISearchClient(tools=[search_products, save_queries, get_past_queries,
                                              get_human_judgments],
                                       model="openai/gpt-5",
                                       system_prompt=system_prompt,
                                       response_model=None)
    chat_adapter = OpenAIChatAdapter(search_client)

    while True:
        message = input("User: ")
        if message in ['reset']:
            chat_adapter.reset()
            print("Chat reset.")
            continue
        if message in ['exit', 'quit']:
            break
        begin = perf_counter()
        response = chat_adapter.chat(message)
        took = perf_counter() - begin
        print(f"(took {took:.2f}s)")
        print("Assistant:", response)


system_no_judgments_prompt = """
    You take user search queries and use a search tool to find furniture products.

    Look at the search tools you have, their limitations, how they work, etc when forming your plan.

    Before searching you MUST use the "get_past_queries" to get similar, past queries
    you have made to tools and whether they were successful. This should help you plan how to
    use tools to satisfy user intent.

    Remember every tool usage you make. After searching with a tool, evaluate the results,
    then save the interaction (immediately after tool usage) with the "save_queries_used" tool

    Finally return results to the user per the SearchResults schema, ranked best to worst.

    Gather results until you have 10 best matches you can find. It's important to return at least 10.

    It's very important you consider carefully the correct ranking as you'll be evaluated on
    how close that is to the average furniture shoppers ideal ranking.
"""

system_prompt_judgments = """
    You take user search queries and use a search tool to find furniture products.

    Look at the search tools you have, their limitations, how they work, etc when forming your plan.

    Before searching you MUST use the "get_past_queries" to get similar, past queries
    you have made to tools and whether they were successful. This should help you plan how to
    use tools to satisfy user intent.

    Before searching you MUST use the "get_human_judgments" tool to get a few human evaluations
    for this query. If any are found, use that to evaluate the relevance of results you find,
    as user expectations and intent may be different than what you expect.

    Remember every tool usage you make. After searching with a tool, evaluate the results,
    then save the interaction (immediately after tool usage) with the "save_queries_used" tool

    Finally return results to the user per the SearchResults schema, ranked best to worst.

    Gather results until you have 10 best matches you can find. It's important to return at least 10.

    It's very important you consider carefully the correct ranking as you'll be evaluated on
    how close that is to the average furniture shoppers ideal ranking.
"""


system_few_shot_prompt = """
    You take user search queries and use a search tool to find furniture products. Examples
    of labeled query-product pairs are listed at the bottom of this prompt to help you
    understand how we will evaluate your results.

    Look at the search tools you have, their limitations, how they work, etc when forming your plan.

    Before searching you MUST use the "get_past_queries" to get similar, past queries
    you have made to tools and whether they were successful. This should help you plan how to
    use tools to satisfy user intent.

    Remember every tool usage you make. After searching with a tool, evaluate the results,
    then save the interaction (immediately after tool usage) with the "save_queries_used" tool

    Finally return results to the user per the SearchResults schema, ranked best to worst.

    Gather results until you have 10 best matches you can find. It's important to return at least 10.

    It's very important you consider carefully the correct ranking as you'll be evaluated on
    how close that is to the average furniture shoppers ideal ranking.

    Finally, some examples:
"""


def agent_search_wands(use_old=True,
                       prompt=system_no_judgments_prompt,
                       iterations=5,
                       num_queries=20,
                       addl_tools=None,
                       seed=42):
    if not use_old:
        global saved_search_interactions, saved_queries, query_embeddings
        saved_search_interactions = {}
        saved_queries = np.array([])
        query_embeddings = np.array([])

    shuffled_queries = wands_queries.sample(frac=1,
                                            random_state=seed)
    queries = shuffled_queries[:num_queries]
    print(f"QUERIES: {queries}")

    # Run BM25 baseline
    bm25 = BM25Search(enriched_products)
    graded_bm25 = run_strategy(bm25, queries)
    bm25_ndcg = graded_bm25['ndcg'].mean()
    print(f"Baseline NDCG: {bm25_ndcg}")

    tools = [search_products, save_queries, get_past_queries]
    if addl_tools:
        tools.extend(addl_tools)

    search_client = OpenAISearchClient(tools=tools,
                                       model="openai/gpt-5",
                                       system_prompt=prompt)
    strategy = ReasoningSearchStrategy(enriched_products, search_client,
                                       prompt="",
                                       cache=False)
    ndcgs = []
    for iter in range(iterations):
        print(f"--- Iteration {iter + 1} of {iterations} ---")
        graded_results = run_strategy(strategy, queries)
        ndcg = graded_results['ndcg'].mean()
        print(f"BM25 Baseline NDCG: {bm25_ndcg}")
        print(f"Overall NDCG: {ndcg}")
        ndcgs.append(ndcg)

        # Now index past interactions to get the benefit on next iteration
        saved_queries, query_embeddings = _index_search_queries()

    print(f"Baseline NDCG: {bm25_ndcg}")
    for idx, ndcg in enumerate(ndcgs):
        print(f"Iteration {idx + 1}: NDCG {ndcg}")


class PostAgentStrategy(SearchStrategy):
    """Use what worked well in the past for tools to retrieve relevant results."""
    def __init__(self, products):
        super().__init__(products)

    def search(self, query, k=10):
        past_queries = get_past_queries(query)

        all_results = []

        for past_query in past_queries:
            tool_query = past_query.interaction.search_tool_query
            tool_category = past_query.interaction.search_tool_category
            if past_query.interaction.quality == 'good':
                print(f"Reusing good query: {tool_query}, category: {tool_category}")
                results = search_products(tool_query, category=tool_category, top_k=k)
                all_results.extend(results)


def build_few_shot_prompt(k=10) -> str:
    labeled_query_products.sample(5, random_state=42)

    labeled = labeled_query_products
    if len(labeled) == 0:
        return []
    relevant = labeled[labeled['label'] == 'Exact']
    irrelevant = labeled[labeled['label'] == 'Irrelevant']
    # Get 3 relevant
    relevant = relevant.sample(min(k // 3, len(relevant)), random_state=42)
    # Get 3 irrelevant
    irrelevant = irrelevant.sample(min(k // 3, len(irrelevant)), random_state=42)
    # Get the rest Partial
    partial = labeled[labeled['label'] == 'Partial']
    partial = partial.sample(min(k - len(relevant) - len(irrelevant), len(partial)), random_state=42)

    # Format into prompt
    labeled = pd.concat([relevant, irrelevant, partial]).sample(frac=1, random_state=42)
    prompt = system_few_shot_prompt
    for item in labeled.to_dict(orient='records'):
        print(item)
        prompt += f"""

        User Query: {item['query']}
        Product Name: {item['product_name']}
        Product Description: {item['product_description']}
        Product Category: {item['category']}
        Human Label: {item['label']}

        """
    print("Prompt is:")
    print(prompt)
    return prompt


if __name__ == "__main__":
    seed = 43
    if sys.argv[-1] == "post_agent_search":
        strategy = PostAgentStrategy(enriched_products)
        graded_results = run_strategy(strategy, wands_queries[:20])
        ndcg = graded_results['ndcg'].mean()
        print(f"Overall NDCG: {ndcg}")
    if sys.argv[-1] == "search_no_judgments":
        agent_search_wands(use_old=False, iterations=3,
                           num_queries=10,
                           prompt=system_no_judgments_prompt,
                           seed=seed)
    elif sys.argv[-1] == "search_with_judgments":
        agent_search_wands(use_old=False, iterations=3,
                           num_queries=10,
                           addl_tools=[get_human_judgments],
                           prompt=system_prompt_judgments,
                           seed=seed)
    elif sys.argv[-1] == "search_few_shot":
        agent_search_wands(use_old=False,
                           iterations=1,
                           num_queries=30,
                           prompt=build_few_shot_prompt(10),
                           seed=seed)
    else:
        chat()
