from cheat_at_search.wands_data import enriched_products, queries as wands_queries, labeled_query_products, judgments
from cheat_at_search.agent.strategy import ReasoningSearchStrategy
from cheat_at_search.strategy.strategy import SearchStrategy
from cheat_at_search.agent.openai_search_client import OpenAISearchClient, OpenAIChatAdapter
from cheat_at_search.search import run_strategy
from cheat_at_search.strategy import BM25Search, BestPossibleResults
from cheat_at_search.agent.history import save_queries, get_past_queries, index
from cheat_at_search.agent.judgments import make_judgments_tool
from cheat_at_search.tokenizers import snowball_tokenizer
from typing import List, Dict, Literal, Optional
from searcharray import SearchArray
import numpy as np
import pandas as pd
import sys
from time import perf_counter


enriched_products['title_snowball'] = SearchArray.index(enriched_products['title'],
                                                        tokenizer=snowball_tokenizer)

enriched_products['description_snowball'] = SearchArray.index(enriched_products['description'],
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
        Search results as a list of dictionaries with 'id', 'title', 'description', and 'score' keys.

    """
    print("Searching for:", keywords, "top_k:", top_k)
    query_tokens = snowball_tokenizer(keywords)
    scores = np.zeros(len(enriched_products))
    for token in query_tokens:
        scores += enriched_products['title_snowball'].array.score(token) * 10
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
            'title': row['title'],
            'description': row['description'],
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

    search_client = OpenAISearchClient(tools=[search_products, save_queries, get_past_queries],
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


system_few_shot_no_history_prompt = """
    You take user search queries and use a search tool to find furniture products. Examples
    of labeled query-product pairs are listed at the bottom of this prompt to help you
    understand how we will evaluate your results.

    Look at the search tools you have, their limitations, how they work, etc when forming your plan.

    Finally return results to the user per the SearchResults schema, ranked best to worst.

    Gather results until you have 10 best matches you can find. It's important to return at least 10.

    It's very important you consider carefully the correct ranking as you'll be evaluated on
    how close that is to the average furniture shoppers ideal ranking.

    Finally, some examples:
"""

system_few_shot_judgmens_no_history_prompt = """
    You take user search queries and use a search tool to find furniture products. Examples
    of labeled query-product pairs are listed at the bottom of this prompt to help you
    understand how we will evaluate your results.

    Before searching you MUST use the "get_human_judgments" tool to get a few human evaluations
    for this query. If any are found, use that to evaluate the relevance of results you find,
    as user expectations and intent may be different than what you expect.

    Look at the search tools you have, their limitations, how they work, etc when forming your plan.

    Finally return results to the user per the SearchResults schema, ranked best to worst.

    Gather results until you have 10 best matches you can find. It's important to return at least 10.

    It's very important you consider carefully the correct ranking as you'll be evaluated on
    how close that is to the average furniture shoppers ideal ranking.

    It's very important to rank as close to the human judgments as possible, with those results labeled 'Exact'
    should be ranked highest. Partial is a mediocre result. Irrelevant should be avoided.

    Ordering Exact above Partial above Irrelevant is what you're evaluated against

    Finally, some general examples:
"""


def agent_search_wands(use_old=True,
                       prompt=system_no_judgments_prompt,
                       iterations=5,
                       num_queries=5,
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

    # Get best possible
    best_possible = BestPossibleResults(enriched_products, judgments)
    graded_best_possible = run_strategy(best_possible, judgments, num_queries=num_queries, seed=seed)
    best_possible_ndcg = graded_best_possible['ndcg'].mean()
    print(f"Best Possible NDCG: {best_possible_ndcg}")

    # Run BM25 baseline
    bm25 = BM25Search(enriched_products)
    graded_bm25 = run_strategy(bm25, judgments, num_queries=num_queries, seed=seed)
    bm25_ndcg = graded_bm25['ndcg'].mean()
    print(f"Baseline NDCG: {bm25_ndcg}")

    tools = [search_products]
    if addl_tools:
        tools.extend(addl_tools)

    search_client = OpenAISearchClient(tools=tools,
                                       model="openai/gpt-5",
                                       system_prompt=prompt)
    strategy = ReasoningSearchStrategy(enriched_products, search_client,
                                       prompt="",
                                       cache=iterations == 1)
    ndcgs = []
    for iter in range(iterations):
        print(f"--- Iteration {iter + 1} of {iterations} ---")
        graded_results = run_strategy(strategy, judgments,
                                      num_queries=num_queries,
                                      seed=seed)
        ndcg = graded_results['ndcg'].mean()
        print(f"BM25 Baseline NDCG: {bm25_ndcg}")
        print(f"Overall NDCG: {ndcg}")
        ndcgs.append(ndcg)

        # Now index past interactions to get the benefit on next iteration
        saved_queries, query_embeddings = index()

    print(f"Ideal NDCG: {best_possible_ndcg}")
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


def build_few_shot_prompt(k=10, prompt=system_few_shot_prompt) -> str:
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
    for item in labeled.to_dict(orient='records'):
        print(item)
        prompt += f"""

        User Query: {item['query']}
        Product Name: {item['title']}
        Product Description: {item['description']}
        Product Category: {item['category']}
        Human Label: {item['label']}

        """
    print("Prompt is:")
    print(prompt)
    return prompt


if __name__ == "__main__":
    seed = 43
    num_queries = 100
    iterations = 1

    if sys.argv[-1] == "post_agent_search":
        strategy = PostAgentStrategy(enriched_products)
        graded_results = run_strategy(strategy, wands_queries[:20])
        ndcg = graded_results['ndcg'].mean()
        print(f"Overall NDCG: {ndcg}")
    if sys.argv[-1] == "search_hist_no_judgments":
        agent_search_wands(use_old=False,
                           iterations=iterations,
                           num_queries=num_queries,
                           addl_tools=[save_queries,
                                       get_past_queries],
                           prompt=system_no_judgments_prompt,
                           seed=seed)
    elif sys.argv[-1] == "search_with_hist_judgments":
        agent_search_wands(use_old=False,
                           iterations=iterations,
                           num_queries=num_queries,
                           addl_tools=[make_judgments_tool(labeled_query_products),
                                       save_queries,
                                       get_past_queries],
                           prompt=system_prompt_judgments,
                           seed=seed)
    elif sys.argv[-1] == "search_few_shot_hist":
        agent_search_wands(use_old=False,
                           iterations=iterations,
                           num_queries=num_queries,
                           addl_tools=[save_queries,
                                       get_past_queries],
                           prompt=build_few_shot_prompt(10, prompt=system_few_shot_prompt),
                           seed=seed)
    elif sys.argv[-1] == "search_few_shot":
        agent_search_wands(use_old=False,
                           iterations=iterations,
                           num_queries=num_queries,
                           prompt=build_few_shot_prompt(10, prompt=system_few_shot_no_history_prompt),
                           seed=seed)
    elif sys.argv[-1] == "search_few_shot_judgments":
        agent_search_wands(use_old=False,
                           iterations=iterations,
                           num_queries=num_queries,
                           addl_tools=[make_judgments_tool(labeled_query_products)],
                           prompt=build_few_shot_prompt(10, prompt=system_few_shot_judgmens_no_history_prompt),
                           seed=seed)
    else:
        chat()
