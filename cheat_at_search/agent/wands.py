from cheat_at_search.wands_data import (
    enriched_products,
    queries as wands_queries,
    labeled_query_products,
    judgments,
)
from cheat_at_search.agent.strategy import ReasoningSearchStrategy
from cheat_at_search.strategy.strategy import SearchStrategy
from cheat_at_search.agent.openai_agent import OpenAIAgent
from cheat_at_search.search import run_strategy
from cheat_at_search.strategy import BM25Search, BestPossibleResults
from cheat_at_search.agent.history import save_queries, get_past_queries, index
from cheat_at_search.agent.judgments import make_judgments_tool
from cheat_at_search.tokenizers import snowball_tokenizer
from typing import List, Dict, Literal, Optional
import argparse
from searcharray import SearchArray
import numpy as np
import pandas as pd


enriched_products["title_snowball"] = SearchArray.index(
    enriched_products["title"], tokenizer=snowball_tokenizer
)

enriched_products["description_snowball"] = SearchArray.index(
    enriched_products["description"], tokenizer=snowball_tokenizer
)


enriched_products["category_snowball"] = SearchArray.index(
    enriched_products["category"], tokenizer=snowball_tokenizer
)


Categories = Literal[
    "Furniture",
    "Kitchen & Tabletop",
    "Browse By Brand",
    "Home Improvement",
    "Décor & Pillows",
    "Outdoor",
    "Storage & Organization",
    "Bed & Bath",
    "Baby & Kids",
    "Pet",
    "Lighting",
    "Rugs",
    "School Furniture and Supplies",
    "Commercial Business Furniture",
    "Holiday Décor",
    "Fountains",
    "Contractor",
    "Appliances",
    "Sale",
    "Reception Area",
    "Foodservice",
    "Institutional Furniture Parts & Accessories",
    "Landscaping Screens & Bridges",
    "Shop Product Type",
    "Clips",
    "Slicers, Peelers And Graters",
    "Bed Accessories",
    "Accommodations",
    "Buffet Accessories",
    "Specialty Serving",
    "Display Cases",
    "Key Organizers",
    "Ergonomic Accessories",
    "Slow Cookers",
    "Bath Rugs & Mats",
    "Furniture Cushions",
    "Early Education",
    "Learning Resources",
    "Physical Education Equipment",
    "Faux Plants and Trees",
    "Desk Parts",
    "Serving Dishes & Platters",
    "Water Filter Pitchers",
    "Shower Curtain Rods",
    "Table Accessories",
    "Sandboxes & Sand Toys",
    "Meeting & Collaborative Spaces",
    "Desktop Organizers & Desk Pads",
    "Napkin Rings, Place Card Holders & Food Markers",
    "Partition & Panel Hardware Accessories",
    "Cash Handling",
    "Hooks",
    "Novelty Lighting",
    "Protection Plans",
    "Stages, Risers and Accessories",
]


def search_products_keywords_cat(
    keywords: str, category: Optional[Categories] = None, top_k: int = 5
) -> List[Dict]:
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
        scores += enriched_products["title_snowball"].array.score(token) * 10
        scores += enriched_products["description_snowball"].array.score(token)

    # Filter by category
    if category:
        print("Filtering by category:", category)
        cat_tokenized = snowball_tokenizer(category)
        category_mask = (
            enriched_products["category_snowball"].array.score(cat_tokenized) > 0
        )
        scores = scores * category_mask

    top_k_indices = np.argsort(scores)[-top_k:][::-1]
    scores = scores[top_k_indices]
    top_products = enriched_products.iloc[top_k_indices].copy()
    top_products.loc[:, "score"] = scores

    results = []

    for id, row in top_products.iterrows():
        results.append(
            {
                "id": id,
                "title": row["title"],
                "description": row["description"],
                "category": row["category"],
                "score": row["score"],
            }
        )
    print(f"Keywords {keywords} -- Found {len(results)} results")
    return results


search_products_keyword_cat = search_products_keywords_cat


def search_products_keywords(keywords: str, top_k: int = 5) -> List[Dict]:
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
        scores += enriched_products["title_snowball"].array.score(token) * 10
        scores += enriched_products["description_snowball"].array.score(token)

    top_k_indices = np.argsort(scores)[-top_k:][::-1]
    scores = scores[top_k_indices]
    top_products = enriched_products.iloc[top_k_indices].copy()
    top_products.loc[:, "score"] = scores

    results = []

    for id, row in top_products.iterrows():
        results.append(
            {
                "id": id,
                "title": row["title"],
                "description": row["description"],
                "category": row["category"],
                "score": row["score"],
            }
        )
    print(f"Keywords {keywords} -- Found {len(results)} results")
    return results


search_products_keyword = search_products_keywords


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

system_prompt_vanilla = """
    You take user search queries and use a search tool to find furniture products.

    Use only the search tool available to you. Formulate straightforward keyword queries
    and return results per the SearchResults schema, ranked best to worst.

    Gather results until you have 10 best matches you can find. It's important to return at least 10.
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


search_few_shot_hist_prompt = """
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


search_few_shot_prompt = """
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

search_few_shot_judgments_prompt = """
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


def agent_search_wands(
    use_old=True,
    prompt=system_no_judgments_prompt,
    prompt_builder=None,
    model="openai/gpt-5",
    search_tool=search_products_keyword_cat,
    search_tool_supports_category=True,
    iterations=5,
    num_queries=5,
    num_workers=4,
    addl_tools=None,
    seed=42,
    num_seeds=1,
):
    ndcgs_by_seed = []
    bm25_by_seed = []
    for curr_seed in range(seed, seed + num_seeds):
        if not use_old:
            global saved_search_interactions, saved_queries, query_embeddings
            saved_search_interactions = {}
            saved_queries = np.array([])
            query_embeddings = np.array([])

        shuffled_queries = wands_queries.sample(frac=1, random_state=curr_seed)
        queries = shuffled_queries[:num_queries]
        print(f"QUERIES: {queries}")

        # Get best possible
        best_possible = BestPossibleResults(enriched_products, judgments)
        graded_best_possible = run_strategy(
            best_possible, judgments, num_queries=num_queries, seed=curr_seed
        )
        best_possible_ndcg = graded_best_possible["ndcg"].mean()
        print(f"Best Possible NDCG: {best_possible_ndcg}")

        # Run BM25 baseline
        bm25 = BM25Search(enriched_products)
        graded_bm25 = run_strategy(
            bm25, judgments, num_queries=num_queries, seed=curr_seed
        )
        bm25_ndcg = graded_bm25["ndcg"].mean()
        print(f"Baseline NDCG: {bm25_ndcg}")

        tools = [search_tool]
        if addl_tools:
            tools.extend(addl_tools)

        prompt_for_seed = prompt
        if prompt_builder is not None:
            prompt_for_seed = prompt_builder(curr_seed)

        search_client = OpenAIAgent(
            tools=tools, model=model, system_prompt=prompt_for_seed
        )
        strategy = ReasoningSearchStrategy(
            enriched_products,
            search_client,
            prompt="",
            cache=iterations == 1,
            workers=num_workers,
        )
        ndcgs = []
        for iter in range(iterations):
            print(f"--- Iteration {iter + 1} of {iterations} ---")
            graded_results = run_strategy(
                strategy, judgments, num_queries=num_queries, seed=curr_seed
            )
            ndcg = graded_results["ndcg"].mean()
            print(f"BM25 Baseline NDCG: {bm25_ndcg}")
            print(f"Overall NDCG: {ndcg}")
            ndcgs.append(ndcg)

            # Now index past interactions to get the benefit on next iteration
            saved_queries, query_embeddings = index()

        print(f"Ideal NDCG: {best_possible_ndcg}")
        print(f"Baseline NDCG: {bm25_ndcg}")
        for idx, ndcg in enumerate(ndcgs):
            print(f"Iteration {idx + 1}: NDCG {ndcg}")
        final_ndcg = ndcgs[-1] if ndcgs else 0
        ndcgs_by_seed.append((curr_seed, final_ndcg))
        bm25_by_seed.append((curr_seed, bm25_ndcg))

    bm25_by_seed_map = dict(bm25_by_seed)
    for seed_value, ndcg in ndcgs_by_seed:
        bm25_ndcg = bm25_by_seed_map.get(seed_value, 0)
        print(f"Seed {seed_value}: NDCG {ndcg} (BM25 NDCG {bm25_ndcg})")


class PostAgentStrategy(SearchStrategy):
    """Use what worked well in the past for tools to retrieve relevant results."""

    def __init__(self, products, search_tool, search_tool_supports_category=True):
        super().__init__(products)
        self.search_tool = search_tool
        self.search_tool_supports_category = search_tool_supports_category

    def search(self, query, k=10):
        past_queries = get_past_queries(query)

        all_results = []

        for past_query in past_queries:
            tool_query = past_query.interaction.search_tool_query
            tool_category = past_query.interaction.search_tool_category
            if past_query.interaction.quality == "good":
                print(f"Reusing good query: {tool_query}, category: {tool_category}")
                if self.search_tool_supports_category:
                    results = self.search_tool(
                        tool_query, category=tool_category, top_k=k
                    )
                else:
                    results = self.search_tool(tool_query, top_k=k)
                all_results.extend(results)


def build_few_shot_prompt(k=10, prompt=search_few_shot_hist_prompt, seed=42) -> str:
    labeled_query_products.sample(5, random_state=seed)

    labeled = labeled_query_products
    if len(labeled) == 0:
        return []
    relevant = labeled[labeled["label"] == "Exact"]
    irrelevant = labeled[labeled["label"] == "Irrelevant"]
    # Get 3 relevant
    relevant = relevant.sample(min(k // 3, len(relevant)), random_state=seed)
    # Get 3 irrelevant
    irrelevant = irrelevant.sample(min(k // 3, len(irrelevant)), random_state=seed)
    # Get the rest Partial
    partial = labeled[labeled["label"] == "Partial"]
    partial = partial.sample(
        min(k - len(relevant) - len(irrelevant), len(partial)), random_state=seed
    )

    # Format into prompt
    labeled = pd.concat([relevant, irrelevant, partial]).sample(
        frac=1, random_state=seed
    )
    for item in labeled.to_dict(orient="records"):
        print(item)
        prompt += f"""

        User Query: {item["query"]}
        Product Name: {item["title"]}
        Product Description: {item["description"]}
        Product Category: {item["category"]}
        Human Label: {item["label"]}

        """
    print("Prompt is:")
    print(prompt)
    return prompt


def resolve_search_tool(name):
    if name == "keywords_cat":
        return search_products_keyword_cat, True
    if name == "keywords":
        return search_products_keyword, False
    raise ValueError(f"Unknown search tool: {name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run WANDS agent search experiments.")
    parser.add_argument(
        "mode",
        choices=[
            "post_agent_search",
            "search_hist_no_judgments",
            "search_with_hist_judgments",
            "search_vanilla",
            "search_few_shot_hist",
            "search_few_shot",
            "search_few_shot_judgments",
        ],
        help="Experiment mode to run.",
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--model", type=str, default="openai/gpt-5")
    parser.add_argument(
        "--search-tool",
        choices=["keywords_cat", "keywords"],
        default="keywords_cat",
    )
    args = parser.parse_args(argv)

    search_tool, search_tool_supports_category = resolve_search_tool(args.search_tool)

    if args.mode == "post_agent_search":
        strategy = PostAgentStrategy(
            enriched_products, search_tool, search_tool_supports_category
        )
        graded_results = run_strategy(strategy, wands_queries[:20])
        ndcg = graded_results["ndcg"].mean()
        print(f"Overall NDCG: {ndcg}")
    if args.mode == "search_hist_no_judgments":
        agent_search_wands(
            use_old=False,
            iterations=args.iterations,
            num_queries=args.num_queries,
            addl_tools=[save_queries, get_past_queries],
            prompt=system_no_judgments_prompt,
            model=args.model,
            search_tool=search_tool,
            search_tool_supports_category=search_tool_supports_category,
            seed=args.seed,
            num_seeds=args.num_seeds,
        )
    elif args.mode == "search_with_hist_judgments":
        agent_search_wands(
            use_old=False,
            iterations=args.iterations,
            num_queries=args.num_queries,
            addl_tools=[
                make_judgments_tool(labeled_query_products),
                save_queries,
                get_past_queries,
            ],
            prompt=system_prompt_judgments,
            model=args.model,
            search_tool=search_tool,
            search_tool_supports_category=search_tool_supports_category,
            seed=args.seed,
            num_seeds=args.num_seeds,
        )
    elif args.mode == "search_vanilla":
        agent_search_wands(
            use_old=False,
            iterations=args.iterations,
            num_queries=args.num_queries,
            prompt=system_prompt_vanilla,
            model=args.model,
            search_tool=search_tool,
            search_tool_supports_category=search_tool_supports_category,
            seed=args.seed,
            num_seeds=args.num_seeds,
        )
    elif args.mode == "search_few_shot_hist":
        agent_search_wands(
            use_old=False,
            iterations=args.iterations,
            num_queries=args.num_queries,
            addl_tools=[save_queries, get_past_queries],
            prompt_builder=lambda curr_seed: build_few_shot_prompt(
                10, prompt=search_few_shot_hist_prompt, seed=curr_seed
            ),
            model=args.model,
            search_tool=search_tool,
            search_tool_supports_category=search_tool_supports_category,
            seed=args.seed,
            num_seeds=args.num_seeds,
        )
    elif args.mode == "search_few_shot":
        agent_search_wands(
            use_old=False,
            iterations=args.iterations,
            num_queries=args.num_queries,
            prompt_builder=lambda curr_seed: build_few_shot_prompt(
                10, prompt=search_few_shot_prompt, seed=curr_seed
            ),
            model=args.model,
            search_tool=search_tool,
            search_tool_supports_category=search_tool_supports_category,
            seed=args.seed,
            num_seeds=args.num_seeds,
        )
    elif args.mode == "search_few_shot_judgments":
        agent_search_wands(
            use_old=False,
            iterations=args.iterations,
            num_queries=args.num_queries,
            addl_tools=[make_judgments_tool(labeled_query_products)],
            prompt_builder=lambda curr_seed: build_few_shot_prompt(
                10,
                prompt=search_few_shot_judgments_prompt,
                seed=curr_seed,
            ),
            model=args.model,
            search_tool=search_tool,
            search_tool_supports_category=search_tool_supports_category,
            seed=args.seed,
            num_seeds=args.num_seeds,
        )


if __name__ == "__main__":
    main()
