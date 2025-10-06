from cheat_at_search.esci_data import corpus, judgments
from cheat_at_search.agent.strategy import ReasoningSearchStrategy
from cheat_at_search.agent.openai_search_client import OpenAISearchClient
from cheat_at_search.search import run_strategy
from cheat_at_search.strategy import BM25Search, BestPossibleResults
from cheat_at_search.tokenizers import snowball_tokenizer
from typing import List, Dict
from searcharray import SearchArray
import numpy as np
import pandas as pd


corpus['title_snowball'] = SearchArray.index(corpus['title'], snowball_tokenizer)
corpus['description_snowball'] = SearchArray.index(corpus['description'], snowball_tokenizer)


def search_esci(keywords: str,
                top_k: int = 5) -> List[Dict]:
    """
    Search amazon products with the given keywords and filters

    This is direct / naive BM25 keyword search with simple snowball stemming and whitespace tokenization.
    Do not expect synonyms, compounting, decompounding, query understanding, or other NLP tricks.

    Instead YOU need to reason about the user's intent and reformulate the query given the constraints.

    Args:
        keywords: The search query string.
        top_k: The number of top results to return.

    Returns:
        Search results as a list of dictionaries with 'id', 'title', 'description', and 'score' keys.

    """
    query_tokens = snowball_tokenizer(keywords)
    scores = np.zeros(len(corpus))
    for token in query_tokens:
        scores += corpus['title_snowball'].array.score(token) * 2
        scores += corpus['description_snowball'].array.score(token)

    top_k_indices = np.argsort(scores)[-top_k:][::-1]
    scores = scores[top_k_indices]
    top_products = corpus.iloc[top_k_indices].copy()
    top_products.loc[:, 'score'] = scores

    results = []

    for id, row in top_products.iterrows():
        results.append({
            'id': id,
            'title': row['title'],
            'description': row['description'],
            'score': row['score']
        })
    print(f"Keywords {keywords} -- Found {len(results)} results")
    return results


system_few_shot_prompt = """
    You take user search queries and use a search tool to find products on the Amazon online store. Examples
    of labeled query-product pairs are listed at the bottom of this prompt to help you
    understand how we will evaluate your results.

    Look at the search tools you have, their limitations, how they work, etc when forming your plan.

    Finally return results to the user per the SearchResults schema, ranked best to worst.

    Gather results until you have 10 best matches you can find. It's important to return at least 10.

    It's very important you consider carefully the correct ranking as you'll be evaluated on
    how close that is to the average shoppers ideal ranking.

    Finally, some examples:
"""


def build_few_shot_prompt(k=10, prompt=system_few_shot_prompt,
                          seed=42) -> str:
    if len(judgments) == 0:
        return []
    relevant = judgments[judgments['grade'] == 3]
    irrelevant = judgments[judgments['grade'] == 0]
    # Get 3 relevant
    relevant = relevant.sample(min(k // 3, len(relevant)), random_state=seed)
    # Get 3 irrelevant
    irrelevant = irrelevant.sample(min(k // 3, len(irrelevant)), random_state=seed)
    # Get the rest Partial
    partial = judgments[judgments['grade'].isin([1, 2])]
    partial = partial.sample(min(k - len(relevant) - len(irrelevant), len(partial)), random_state=seed)

    # Format into prompt
    labeled = pd.concat([relevant, irrelevant, partial]).sample(frac=1, random_state=seed)
    labeled = labeled.merge(corpus, on='product_id', how='left', suffixes=('', '_y'))
    for item in labeled.to_dict(orient='records'):
        prompt += f"""

        User Query: {item['query']}
        Product Name: {item['title']}
        Product Description: {item['description']}
        Human Label: {item['label']}

        """
    print("Prompt is:")
    print(prompt)
    return prompt


if __name__ == "__main__":
    prompt = build_few_shot_prompt(k=10)
    num_queries = 100
    bm25 = BM25Search(corpus)
    graded_bm25 = run_strategy(bm25, judgments, num_queries=num_queries)
    bm25_ndcg = graded_bm25['ndcg'].mean()
    print(f"Baseline NDCG: {bm25_ndcg}")
    best = BestPossibleResults(corpus, judgments)
    graded_best = run_strategy(best, judgments, num_queries=num_queries)
    best_ndcg = graded_best['ndcg'].mean()
    print(f"Best Possible NDCG: {best_ndcg}")
    tools = [search_esci]

    search_client = OpenAISearchClient(tools=tools,
                                       model="openai/gpt-5",
                                       system_prompt=prompt)
    strategy = ReasoningSearchStrategy(corpus, search_client,
                                       prompt="",
                                       cache=True,
                                       workers=4)
    graded_agent = run_strategy(strategy, judgments, num_queries=num_queries)
    print(f"Agent NDCG: {graded_agent['ndcg'].mean()}")
