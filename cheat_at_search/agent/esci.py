from cheat_at_search.esci_data import corpus, judgments
from cheat_at_search.agent.strategy import ReasoningSearchStrategy
from cheat_at_search.agent.openai_search_client import OpenAISearchClient
from cheat_at_search.search import run_strategy
from cheat_at_search.data_dir import ensure_data_subdir
from cheat_at_search.strategy import BM25Search, BestPossibleResults
from cheat_at_search.tokenizers import snowball_tokenizer
from typing import List, Dict, Optional, Literal
from searcharray import SearchArray
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


corpus_dir = ensure_data_subdir("esci_indexed_corpus")
model = SentenceTransformer('all-MiniLM-L6-v2')


def resolve(query_item: str, embeddings_to_search, names_lookup, k=5) -> str:
    query_embedding = model.encode([query_item])[0]
    similarities = np.dot(embeddings_to_search, query_embedding) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(embeddings_to_search, axis=1) + 1e-10)
    top_k = np.argsort(-similarities)[:k]
    return names_lookup[top_k], similarities[top_k]


def resolve_then_filter(query_item: str, embeddings_to_search,
                        names_lookup, corpus, field_name, k=5) -> List[str]:
    resolved_items, similarities = resolve(query_item, embeddings_to_search, names_lookup, k=k)
    matches = np.zeros(len(corpus), dtype=bool)
    for resolved_item in resolved_items:
        tokenized = snowball_tokenizer(resolved_item)
        matches |= (corpus[field_name].array.score(tokenized) > 0)
    return matches


try:
    corpus = pd.read_pickle(corpus_dir / "corpus.pkl")  # noqa
except FileNotFoundError:
    corpus['brand_snowball'] = SearchArray.index(corpus['product_brand'].fillna(''), snowball_tokenizer)
    corpus['product_color_snowball'] = SearchArray.index(corpus['product_color'].fillna(''), snowball_tokenizer)
    corpus['title_snowball'] = SearchArray.index(corpus['title'], snowball_tokenizer)
    corpus['description_snowball'] = SearchArray.index(corpus['description'], snowball_tokenizer)
    corpus.to_pickle(corpus_dir / "corpus.pkl")


def search_esci(keywords: str,
                locale: Literal['es', 'us', 'jp'] = 'us',
                top_k: int = 5) -> List[Dict]:
    """
    Search amazon products with the given keywords and filters

    This is direct / naive BM25 keyword search with simple snowball stemming and whitespace tokenization.
    Do not expect synonyms, compounting, decompounding, query understanding, or other NLP tricks.

    Instead YOU need to reason about the user's intent and reformulate the query given the constraints.

    Args:
        keywords: The search query string.
        locale: The locale to search in. Default is 'us'. Other options are 'es' and 'jp'.
                Consider the language of the query when choosing the locale.
        top_k: The number of top results to return.

    Returns:
        Search results as a list of dictionaries with 'id', 'title', 'description', and 'score' keys.

    """
    excluded = None
    required = None
    product_color = None
    brand_name = None

    query_tokens = snowball_tokenizer(keywords)
    scores = np.zeros(len(corpus))
    for token in query_tokens:
        scores += corpus['title_snowball'].array.score(token) * 2
        scores += corpus['description_snowball'].array.score(token)

    if excluded is None:
        excluded = []

    if required is None:
        required = []

    for phrase in excluded:
        exclude_tokens = snowball_tokenizer(phrase)
        exclude_mask = (corpus['title_snowball'].array.score(exclude_tokens) > 0) | \
                       (corpus['description_snowball'].array.score(exclude_tokens) > 0)
        scores = scores * (~exclude_mask)

    for phrase in required:
        require_tokens = snowball_tokenizer(phrase)
        require_mask = (corpus['title_snowball'].array.score(require_tokens) > 0) | \
                       (corpus['description_snowball'].array.score(require_tokens) > 0)
        scores = scores * require_mask

    if locale:
        locale_filter = (corpus['product_locale'] == locale)
        scores = scores * locale_filter

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
        })
    print(f"Keywords {keywords} required: {required} excluded: {excluded} color:{product_color} brand:{brand_name} locale:{locale}-- Found {len(results)} results")
    return results


def multi_search_esci_results(keywords: list[str],
                              locale: Literal['es', 'us', 'jp'] = 'us',
                              top_k: int = 5) -> List[Dict]:
    """Gather search results for multiple keywords."""
    all_results = []
    for keyword in keywords:
        results = search_esci(keyword, locale=locale, top_k=top_k)
        for result in results:
            result['query'] = keyword
        all_results.extend(results)
    return all_results


def inspect_product(product_id: str) -> Optional[Dict]:
    """Inspect a product by its ID."""
    print(f"Inspecting product {product_id}")
    product = corpus[corpus['product_id'] == product_id]
    if len(product) == 0:
        return None
    product = product.iloc[0]
    return {
        'id': product['product_id'],
        'title': product['title'],
        'description': product['description'],
        'color': product['product_color'],
        'brand': product['product_brand'],
        'locale': product['product_locale']
    }


system_few_shot_prompt = """
    You take user search queries and use a search tool to find products on the Amazon online store. Examples
    of labeled query-product pairs are listed at the bottom of this prompt to help you
    understand how we will evaluate your results.

    Look at the search tools you have, their limitations, how they work, etc when forming your plan. Exercise the different
    parameters of the search tools, review how well they worked, and iterate to improve.

    If you get "token alert!" in a user message, that's a sign to wrap up tool calling and return results.

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
        Title: {item['title']}
        Human Label: {item['label']}

        """
    print("Prompt is:")
    print(prompt)
    return prompt


if __name__ == "__main__":
    prompt = build_few_shot_prompt(k=10)
    num_queries = 10
    bm25 = BM25Search(corpus)
    graded_bm25 = run_strategy(bm25, judgments, num_queries=num_queries)
    bm25_ndcg = graded_bm25.groupby('query')['ndcg'].mean().mean()
    print(f"Baseline NDCG: {bm25_ndcg}")
    best = BestPossibleResults(corpus, judgments)
    graded_best = run_strategy(best, judgments, num_queries=num_queries)
    # best_ndcg = graded_best['ndcg'].mean()
    # print(f"Best Possible NDCG: {best_ndcg}")
    tools = [multi_search_esci_results]

    search_client = OpenAISearchClient(tools=tools,
                                       model="openai/gpt-5",
                                       system_prompt=prompt)
    strategy = ReasoningSearchStrategy(corpus, search_client,
                                       prompt="",
                                       cache=False,
                                       workers=1)
    print(f"Avg tokens per query: {strategy.total_tokens / num_queries}")
    graded_agent = run_strategy(strategy, judgments, num_queries=num_queries)
    print(f"Agent NDCG: {graded_agent.groupby('query')['ndcg'].mean().mean()}")

    sxs = graded_best.merge(
        graded_agent,
        on=['query', 'query_id', 'rank'],
        how='left',
        suffixes=('_best', '_agent')
    )
    sxs = sxs[['query', 'product_id_best', 'product_title_best',
               'product_color_best', 'product_brand_best',
               'grade_best',
               'product_id_agent',
               'product_color_agent', 'product_brand_agent',
               'ndcg_agent', 'product_title_agent', 'grade_agent']][sxs['ndcg_agent'] < 0.1]
    print(sxs)
