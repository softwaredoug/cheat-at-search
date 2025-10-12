from cheat_at_search.esci_data import corpus, judgments
from cheat_at_search.strategy.strategy import SearchStrategy
from cheat_at_search.agent.strategy import ReasoningSearchStrategy
from cheat_at_search.agent.openai_search_client import OpenAISearchClient
from cheat_at_search.search import run_strategy
from cheat_at_search.data_dir import ensure_data_subdir
from cheat_at_search.strategy import BM25Search, BestPossibleResults
from cheat_at_search.tokenizers import snowball_tokenizer
from typing import List, Dict, Optional, Literal
from searcharray import SearchArray
from pydantic import BaseModel, Field
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
                field_to_search: Literal['product_name', 'product_description'],
                operator: Literal['and', 'or'],
                locale: Literal['es', 'us', 'jp'] = 'us',
                top_k: int = 5) -> List[Dict]:
    """
    Retrieve a BM25 score for Amazon search results by searching the specific field with the given keywords

    This is direct / naive BM25 keyword search with simple snowball stemming and whitespace tokenization.
    Do not expect synonyms, compounting, decompounding, query understanding, or other NLP tricks.

    Instead YOU need to reason about the user's intent and reformulate the query given the constraints.

    Args:
        keywords: The search query string.
        field_to_search: The field to search in. Options are 'product_name' and 'product_description'.
        operator: The logical operator to combine search terms. Options are 'and' and 'or'. Use and to require all terms.
        locale: The locale to search in. Default is 'us'. Other options are 'es' and 'jp'.
                Consider the language of the query when choosing the locale.
        top_k: The number of top results to return.

    Returns:
        Search results as a list of dictionaries with 'id', 'title', 'description', and 'score' keys.

    """
    query_tokens = snowball_tokenizer(keywords)
    scores = np.zeros(len(corpus))
    if field_to_search == 'product_name':
        field_name = 'title_snowball'
    elif field_to_search == 'product_description':
        field_name = 'description_snowball'
    else:
        raise ValueError("field_to_search must be 'product_name' or 'product_description'")

    for token in query_tokens:
        scores += corpus[field_name].array.score(token)

    if operator == 'and':
        for token in query_tokens:
            require_mask = (corpus[field_name].array.score(token) > 0)
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
    print(f"Keywords {keywords} field: {field_to_search} operator: {operator} locale: {locale} -> {len(results)} results")
    return results


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
    Generate python code using 'search_esci' as a function (the tool here). Assume search_esci has the python
    signature as defined in tools.

    Issue test queries using search_esci to see how it works, what its limitations are, etc before formulating
    your plan. Create novel queries to test the tool, see how well it works, and iterate to improve your code and
    reranker.

    Then generate a *generalized* reranker as a function of search_esci, call this function 'rerank_esci', it should
    return a list of product IDs in the order you think best matches the query. It should take as a parameter the 'search_esci'
    tool to help (as the tool will be injected from outside).

    It's very important you consider carefully the correct ranking as you'll be evaluated on
    how close that is to the average shoppers ideal ranking.

    Examples of labeled query-product pairs are listed at the bottom of this prompt to help you
    understand how we will evaluate your results.

    Your goal is to generate a ranker that doesn't need to have specific query terms mentioned in the code,
    but can figure out the intent of the query and use the search tool to find the best products. You can use general, broad categories,
    stopwords, concepts, etc, but think beyond the queries you're presented

    In other words, overfitting is bad!!!

    Generate valid python. Double check your code.

    Finally, some examples:
"""


class GeneratedRerankerCode(BaseModel):
    """Python code that would best rerank search results."""
    query: str = Field(..., description="The original user search query")
    code: str = Field(..., description="Python code that would best rerank search results")
    code_explanation: str = Field(..., description="Explanation of why the code will generalize beyond the examples")


def build_few_shot_prompt(num_queries=10, num_per_query=10,
                          prompt=system_few_shot_prompt,
                          seed=42) -> str:
    if len(judgments) == 0:
        return []
    queries = judgments[['query', 'query_id']].drop_duplicates()
    queries = queries.sample(num_queries, random_state=seed)
    for query in queries['query']:
        query_judgments = judgments[judgments['query'] == query]
        relevant = query_judgments[query_judgments['grade'] == 3]
        irrelevant = query_judgments[query_judgments['grade'] == 0]
        # Get 3 relevant
        relevant = relevant.sample(min(num_per_query // 3, len(relevant)), random_state=seed)
        # Get 3 irrelevant
        irrelevant = irrelevant.sample(min(num_per_query // 3, len(irrelevant)), random_state=seed)
        # Get the rest Partial
        partial = judgments[judgments['grade'].isin([1, 2])]
        partial = partial.sample(min(num_per_query - len(relevant) - len(irrelevant), len(partial)), random_state=seed)

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


class CodeGenSearchStrategy(SearchStrategy):
    def __init__(self, corpus, cache=True,
                 workers=1):
        super().__init__(corpus, workers=workers)
        self.index = corpus

    def search(self, query, k=10):

        from rerank_esci import rerank_esci
        product_ids = rerank_esci(search_esci, query)
        scores = np.arange(len(product_ids), 0, -1)
        top_k = product_ids[:k]
        scores = scores[:k]
        return top_k, scores


if __name__ == "__main__":
    prompt = build_few_shot_prompt()
    num_queries = 100
    bm25 = BM25Search(corpus)
    graded_bm25 = run_strategy(bm25, judgments, num_queries=num_queries)
    bm25_ndcg = graded_bm25.groupby('query')['ndcg'].mean().mean()
    print(f"Baseline NDCG: {bm25_ndcg}")
    # best = BestPossibleResults(corpus, judgments)
    # graded_best = run_strategy(best, judgments, num_queries=num_queries)
    # best_ndcg = graded_best['ndcg'].mean()
    # print(f"Best Possible NDCG: {best_ndcg}")
    tools = [search_esci]

    search_client = OpenAISearchClient(tools=tools,
                                       model="openai/gpt-5",
                                       system_prompt=prompt,
                                       response_model=GeneratedRerankerCode)
    resp = search_client.search(prompt="")
    code = resp.code
    with open("rerank_esci.py", "w") as f:
        f.write(code)

    codegen_strategy = CodeGenSearchStrategy(corpus, workers=16)
    results_codegen = run_strategy(codegen_strategy, judgments, num_queries=num_queries)
    ndcg = results_codegen.groupby('query')['ndcg'].mean().mean()
    print(f"Baseline NDCG: {bm25_ndcg}")
    print(f"Codegen NDCG: {ndcg}")
