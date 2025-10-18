from cheat_at_search.esci_data import corpus, judgments
from cheat_at_search.agent.openai_agent import OpenAIAgent
from cheat_at_search.search import run_strategy
from cheat_at_search.logger import log_at
from cheat_at_search.data_dir import ensure_data_subdir
from cheat_at_search.strategy import BM25Search
from cheat_at_search.tokenizers import snowball_tokenizer
from cheat_at_search.tools.code import make_patch_fn
from cheat_at_search.tools.eval import make_eval_fn, CodeGenSearchStrategy
from typing import List, Dict, Optional, Literal
from searcharray import SearchArray
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


corpus_dir = ensure_data_subdir("esci_indexed_corpus")
model = SentenceTransformer('all-MiniLM-L6-v2')


log_at("INFO")


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
            'id': row['product_id'],
            'title': row['title'],
            'description': row['description'],
            'score': row['score']
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
    Your task is to look at the data and improve the reranker code so that it returns more relevant results

    Edit the reranker python module using apply_patch method.

    You can run the reranker using the 'run_reranker' function, which takes a query and returns ranked, matching
    products.

    You can evaluate the reranker using the 'run_evals' function, which returns NDCG scores for all queries and mean NDCG. Your goal is to
    increase mean NDCG.

    Experiment with the current reranker by calling it with test queries. Improve the reranker based on the behavior you observe. Make edits and test while you edit.

    If NDCG does not go up after your edits, revert your changes using the 'revert_changes' function.

    Your code MUST have a function rerank_esci. It takes as parameters search_esci function and a query string. It
    returns a list of product IDs in the order you think best matches the query.

    Here are some examples of user queries, product titles, and human labels (Relevant, Partially Relevant, Irrelevant) that
    you are ranking:
"""


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
        partial = query_judgments[query_judgments['grade'].isin([1, 2])]

        partial = partial.sample(min(num_per_query - len(relevant) - len(irrelevant), len(partial)), random_state=seed)

        if len(irrelevant) == 0:
            # Sample random docs
            irrelevant = corpus.sample(num_per_query // 3, random_state=seed).copy()[['product_id']]
            irrelevant['grade'] = 0
            irrelevant['label'] = '😭'
            irrelevant['query'] = query

        # Format into prompt
        labeled = pd.concat([relevant, partial, irrelevant]).sample(frac=1, random_state=seed)
        labeled = labeled.sort_values(by='grade', ascending=False).head(num_per_query)
        labeled = labeled.merge(corpus, on='product_id', how='left', suffixes=('', '_y'))
        for item in labeled.to_dict(orient='records'):
            prompt += f"""

            User Query: {item['query']}
            Title: {item['title']}
            Description: {item['description']}
            Human Label: {item['label']} (grade: {item['grade']})

            """
    return prompt


class FinalMessage(BaseModel):
    """Final message indicating completion of the reranker improvement process."""
    message: str = Field(..., description="A message indicating that the reranker improvement process is complete.")


if __name__ == "__main__":
    num_queries = 20
    bm25 = BM25Search(corpus)
    graded_bm25 = run_strategy(bm25, judgments, num_queries=num_queries)
    bm25_ndcg = graded_bm25.groupby('query')['ndcg'].mean().mean()
    print(f"Baseline NDCG: {bm25_ndcg}")
    # best = BestPossibleResults(corpus, judgments)
    # graded_best = run_strategy(best, judgments, num_queries=num_queries)
    # best_ndcg = graded_best['ndcg'].mean()
    # print(f"Best Possible NDCG: {best_ndcg}")
    apply_patch, revert_changes = make_patch_fn(
        search_fn=search_esci,
        corpus=corpus,
        module_name="rerank_esci"
    )
    run_evals, run_reranker = make_eval_fn(
        corpus=corpus,
        judgments=judgments,
        module_name="rerank_esci",
        search_fn=search_esci,
        workers=16,
        num_queries=num_queries,
        seed=42
    )

    tools = [search_esci, apply_patch, run_reranker, run_evals,
             revert_changes]

    start_code = ""
    with open("cheat_at_search/start_rerank_esci.py", "r") as f:
        code = f.read()

    with open("rerank_esci.py", "w") as f:
        f.write(code)

    ndcgs = []
    for rounds in range(10):
        print(f"=== Generating Reranker Code Round {rounds} ===")

        with open("rerank_esci.py", "r") as f:
            code = f.read()

        prompt = build_few_shot_prompt(seed=42 + rounds * 100, num_queries=4, num_per_query=4)

        prompt += f"""

        Reranker code to improve:

{code}
"""
        print("Prompt is:")
        print(prompt)

        search_client = OpenAIAgent(tools=tools,
                                    model="openai/gpt-5",
                                    system_prompt=prompt,
                                    response_model=FinalMessage)
        resp: FinalMessage = search_client.loop()
        print("Final message from agent:")
        print(resp.message)

        ndcg = 0
        try:
            codegen_strategy = CodeGenSearchStrategy(corpus, workers=16)
            results_codegen = run_strategy(codegen_strategy, judgments, num_queries=num_queries)
            ndcg = results_codegen.groupby('query')['ndcg'].mean().mean()
        except Exception as e:
            print("Error running codegen strategy:", e)
            ndcg = 0
        print("=== End of Round ===")
        print(f"Round {rounds} complete.")
        print(f"Codegen NDCG: {ndcg}")
        ndcgs.append(ndcg)
    print(f"Baseline NDCG: {bm25_ndcg}")
    for i, ndcg in enumerate(ndcgs):
        print(f"Round {i} NDCG: {ndcg}")
