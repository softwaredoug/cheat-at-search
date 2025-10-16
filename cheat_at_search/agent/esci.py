from cheat_at_search.esci_data import corpus, judgments
from cheat_at_search.strategy.strategy import SearchStrategy
from cheat_at_search.agent.strategy import ReasoningSearchStrategy
from cheat_at_search.agent.openai_agent import OpenAIAgent
from cheat_at_search.search import run_strategy
from cheat_at_search.data_dir import ensure_data_subdir
from cheat_at_search.strategy import BM25Search, BestPossibleResults
from cheat_at_search.tokenizers import snowball_tokenizer
from typing import List, Dict, Optional, Literal, Union
from searcharray import SearchArray
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import importlib

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
        })
    print(f"Keywords {keywords} field: {field_to_search} operator: {operator} locale: {locale} -> {len(results)} results")
    return results


class Edit(BaseModel):
    """A single edit to apply to the reranker code."""
    anchor: str = Field(..., description="The anchor text to identify where the patch should be applied.")
    block_until: str = Field(..., description="The end of the block of text which the patch should be applied. Do not leave blank.")
    action: Literal['insert_after', 'replace', 'delete'] = Field(..., description="The action to perform: insert_after, replace, or delete.")
    text: str = Field(..., description="The text to insert or replace with. Ignored for delete action.")


class Edits(BaseModel):
    """A set of edits to apply to the reranker code and test queries to validate."""
    edits: List[Edit] = Field(..., description="A list of edits to apply to the reranker code.")
    test_queries: List[str] = Field(..., description="A list of test queries to validate the reranker after applying edits.")


class EditResult(BaseModel):
    """The result of applying edits to the reranker code."""
    success: bool = Field(..., description="Whether the edits were applied successfully and the reranker passed tests.")
    error_message: Optional[str] = Field(None, description="An error message if the edits failed to apply or tests failed.")
    query_results: Dict[str, Union[List[Dict], str]] = Field(..., description="The results of running the reranker on the test queries after applying edits.")


def apply_patch(patch: Edits) -> EditResult:
    """Apply an edit to reranker code."""
    try:
        print("Applying patch with edits:")
        with open("rerank_esci.py", "r") as f:
            code = f.read()

            for edit in patch.edits:
                anchor_index = code.find(edit.anchor)
                if anchor_index == -1:
                    raise ValueError(f"Anchor '{edit.anchor}' not found in code.")
                block_index = code.find(edit.block_until, anchor_index)
                if block_index == -1:
                    raise ValueError(f"Block until '{edit.block_until}' not found after anchor in code.")

                if edit.action == 'insert_after':
                    insertion_point = block_index + len(edit.block_until)
                    code = code[:insertion_point] + '\n' + edit.text + '\n' + code[insertion_point:]
                elif edit.action == 'replace':
                    code = code[:anchor_index] + edit.text + code[block_index + len(edit.block_until):]
                elif edit.action == 'delete':
                    code = code[:anchor_index] + code[block_index + len(edit.block_until):]
                else:
                    raise ValueError(f"Unknown action '{edit.action}'.")
        # Attempt to eval the code
        local_vars = {}
        exec(code, {}, local_vars)
        if 'rerank_esci' not in local_vars:
            print("Edited code does not define 'rerank_esci'")
            raise ValueError("The edited code does not define 'rerank_esci'.")
        # Test that rerank_esci is callable
        if not callable(local_vars['rerank_esci']):
            print("'rerank_esci' is not callable.")
            raise ValueError("'rerank_esci' is not callable.")
        # Call with test_queries
        edit_result = EditResult(success=True, error_message=None, query_results={})
        for query in patch.test_queries:
            try:
                results = local_vars['rerank_esci'](search_esci, query)[:10]
            except Exception as e:
                print(f"Error calling 'rerank_esci' with query '{query}': {e}")
                print("---")
                print(code)
                raise ValueError(f"Error calling 'rerank_esci' with query '{query}': {e}")

            try:
                if not isinstance(results, list):
                    print(f"'rerank_esci' did not return a list for query '{query}'.")
                    raise ValueError(f"'rerank_esci' did not return a list for query '{query}'.")
                dict_results = []
                for result in results:
                    product = corpus[corpus['product_id'] == result]
                    if len(product) == 0:
                        continue
                    product = product.iloc[0]
                    dict_results.append({
                        'id': product['product_id'],
                        'title': product['title'],
                        'description': product['description'],
                    })
                edit_result.query_results[query] = dict_results
            except Exception as e:
                print(f"Error collecting results with query '{query}': {e}")
                raise ValueError(f"Error calling 'rerank_esci' with query '{query}': {e}")

        with open("rerank_esci.py", "w") as f:
            f.write(code)
            print("Patched rerank_esci.py successfully.")
            return edit_result
    except Exception as e:
        print("Error applying patch:", e)
        return EditResult(success=False, error_message=str(e), query_results={})


class EvalResults(BaseModel):
    """The result of evaluating the reranker on ground truth judgments."""
    query_ndcgs: List[Dict] = Field(..., description="A list of dictionaries with 'query' and 'ndcg' keys.")
    mean_ndcg: float = Field(..., description="The mean NDCG across all queries.")


def run_evals() -> EvalResults:
    """Evaluate the current reranker on random sample of query document ground truth."""
    print("Running evals on all judgments")
    if hasattr(run_evals, "seed"):
        run_evals.seed += 1
    else:
        run_evals.seed = 50
    codegen_strategy = CodeGenSearchStrategy(corpus, workers=16)
    results_codegen = run_strategy(codegen_strategy, judgments, num_queries=20, seed=run_evals.seed)
    ndcgs = results_codegen.groupby('query')['ndcg'].mean()
    result = []
    for query, ndcg in ndcgs.items():
        result.append({'query': query, 'ndcg': ndcg})
        print(f"Query: {query} NDCG: {ndcg}")

    eval_result = EvalResults(
        query_ndcgs=result,
        mean_ndcg=ndcgs.mean()
    )
    print(f"Mean NDCG: {eval_result.mean_ndcg}")

    return eval_result


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


class CodeGenSearchStrategy(SearchStrategy):
    def __init__(self, corpus, cache=True,
                 workers=1):
        super().__init__(corpus, workers=workers)
        self.index = corpus

    def search(self, query, k=10):

        from rerank_esci import rerank_esci
        importlib.reload(importlib.import_module("rerank_esci"))

        product_ids = rerank_esci(search_esci, query)[:k]
        scores = np.arange(len(product_ids), 0, -1)
        top_k_ilocs = []
        for product_id in product_ids:
            iloc = self.index.index[self.index['product_id'] == product_id].tolist()
            if len(iloc):
                top_k_ilocs.append(iloc[0])
            else:
                print(f"Product ID {product_id} not found in corpus")
                continue
        scores = scores[:k]
        return top_k_ilocs, scores


def run_reranker(query, label=False) -> Union[List[Dict], str]:
    """Run the reranker. Returns a list of products or an error message.

    Set label=True to return human labels with product details (only use if query is from judgments).

    """
    query_judgments = None
    if label:
        query_judgments = judgments[judgments['query'] == query]
        if len(query_judgments) == 0:
            return "No judgments found for query: " + query
    try:
        print(f"Running reranker for query: {query}")
        from rerank_esci import rerank_esci
        importlib.reload(importlib.import_module("rerank_esci"))

        k = 10
        product_ids = rerank_esci(search_esci, query)
        scores = np.arange(len(product_ids), 0, -1)
        top_k = product_ids[:k]
        scores = scores[:k]

        products = corpus[corpus['product_id'].isin(top_k)]

        results = []
        for id, row in products.iterrows():
            grade = None
            results.append({
                'id': row['product_id'],
                'title': row['title'],
                'description': row['description'],
                'score': scores[np.where(top_k == row['product_id'])[0][0]]
            })
            if label:
                grade = query_judgments[query_judgments['product_id'] == row['product_id']]['grade'].values
                results[-1]['grade'] = int(grade[0]) if len(grade) > 0 else None

        return results
    except Exception as e:
        print("Error running reranker:", e)
        return "Error running reranker: " + str(e)


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
    tools = [search_esci, apply_patch, inspect_product, run_reranker, run_evals]

    start_code = ""
    with open("cheat_at_search/start_rerank_esci.py", "r") as f:
        code = f.read()

    with open("rerank_esci.py", "w") as f:
        f.write(code)

    ndcgs = []
    for rounds in range(3):
        print(f"=== Generating Reranker Code Round {rounds} ===")

        prompt = build_few_shot_prompt(seed=42 + rounds * 100, num_queries=4, num_per_query=4)

        prompt += f"""

        Reranker code to improve:

        ```python
        {code}
        ```

        """
        print("Prompt is:")
        print(prompt)

        search_client = OpenAIAgent(tools=tools,
                                    model="openai/gpt-5",
                                    system_prompt=prompt,
                                    response_model=FinalMessage)
        resp: FinalMessage = search_client.loop(prompt="")
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
