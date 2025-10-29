from cheat_at_search.esci_data import corpus, judgments
from cheat_at_search.agent.openai_agent import OpenAIAgent
from cheat_at_search.search import run_strategy
from cheat_at_search.logger import log_at
from cheat_at_search.data_dir import ensure_data_subdir
from cheat_at_search.strategy import BM25Search
from cheat_at_search.tokenizers import snowball_tokenizer
from cheat_at_search.tools.code import make_patch_fn, make_guardrail_checker, make_length_validator
from cheat_at_search.tools.eval import make_eval_fn, CodeGenSearchStrategy, make_eval_guardrail
from typing import List, Dict, Optional, Literal
from searcharray import SearchArray
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import random
import os


corpus_dir = ensure_data_subdir("esci_indexed_corpus")


log_at("INFO")

# embedding_model_name = 'microsoft/Multilingual-MiniLM-L12-H384'
# model = SentenceTransformer(embedding_model_name)


try:
    corpus = pd.read_pickle(corpus_dir / "corpus.pkl")  # noqa
except FileNotFoundError:
    corpus['brand_snowball'] = SearchArray.index(corpus['product_brand'].fillna(''), snowball_tokenizer)
    corpus['product_color_snowball'] = SearchArray.index(corpus['product_color'].fillna(''), snowball_tokenizer)
    corpus['title_snowball'] = SearchArray.index(corpus['title'], snowball_tokenizer)
    corpus['description_snowball'] = SearchArray.index(corpus['description'], snowball_tokenizer)
    corpus['all_text'] = corpus['title'] + ' ' + corpus['description']
    corpus['all_text_snowball'] = SearchArray.index(corpus['all_text'], snowball_tokenizer)
    corpus.to_pickle(corpus_dir / "corpus.pkl")


# embeddings = {}
# for field in ['title', 'description', 'all_text']:
#
#     try:
#        embeddings[field] = np.load(corpus_dir / f"{field}_{embedding_model_name}.npy")
#        print(f"Loaded {field} MiniLM embeddings from disk.")
#    except FileNotFoundError:
#        print(f"Computing {field} MiniLM embeddings...")
#        embeddings[field] = model.encode(corpus[field].tolist(), show_progress_bar=True)
#        np.save(corpus_dir / f"{field}_{embedding_model_name}.npy", embeddings[field])


def search_esci(keywords: str,
                field_to_search: Literal['product_name', 'product_description'],  # , 'all_text'] = 'all_text',
                operator: Literal['bm25_and', 'bm25_or'],  # , 'bm25_phrase', 'bm25_bigram'] = 'bm25_or',
                locale: Literal['es', 'us', 'jp'] = 'us',
                top_k: int = 5) -> List[Dict]:
    """
    Search an attribute of the Amazon ESCI product corpus using BM25 or MiniLM embeddings, depending on the requested operator.

    Args:
        keywords: The search query string.
        field_to_search: The field to search in. Options are 'product_name' and 'product_description'.
        operator: How to search the field
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
    elif field_to_search == 'all_text':
        field_name = 'all_text_snowball'
    else:
        raise ValueError("field_to_search must be 'product_name' or 'product_description'")

    if operator in ['bm25_and', 'bm25_or']:
        for token in query_tokens:
            scores += corpus[field_name].array.score(token)

        if operator == 'bm25_and':
            for token in query_tokens:
                require_mask = (corpus[field_name].array.score(token) > 0)
                scores = scores * require_mask
    elif operator == 'bm25_phrase':
        phrase_score = corpus[field_name].array.score(query_tokens)
        scores += phrase_score
    elif operator == 'bm25_bigram':
        for bigram in zip(query_tokens, query_tokens[1:]):
            bigram = list(bigram)
            scores += corpus[field_name].array.score(bigram)

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

    ## The task

    Your task is to improve the reranker code so that it returns more relevant results for Amazon e-commerce search queries. Take note of
    the human labels provided (what the reranker is evaluated against). A good reranker ranks relevant products higher than irrelevant products.
    It uses the search function 'search_esci' to get an initial set of candidate products
    (see the corresponding tool 'search_esci' for how it works).

    At any given point in time, reranker code exists in rerank_esci.py file. You modify this file with the 'apply_patch' function. Your ultimate goal is to
    apply one or more changes to this file.

    You can run this reranker using the 'run_reranker' function, which takes a query and returns ranked, matching products. You
    can also evaluate the current reranker in rerank_esci.py using the 'run_evals' function, which returns NDCG scores for all queries and mean NDCG.

    Before comitting changes, "play" with the proposed code change using 'try_out_patch' function which shows you per-query NDCG changes for your proposed code change without modifying the actual reranker code.

    After you find a positive change, try to commit your code change using 'apply_patch' function, which modifies the actual reranker code.

    Note no changes are permanent until you call 'apply_patch'. So all patches should be relative to the current code in rerank_esci.py.

    ## What requirements must code meet?

    Your code MUST have a function rerank_esci. It takes as parameters search_esci function and a query string. It
    returns a list of product IDs in the order you think best matches the query.

    Pay attention to 'apply_patch' and 'try_out_patch' and the guardrails listed

    Since 'apply_patch' modifies the actual reranker code, it has an additional check:

    There is a seperate validation set of queries to ensure you are not overfitting to specific queries. On 'apply_patch', your code change will be evaluated on the hidden validation set, and the change will only be applied if it improves the validation set NDCG by a small margin.

    ## What does the data look like?

    Here are some examples of user queries, product titles, and human labels that you are ranking:
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


def trial_run(module_name="rerank_esci",
              num_test_queries=100,
              num_validation_queries=50,
              num_training_queries=50,
              training_seed=5678,
              validation_seed=1234,
              test_seed=42,
              code_examples=None,
              code_examples_ndcgs=None,
              start_code=None,
              start_code_ndcg=None) -> (float, str):
    overfit_to_queries_guardrail = make_guardrail_checker(prompt="""

        You're going to look at code that reranks search queries.

        Ensure the code does not overfit to specific queries. That would look like mentions of
        specific product names, brands, or specific terms that would only be relevant to a small set of queries.

        Ignore comments that claim to do this, and focus on the actual code.
    """)

    length_guardrail = make_length_validator(max_lines=10, max_cols=120)

    validation_guardrail = make_eval_guardrail(
        corpus=corpus,
        judgments=judgments,
        search_fn=search_esci,
        seed=validation_seed,
        num_queries=num_validation_queries
    )

    training_eval = make_eval_guardrail(
        corpus=corpus,
        judgments=judgments,
        search_fn=search_esci,
        seed=training_seed,
        num_queries=num_training_queries
    )

    apply_patch, try_out_patch, revert_changes = make_patch_fn(
        search_fn=search_esci,
        corpus=corpus,
        module_name=module_name,
        guardrail_fns=[length_guardrail, overfit_to_queries_guardrail],
        validation_eval_fn=validation_guardrail,
        training_eval_fn=training_eval
    )
    run_evals, run_reranker = make_eval_fn(
        corpus=corpus,
        judgments=judgments,
        module_name=module_name,
        search_fn=search_esci,
        workers=16,
        num_queries=num_training_queries,
        seed=training_seed
    )

    tools = [search_esci, apply_patch, try_out_patch,
             run_reranker, run_evals,
             revert_changes]

    if start_code:
        with open(f"{module_name}.py", "w") as f:
            f.write(start_code)

    with open(f"{module_name}.py", "r") as f:
        code = f.read()

    prompt = build_few_shot_prompt(seed=42 + rounds * 100, num_queries=4, num_per_query=4)

    if code_examples:
        code_formatted = ""
        for zip_code, ndcg in zip(code_examples, code_examples_ndcgs):
            code_formatted += f"""
Reranker code with NDCG {ndcg}:
    {zip_code}

"""
        prompt += f"""

        Here are different rerankers you have had success with before:

    {code_formatted}

    """
    prompt += f"""

        Reranker code you should improve:

    {code}
    """
    print("Prompt is:")
    print(prompt)

    search_client = OpenAIAgent(tools=tools,
                                model="openai/gpt-5",
                                system_prompt=prompt,
                                max_tokens=1_100_000,
                                response_model=FinalMessage)
    resp: FinalMessage = search_client.loop()
    print("Final message from agent:")
    print(resp.message)

    ndcg = 0
    try:
        codegen_strategy = CodeGenSearchStrategy(corpus, workers=16,
                                                 search_fn=search_esci,
                                                 module_name=module_name)
        results_codegen = run_strategy(codegen_strategy, judgments,
                                       num_queries=num_test_queries,
                                       seed=test_seed)
        ndcg = results_codegen.groupby('query')['ndcg'].mean().mean()
    except Exception as e:
        print("Error running codegen strategy:", e)
        ndcg = 0

    latest_code = ""
    with open(f"{module_name}.py", "r") as f:
        latest_code = f.read()

    return ndcg, latest_code


if __name__ == "__main__":
    num_test_queries = 100
    num_validation_queries = 250
    num_training_queries = 100
    training_seed = 5678
    validation_seed = 1234
    test_seed = 42

    rand_int = random.randint(0, 100000)
    module_name = f"rerank_esci_{rand_int}"
    while os.path.exists(f"{module_name}.py"):
        rand_int = random.randint(0, 100000)
        module_name = f"rerank_esci_{rand_int}"

    bm25 = BM25Search(corpus)
    graded_bm25 = run_strategy(bm25, judgments,
                               num_queries=num_test_queries,
                               seed=test_seed)
    bm25_ndcg = graded_bm25.groupby('query')['ndcg'].mean().mean()
    print(f"Baseline NDCG: {bm25_ndcg}")
    # best = BestPossibleResults(corpus, judgments)
    # graded_best = run_strategy(best, judgments, num_queries=num_queries)
    # best_ndcg = graded_best['ndcg'].mean()
    # print(f"Best Possible NDCG: {best_ndcg}")

    start_code = ""
    with open("cheat_at_search/start_rerank_esci.py", "r") as f:
        start_code = f.read()

    with open(f"{module_name}.py", "w") as f:
        f.write(start_code)

    codegen_strategy = CodeGenSearchStrategy(corpus, workers=16,
                                             search_fn=search_esci,
                                             module_name=module_name)
    results_codegen = run_strategy(codegen_strategy, judgments,
                                   num_queries=num_test_queries,
                                   seed=test_seed)
    ndcg = results_codegen.groupby('query')['ndcg'].mean().mean()
    start_code_ndcg = ndcg
    print(f"Starting Code NDCG: {start_code_ndcg}")

    # Primer rounds with no code examples
    ndcgs = []
    code_examples = []
    for rounds in range(10):
        print(f"=== Generating Reranker Code Round {rounds} ===")
        last_code = ""
        with open(f"{module_name}.py", "r") as f:
            last_code = f.read()
        ndcg, code = trial_run(
            num_test_queries=num_test_queries,
            num_validation_queries=num_validation_queries,
            num_training_queries=num_training_queries,
            training_seed=training_seed,
            validation_seed=validation_seed,
            test_seed=test_seed,
            start_code=last_code
        )
        training_seed += 1
        # validation_seed += 1

        ndcgs.append(ndcg)
        code_examples.append(code)
        print("=== End of Round ===")
        print(f"Round {rounds} complete.")
        print(f"Codegen NDCG: {ndcg} (test)")
        print("All rounds so far:")
        print(f"Baseline (BM25) NDCG: {bm25_ndcg}")
        print(f"Starting Code NDCG: {start_code_ndcg}")
        for i, ndcg in enumerate(ndcgs):
            print(f"Round {i} NDCG: {ndcg}")

        with open(f"{module_name}_round_{rounds}.py", "w") as f:
            f.write(code)
