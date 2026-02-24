from typing import List, Dict, Union, Optional, Literal
from pydantic import BaseModel, Field
from cheat_at_search.search import run_strategy
from cheat_at_search.strategy import SearchStrategy
from cheat_at_search.logger import log_to_stdout
import importlib
import numpy as np
import pandas as pd
import os


logger = log_to_stdout(logger_name="eval")


def _get_rerank_fn(module_name: str):
    mod = importlib.import_module(module_name)
    importlib.reload(mod)
    # Look for fn beginning with "rerank_"
    rerank_fn = None
    for attr in dir(mod):
        if attr.startswith("rerank_"):
            rerank_fn = getattr(mod, attr)
            break
    return rerank_fn


def _rerank_fn_from_code(code: str):
    exec_globals = {}
    exec(code, exec_globals)
    rerank_fn = None
    for name, obj in exec_globals.items():
        if name.startswith("rerank_"):
            rerank_fn = obj
            break
    return rerank_fn


class CodeGenSearchStrategy(SearchStrategy):
    def __init__(self, corpus,
                 search_fn,
                 code: Optional[str] = None,
                 cache=True,
                 workers=1):
        super().__init__(corpus, workers=workers)
        self.index = corpus
        self.search_fn = search_fn
        self.code = code

    def search(self, query, k=10):
        if self.code:
            rerank_fn = _rerank_fn_from_code(self.code)

        product_ids = rerank_fn(self.search_fn, query)[:k]
        scores = np.arange(len(product_ids), 0, -1)
        top_k_ilocs = []
        for product_id in product_ids:
            iloc = self.index.index[self.index['product_id'] == product_id].tolist()
            if len(iloc):
                top_k_ilocs.append(iloc[0])
            else:
                logger.info(f"Product ID {product_id} not found in corpus")
                continue
        scores = scores[:k]
        return top_k_ilocs, scores


def grade_to_emoji(grade):
    if grade == 3:
        return '🤩'
    elif grade == 2:
        return '🙂'
    elif grade == 1:
        return '😐'
    elif grade == 0:
        return '😭'


class Doc(BaseModel):
    """A document returned by the search system."""
    title: str = Field(..., description="The title of the document.")
    label: Literal['🤩', '🙂', '😐', '😭', ''] = Field(..., description="The human judgment label for the document.")


class QueryEvalResult(BaseModel):
    query: str = Field(..., description="The user query being evaluated.")
    ndcg: float = Field(..., description="The NDCG score for the query.")
    relevant_doc: Doc = Field(..., description="An example of a relevant document for the query.")


class EvalResults(BaseModel):
    """The result of evaluating the reranker on ground truth judgments."""
    query_ndcgs: List[QueryEvalResult] = Field(..., description="The NDCG scores for each query.")
    mean_ndcg: float = Field(..., description="The mean NDCG across all queries.")


def make_eval_fn(corpus, judgments, code_dir: str, search_fn,
                 module_name="rerank_esci",
                 workers=4,
                 num_queries=20,
                 seed=42) -> callable:
    filepath = os.path.join(code_dir, module_name + ".py")

    def run_evals() -> EvalResults:
        """Evaluate the current reranker on random sample of query document ground truth."""
        logger.info("Running evals on all judgments")
        code = None
        with open(filepath, 'r') as f:
            code = f.read()
        codegen_strategy = CodeGenSearchStrategy(corpus, workers=workers,
                                                 code=code,
                                                 search_fn=search_fn)
        results_codegen = run_strategy(codegen_strategy, judgments, num_queries=num_queries,
                                       seed=seed)
        ndcgs = results_codegen.groupby('query')['ndcg'].mean()
        result: List[QueryEvalResult] = []
        for query, ndcg in ndcgs.items():
            relevant_doc = None
            for grade in [3, 2, 1, 0]:
                relevant_docs = judgments[(judgments['query'] == query) & (judgments['grade'] == grade)]
                if len(relevant_docs) > 0:
                    doc_row = relevant_docs.iloc[0]
                    doc = corpus[corpus['product_id'] == doc_row['product_id']]
                    relevant_doc = Doc(title=doc['title'].iloc[0],
                                       label=grade_to_emoji(doc_row['grade']))
                    break
            if relevant_doc is None:
                relevant_doc = Doc(title="No relevant doc found", label='😭')
            result.append(QueryEvalResult(
                query=query,
                ndcg=ndcg,
                relevant_doc=relevant_doc
            ))

            logger.info(f"Query: {query} NDCG: {ndcg:.4f} Relevant Doc Title: {relevant_doc.title}")
        assert len(result) == len(ndcgs), "Result length does not match number of queries"

        eval_result = EvalResults(
            query_ndcgs=result,
            mean_ndcg=ndcgs.mean()
        )
        logger.info(f"Mean NDCG (eval tool): {eval_result.mean_ndcg}")

        return eval_result

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
            logger.info(f"Running reranker for query: {query} (label={label})")
            rerank_fn = _get_rerank_fn(module_name)

            k = 10
            product_ids = rerank_fn(search_fn, query)
            scores = np.arange(len(product_ids), 0, -1)
            scores = scores[:k]

            results = []
            for product_id, score in zip(product_ids, scores):
                grade = None
                corpus_row = corpus[corpus['product_id'] == product_id]
                results.append({
                    'id': product_id,
                    'title': corpus_row['title'].iloc[0],
                    'description': corpus_row['description'].iloc[0],
                    'score': int(score)
                })
                if label:
                    grade = query_judgments[query_judgments['product_id'] == product_id]['grade'].values
                    if len(grade) == 0:
                        grade = None
                    else:
                        grade = grade[0]
                        grade = int(grade)
                        grade_emoji = grade_to_emoji(grade)
                        if grade:
                            results[-1]['grade'] = int(grade)
                            results[-1]['label'] = grade_emoji

            return results
        except Exception as e:
            logger.info("Error running reranker:", e)
            return "Error running reranker: " + str(e)

    return run_evals, run_reranker


def make_eval_guardrail(corpus, judgments, search_fn, seed=1234, num_queries=100, workers=4) -> callable:

    def eval_guardrail(code: str) -> float:
        """Evaluate on validation set to avoid overfitting. Returns query NDCGs."""
        strategy = CodeGenSearchStrategy(corpus,
                                         search_fn=search_fn,
                                         code=code,
                                         workers=workers)
        results = run_strategy(strategy, judgments, num_queries=num_queries,
                               seed=seed)
        ndcgs = results.groupby('query')['ndcg'].mean()
        return ndcgs
    return eval_guardrail


class HumanEvaluation(BaseModel):
    """A human judgment of a search result for a given user query."""
    user_query: str = Field(..., description="The original user search query")
    label: Literal['🤩', '🙂', '😐', '😭'] = Field(..., description="The human judgment label for the search results")
    grade: int = Field(..., description="The numeric grade for the search result")
    doc_title: str = Field(..., description="The name of the item judged")
    dec_description: str = Field(..., description="The description of the item judged")


def make_judgments_tool(query_to_corpus, max_grade=2, min_grade=0):

    def get_human_judgments(user_query: str) -> List[HumanEvaluation]:
        """Get a sample of evals for a given user query. When you want to deep-dive
            into a specific query and see how to optimize it (though you should not
            overfit, try to generalize from what you learn).

            This is different from running the reranker with labels on, as it gives
            a balanced set of relevant, irrelevant, and partially relevant items.

           Returns list of human evaluations. Empty if query not in ground truth.
        """
        K = 10
        print(f"Getting human judgments for query: {user_query}")
        labeled = query_to_corpus.loc[query_to_corpus['query'] == user_query]
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
                                           label=grade_to_emoji(item['label']),
                                           grade=item['grade'],
                                           doc_title=item['title'],
                                           doc_description=item['description']))

        return results
    return get_human_judgments
