from typing import List, Dict, Union, Optional
from pydantic import BaseModel, Field
from cheat_at_search.search import run_strategy
from cheat_at_search.strategy import SearchStrategy
from cheat_at_search.logger import log_to_stdout
import importlib
import numpy as np


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
                 module_name: Optional[str] = None,
                 code: Optional[str] = None,
                 cache=True,
                 workers=1):
        super().__init__(corpus, workers=workers)
        self.index = corpus
        self.module_name = module_name
        self.search_fn = search_fn
        self.code = code
        if not module_name and not code:
            raise ValueError("Either module_name or code must be provided.")

    def search(self, query, k=10):
        if self.code:
            rerank_fn = _rerank_fn_from_code(self.code)
        elif self.module_name:
            rerank_fn = _get_rerank_fn(self.module_name)

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


class EvalResults(BaseModel):
    """The result of evaluating the reranker on ground truth judgments."""
    query_ndcgs: List[Dict] = Field(..., description="A list of dictionaries with 'query' and 'ndcg' keys.")
    mean_ndcg: float = Field(..., description="The mean NDCG across all queries.")


def make_eval_fn(corpus, judgments, module_name: str, search_fn,
                 workers=16,
                 num_queries=20,
                 seed=42) -> callable:

    def run_evals() -> EvalResults:
        """Evaluate the current reranker on random sample of query document ground truth."""
        logger.info("Running evals on all judgments")
        codegen_strategy = CodeGenSearchStrategy(corpus, workers=workers,
                                                 module_name=module_name,
                                                 search_fn=search_fn)
        results_codegen = run_strategy(codegen_strategy, judgments, num_queries=num_queries,
                                       seed=seed)
        ndcgs = results_codegen.groupby('query')['ndcg'].mean()
        result = []
        for query, ndcg in ndcgs.items():
            result.append({'query': query, 'ndcg': ndcg})
            logger.info(f"Query: {query} NDCG: {ndcg:.4f}")

        eval_result = EvalResults(
            query_ndcgs=result,
            mean_ndcg=ndcgs.mean()
        )
        logger.info(f"Mean NDCG (eval tool): {eval_result.mean_ndcg}")

        return eval_result

    def grade_to_emoji(grade):
        if grade == 3:
            return '🤩'
        elif grade == 2:
            return '🙂'
        elif grade == 1:
            return '😐'
        elif grade == 0:
            return '😭'

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


def make_eval_guardrail(corpus, judgments, search_fn, seed=1234, num_queries=100):

    def eval_guardrail(code: str) -> float:
        """Evaluate on validation set to avoid overfitting. Returns mean NDCG."""
        strategy = CodeGenSearchStrategy(corpus,
                                         search_fn=search_fn,
                                         code=code,
                                         workers=16)
        results = run_strategy(strategy, judgments, num_queries=num_queries,
                               seed=seed)
        ndcg = results.groupby('query')['ndcg'].mean().mean()
        return ndcg
