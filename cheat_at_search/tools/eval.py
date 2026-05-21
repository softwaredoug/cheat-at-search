from typing import List, Dict, Union
import os

import numpy as np
import pandas as pd

from cheat_at_search.logger import log_to_stdout
from cheat_at_search.search import run_strategy
from cheat_at_search.tools.code import (
    CodeGenSearchStrategy,
    Doc,
    EvalResults,
    QueryEvalResult,
    Reranker,
    grade_to_emoji,
)


def _resolve_logger(logger=None, logger_name: str = "eval"):
    if logger is not None:
        return logger
    return log_to_stdout(logger_name=logger_name)


def _id_column(corpus: pd.DataFrame) -> str:
    return "doc_id" if "doc_id" in corpus.columns else "product_id"


def make_eval_fn(
    corpus,
    judgments,
    code_dir: str,
    search_fn,
    module_name="rerank_esci",
    workers=4,
    num_queries=20,
    seed=42,
    logger=None,
) -> callable:
    filepath = os.path.join(code_dir, module_name + ".py")
    logger = _resolve_logger(logger)
    id_col = _id_column(corpus)
    reranker = Reranker(
        code_dir=code_dir,
        tool_fns=[search_fn],
        module_name=module_name,
        logger=logger,
    )

    def run_evals() -> EvalResults:
        """Evaluate the current reranker on random sample of query document ground truth."""
        logger.info("Running evals on all judgments")
        code = None
        with open(filepath, 'r') as f:
            code = f.read()
        codegen_strategy = CodeGenSearchStrategy(
            corpus,
            tool_fns=[search_fn],
            module_name=module_name,
            code=code,
            workers=workers,
            logger=logger,
        )
        results_codegen = run_strategy(
            codegen_strategy,
            judgments,
            num_queries=num_queries,
            seed=seed,
            cache=False,
        )
        ndcgs = results_codegen.groupby('query')['ndcg'].mean()
        result: List[QueryEvalResult] = []
        for query, ndcg in ndcgs.items():
            relevant_doc = None
            for grade in [3, 2, 1, 0]:
                relevant_docs = judgments[(judgments['query'] == query) & (judgments['grade'] == grade)]
                if len(relevant_docs) > 0:
                    doc_row = relevant_docs.iloc[0]
                    doc = corpus[corpus[id_col] == doc_row[id_col]]
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
            k = 10
            doc_ids = reranker.run_reranker(query, top_k=k)
            scores = np.arange(len(doc_ids), 0, -1)
            scores = scores[:k]

            results = []
            for doc_id, score in zip(doc_ids, scores):
                grade = None
                corpus_row = corpus[corpus[id_col] == doc_id]
                results.append({
                    'id': doc_id,
                    'title': corpus_row['title'].iloc[0],
                    'description': corpus_row['description'].iloc[0],
                    'score': int(score)
                })
                if label:
                    grade = query_judgments[query_judgments[id_col] == doc_id]['grade'].values
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


def make_eval_guardrail(
    corpus,
    judgments,
    search_fn,
    module_name="rerank_esci",
    seed=1234,
    num_queries=100,
    workers=4,
    logger=None,
) -> callable:
    return Reranker.make_eval_guardrail(
        corpus,
        judgments,
        tool_fns=[search_fn],
        module_name=module_name,
        seed=seed,
        num_queries=num_queries,
        workers=workers,
        logger=logger,
    )
