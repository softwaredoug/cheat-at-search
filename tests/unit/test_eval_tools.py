from unittest.mock import patch

import pandas as pd

from cheat_at_search.codegen.code import CodeGenSearchStrategy, _make_eval_guardrail


def test_codegen_search_strategy_rerank_signature():
    code = """
def rerank_esci(query, top_k, search_fn):
    return search_fn(query)
"""

    def search_fn(query):
        assert query == "hello"
        return [101, 102]

    corpus = pd.DataFrame({
        "product_id": [101, 102],
        "doc_id": [101, 102],
        "title": ["one", "two"],
        "description": ["alpha", "beta"],
    })
    strategy = CodeGenSearchStrategy(
        corpus,
        tool_fns=[search_fn],
        module_name="rerank_esci",
        code=code,
        workers=1,
    )
    top_k, scores = strategy.search("hello", k=2)
    assert top_k == [0, 1]
    assert len(scores) == 2


def test_codegen_search_strategy_coerces_string_doc_ids():
    code = """
def rerank_esci(query, top_k):
    return ["101", "102"]
"""
    corpus = pd.DataFrame({
        "doc_id": [101, 102],
        "title": ["one", "two"],
        "description": ["alpha", "beta"],
    })
    strategy = CodeGenSearchStrategy(
        corpus,
        tool_fns=[],
        module_name="rerank_esci",
        code=code,
        workers=1,
    )

    top_k, scores = strategy.search("hello", k=2)

    assert top_k == [0, 1]
    assert len(scores) == 2


def test_codegen_search_strategy_coerces_int_doc_ids():
    code = """
def rerank_esci(query, top_k):
    return [101, 102]
"""
    corpus = pd.DataFrame({
        "doc_id": ["101", "102"],
        "title": ["one", "two"],
        "description": ["alpha", "beta"],
    })
    strategy = CodeGenSearchStrategy(
        corpus,
        tool_fns=[],
        module_name="rerank_esci",
        code=code,
        workers=1,
    )

    top_k, scores = strategy.search("hello", k=2)

    assert top_k == [0, 1]
    assert len(scores) == 2


@patch("cheat_at_search.codegen.code.run_strategy")
def test_make_eval_guardrail_disables_cache(mock_run_strategy):
    corpus = pd.DataFrame({
        "doc_id": [101],
        "doc_id": [101],
        "title": ["one"],
        "description": ["alpha"],
    })
    judgments = pd.DataFrame({
        "query_id": ["q1"],
        "query": ["q1"],
        "doc_id": [101],
        "grade": [3],
    })

    mock_run_strategy.return_value = pd.DataFrame({
        "query": ["q1"],
        "ndcg": [0.2],
    })

    eval_guardrail = _make_eval_guardrail(
        corpus,
        judgments,
        tool_fns=[lambda query: []],
        num_queries=1,
        seed=1,
        queries=["q1"],
    )
    ndcgs, results_df = eval_guardrail(
        "def rerank_esci(query, top_k, search_fn): return [101]",
        results=True,
    )
    assert ndcgs["q1"] == 0.2
    assert results_df["query"].tolist() == ["q1"]
    assert mock_run_strategy.call_args.kwargs["cache"] is False
