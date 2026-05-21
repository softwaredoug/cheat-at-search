from pathlib import Path
from unittest.mock import patch

import pandas as pd

from cheat_at_search.tools.eval import CodeGenSearchStrategy, make_eval_fn, make_eval_guardrail


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


@patch("cheat_at_search.tools.eval.run_strategy")
def test_make_eval_fn_disables_cache(mock_run_strategy, tmp_path):
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
    code_path = Path(tmp_path) / "rerank_esci.py"
    code_path.write_text(
        "def rerank_esci(query, top_k, search_fn):\n    return [101]\n",
        encoding="utf-8",
    )

    mock_run_strategy.return_value = pd.DataFrame({
        "query": ["q1"],
        "ndcg": [0.5],
    })

    run_evals, _ = make_eval_fn(
        corpus,
        judgments,
        code_dir=str(tmp_path),
        search_fn=lambda query: [],
        num_queries=1,
        seed=1,
    )
    result = run_evals()
    assert result.mean_ndcg == 0.5
    assert mock_run_strategy.call_args.kwargs["cache"] is False


@patch("cheat_at_search.tools.code.run_strategy")
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

    eval_guardrail = make_eval_guardrail(
        corpus,
        judgments,
        search_fn=lambda query: [],
        num_queries=1,
        seed=1,
    )
    ndcgs, results_df = eval_guardrail(
        "def rerank_esci(query, top_k, search_fn): return [101]",
        results=True,
    )
    assert ndcgs["q1"] == 0.2
    assert results_df["query"].tolist() == ["q1"]
    assert mock_run_strategy.call_args.kwargs["cache"] is False
