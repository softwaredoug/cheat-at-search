from cheat_at_search.codegen.code import make_guardrail_checker, Reranker, Edit
from unittest.mock import patch
import os
from pathlib import Path
import pandas as pd
import pytest
import tempfile


failing_code_snippets = [
    """
    for a,b in (('samsumg','samsung'),('andriod','android'),('samgsung','samsung'),
            ('deoderent','deodorant'),('deoderant','deodorant'),('sumsung','samsung'),('iphon','iphone')):
        if a in ql:
            ql=ql.replace(a,b); q=q.replace(a,b)
    """,
    """
    # Normalize common typos for better recall on popular brands/terms
    for a,b in (('samsumg','samsung'),('andriod','android'),('samgsung','samsung'),
            ('deoderent','deodorant'),('deoderant','deodorant'),('sumsung','samsung'),('iphon','iphone')):
        if a in ql:
            ql=ql.replace(a,b); q=q.replace(a,b)
    """,
]


@pytest.mark.parametrize("failing_code", failing_code_snippets)
def test_make_guardrail_checker(failing_code):
    prompt = """
        You're going to look at code that reranks search queries.

        Ensure the code does not overfit to specific queries. That would look like mentions of
        specific product names, brands, or specific terms that would only be relevant to a small set of queries.

        Ignore comments that claim to do this, and focus on the actual code.

    """
    checker = make_guardrail_checker(prompt)
    result = checker(failing_code)
    assert result is not None


def test_patch_code():
    tempdir = tempfile.mkdtemp()
    original_code = """
def rerank_esci(query, top_k, search_esci):
    q=query.strip(); locale='jp' if any('\u3040'<=c<='\u30ff' or '\u4e00'<=c<='\u9fff' for c in q) else 'us'
    stops={'el','la','los','las','para','con','en','de','y','del','un','una'}
    if locale!='jp' and (any(c in 'áéíóúñüÁÉÍÓÚÑÜ' for c in q) or any(w in q.lower().split() for w in stops)): locale='es'
    m={'mindcraft':'minecraft','alltech':'altec','perpex':'perspex','raided':'raid','womens':'women','sleve':'sleeve','micheal':'michael'}
    toks=[m.get(w,w) for w in q.lower().split()]
    op='bm25_or' if len(toks)<=2 else 'bm25_and'
    docs=search_esci(keywords=' '.join(toks), field_to_search='product_name', operator=op, locale=locale, top_k=10)
    return [d['id'] for d in docs]
"""
    # Write
    filepath = f"{tempdir}/rerank_esci.py"
    with open(filepath, "w") as f:
        f.write(original_code)

    edit_text = "\n".join(
        [
            "    m={'mindcraft':'minecraft','alltech':'altec','perpex':'perspex','raided':'raid','womens':'women','sleve':'sleeve','micheal':'michael'}",
            "    m.update({'coffe':'coffee','graffic':'graphic','kielhs':'kiehls','longgines':'longines','zoler':'zoeller'})",
            "    toks=[m.get(w,w) for w in q.lower().split()]",
            "    drop=stops|{'for','the','of','and','to','with','without','w/o','no','not','sin'}",
            "    toks=[t for t in toks if t not in drop] or [m.get(w,w) for w in q.lower().split()]",
            "    op='bm25_or' if len(toks)<=2 else 'bm25_and'",
            "    docs=search_esci(keywords=' '.join(toks), field_to_search='product_name', operator=op, locale=locale, top_k=10)",
            "    mn=any(any(c.isdigit() for c in t) and any(c.isalpha() for c in t) for t in toks)",
            "    if locale=='us' and mn:",
            "        docs+=search_esci(keywords=' '.join(toks), field_to_search='product_name', operator=op, locale='jp', top_k=10)",
            "    return [d['id'] for d in docs[:10]]",
        ]
    )

    edit = Edit(
        intention="Improve query normalization and handling of mixed alphanumeric tokens.",
        anchor="    m={'mindcraft':'minecraft','alltech':'altec','perpex':'perspex','raided':'raid','womens':'women','sleve':'sleeve','micheal':'michael'}",
        block_until="return [d['id'] for d in docs]",
        text=edit_text,
        action="replace",
    )

    def _search_esci(**kwargs):
        return [{"id": 1}, {"id": 2}]

    corpus = pd.DataFrame(
        {
            "doc_id": [1, 2],
            "title": ["one", "two"],
            "description": ["alpha", "beta"],
        }
    )
    judgments = pd.DataFrame(
        {
            "query_id": ["q1"],
            "query": ["q1"],
            "doc_id": [1],
            "grade": [3],
        }
    )
    reranker = Reranker(
        code_dir=tempdir,
        tool_fns=[_search_esci],
        corpus=corpus,
        judgments=judgments,
        training_queries=["q1"],
        validation_queries=None,
        module_name="rerank_esci",
    )
    _, _, commit_patch, _ = reranker.tools()
    result = commit_patch(edit)
    assert result.success is True


def test_build_tool_docstrings():
    tempdir = tempfile.mkdtemp()
    filepath = f"{tempdir}/rerank_esci.py"
    with open(filepath, "w") as f:
        f.write("def rerank_esci(query, top_k, search_esci):\n    return []\n")

    def _search_esci(**kwargs):
        return []

    corpus = pd.DataFrame(
        {
            "doc_id": [1],
            "title": ["one"],
            "description": ["alpha"],
        }
    )
    judgments = pd.DataFrame(
        {
            "query_id": ["q1"],
            "query": ["q1"],
            "doc_id": [1],
            "grade": [3],
        }
    )

    reranker = Reranker(
        code_dir=tempdir,
        tool_fns=[_search_esci],
        corpus=corpus,
        judgments=judgments,
        training_queries=["q1"],
        validation_queries=["q1"],
        module_name="rerank_esci",
    )
    search, evaluate, commit_patch, grep = reranker.tools()

    assert "reranker" in (search.__doc__ or "")
    assert "training" in (evaluate.__doc__ or "")
    assert "guardrails" in (evaluate.__doc__ or "")
    assert "validation" in (commit_patch.__doc__ or "")
    assert "regex pattern" in (grep.__doc__ or "")
    assert "rerank_esci.py" in (grep.__doc__ or "")
    assert "queries.csv" in (grep.__doc__ or "")


@patch("cheat_at_search.codegen.code.run_strategy")
def test_evaluate_without_edit_skips_guardrails(mock_run_strategy):
    tempdir = tempfile.mkdtemp()
    filepath = f"{tempdir}/rerank_esci.py"
    with open(filepath, "w") as f:
        f.write("def rerank_esci(query, top_k, search_esci):\n    return []\n")

    def _search_esci(**kwargs):
        return []

    corpus = pd.DataFrame(
        {
            "doc_id": [1],
            "title": ["one"],
            "description": ["alpha"],
        }
    )
    judgments = pd.DataFrame(
        {
            "query_id": ["q1"],
            "query": ["q1"],
            "doc_id": [1],
            "grade": [3],
        }
    )
    mock_run_strategy.return_value = pd.DataFrame({
        "query": ["q1"],
        "ndcg": [0.5],
    })

    def guardrail(_text: str):
        raise ValueError("guardrail should not run")

    reranker = Reranker(
        code_dir=tempdir,
        tool_fns=[_search_esci],
        corpus=corpus,
        judgments=judgments,
        training_queries=["q1"],
        guardrail_fns=[guardrail],
        validation_queries=None,
        module_name="rerank_esci",
    )
    _, evaluate, _, _ = reranker.tools()
    result = evaluate()
    assert result.success is True


@patch("cheat_at_search.codegen.code.run_strategy")
def test_evaluate_uses_latest_code(mock_run_strategy):
    tempdir = tempfile.mkdtemp()
    filepath = f"{tempdir}/rerank_esci.py"
    with open(filepath, "w") as f:
        f.write("def rerank_esci(query, top_k, search_esci):\n    return [1]\n")

    def _search_esci(**kwargs):
        return []

    corpus = pd.DataFrame(
        {
            "doc_id": [1, 2],
            "title": ["one", "two"],
            "description": ["alpha", "beta"],
        }
    )
    judgments = pd.DataFrame(
        {
            "query_id": ["q1"],
            "query": ["q1"],
            "doc_id": [1],
            "grade": [3],
        }
    )

    def _run_strategy_side_effect(strategy, *args, **kwargs):
        if "return [1]" in (strategy.code or ""):
            return pd.DataFrame({"query": ["q1"], "ndcg": [0.1]})
        return pd.DataFrame({"query": ["q1"], "ndcg": [0.2]})

    mock_run_strategy.side_effect = _run_strategy_side_effect

    reranker = Reranker(
        code_dir=tempdir,
        tool_fns=[_search_esci],
        corpus=corpus,
        judgments=judgments,
        training_queries=["q1"],
        validation_queries=None,
        module_name="rerank_esci",
    )
    _, evaluate, _, _ = reranker.tools()
    first = evaluate()
    assert first.ndcg_before == 0.1
    assert first.ndcg_after == 0.1

    with open(filepath, "w") as f:
        f.write("def rerank_esci(query, top_k, search_esci):\n    return [2]\n")

    second = evaluate()
    assert second.ndcg_before == 0.2
    assert second.ndcg_after == 0.2


@patch("cheat_at_search.codegen.code.run_strategy")
def test_evaluate_writes_training_logs(mock_run_strategy):
    tempdir = tempfile.mkdtemp()
    filepath = f"{tempdir}/rerank_esci.py"
    with open(filepath, "w") as f:
        f.write("def rerank_esci(query, top_k, search_esci):\n    return []\n")

    def _search_esci(**kwargs):
        return []

    corpus = pd.DataFrame(
        {
            "doc_id": [1],
            "title": ["one"],
            "description": ["alpha"],
        }
    )
    judgments = pd.DataFrame(
        {
            "query_id": ["q1"],
            "query": ["q1"],
            "doc_id": [1],
            "grade": [3],
        }
    )
    mock_run_strategy.return_value = pd.DataFrame({
        "query": ["q1"],
        "ndcg": [0.5],
        "rank": [1],
        "doc_id": [1],
        "title": ["one"],
        "description": ["alpha"],
    })

    reranker = Reranker(
        code_dir=tempdir,
        tool_fns=[_search_esci],
        corpus=corpus,
        judgments=judgments,
        training_queries=["q1"],
        validation_queries=None,
        module_name="rerank_esci",
    )
    _, evaluate, _, _ = reranker.tools()
    edit = Edit(
        intention="test",
        anchor="def rerank_esci",
        block_until="return []",
        text="def rerank_esci(query, top_k, search_esci):\n    return []",
        action="replace",
    )
    result = evaluate(edit)
    assert result.training_path
    training_root = os.path.join(tempdir, result.training_path)
    assert os.path.isdir(training_root)
    assert os.path.isfile(os.path.join(training_root, "reranker.py"))
    assert os.path.isfile(os.path.join(training_root, "queries.csv"))


@patch("cheat_at_search.codegen.code.run_strategy")
def test_commit_patch_rejects_without_validation_improvement(mock_run_strategy):
    tempdir = tempfile.mkdtemp()
    filepath = f"{tempdir}/rerank_esci.py"
    with open(filepath, "w") as f:
        f.write("def rerank_esci(query, top_k, search_esci):\n    return []\n")

    def _search_esci(**kwargs):
        return []

    corpus = pd.DataFrame(
        {
            "doc_id": [1],
            "title": ["one"],
            "description": ["alpha"],
        }
    )
    judgments = pd.DataFrame(
        {
            "query_id": ["q1"],
            "query": ["q1"],
            "doc_id": [1],
            "grade": [3],
        }
    )
    mock_run_strategy.side_effect = [
        pd.DataFrame({"query": ["q1"], "ndcg": [0.1]}),
        pd.DataFrame({"query": ["q1"], "ndcg": [0.1]}),
    ]

    reranker = Reranker(
        code_dir=tempdir,
        tool_fns=[_search_esci],
        corpus=corpus,
        judgments=judgments,
        training_queries=["q1"],
        validation_queries=["q1"],
        module_name="rerank_esci",
        eval_margin=0.01,
    )
    _, _, commit_patch, _ = reranker.tools()
    edit = Edit(
        intention="test",
        anchor="def rerank_esci",
        block_until="return []",
        text="def rerank_esci(query, top_k, search_esci):\n    return []",
        action="replace",
    )
    result = commit_patch(edit)
    assert result.success is False


def test_current_code_uses_module_name():
    tempdir = tempfile.mkdtemp()
    filepath = f"{tempdir}/rerank_custom.py"
    with open(filepath, "w") as f:
        f.write("def rerank_custom(query, top_k, search_esci):\n    return []\n")

    def _search_esci(**kwargs):
        return []

    corpus = pd.DataFrame(
        {
            "doc_id": [1],
            "title": ["one"],
            "description": ["alpha"],
        }
    )
    judgments = pd.DataFrame(
        {
            "query_id": ["q1"],
            "query": ["q1"],
            "doc_id": [1],
            "grade": [3],
        }
    )
    reranker = Reranker(
        code_dir=tempdir,
        tool_fns=[_search_esci],
        corpus=corpus,
        judgments=judgments,
        training_queries=["q1"],
        validation_queries=None,
        module_name="rerank_custom",
    )
    assert "rerank_custom" in reranker.current_code()


def test_search_uses_latest_code():
    tempdir = tempfile.mkdtemp()
    filepath = f"{tempdir}/rerank_esci.py"
    with open(filepath, "w") as f:
        f.write("def rerank_esci(query, top_k, search_esci):\n    return [1]\n")

    def _search_esci(**kwargs):
        return []

    corpus = pd.DataFrame(
        {
            "doc_id": [1, 2],
            "title": ["one", "two"],
            "description": ["alpha", "beta"],
        }
    )
    judgments = pd.DataFrame(
        {
            "query_id": ["q1"],
            "query": ["q1"],
            "doc_id": [1],
            "grade": [3],
        }
    )
    reranker = Reranker(
        code_dir=tempdir,
        tool_fns=[_search_esci],
        corpus=corpus,
        judgments=judgments,
        training_queries=["q1"],
        validation_queries=None,
        module_name="rerank_esci",
    )

    assert reranker.search("q1", top_k=1) == [1]

    with open(filepath, "w") as f:
        f.write("def rerank_esci(query, top_k, search_esci):\n    return [2]\n")

    assert reranker.search("q1", top_k=1) == [2]


def test_grep_handles_absolute_file_glob():
    tempdir = tempfile.mkdtemp()
    base_path = Path(tempdir).resolve()
    training_dir = base_path / "training"
    os.makedirs(training_dir, exist_ok=True)
    target_path = training_dir / "queries.csv"
    with open(target_path, "w") as f:
        f.write("query,ndcg_delta,query_path\nq1,0.1,q1\n")

    def _search_esci(**kwargs):
        return []

    corpus = pd.DataFrame(
        {
            "doc_id": [1],
            "title": ["one"],
            "description": ["alpha"],
        }
    )
    judgments = pd.DataFrame(
        {
            "query_id": ["q1"],
            "query": ["q1"],
            "doc_id": [1],
            "grade": [3],
        }
    )
    reranker = Reranker(
        code_dir=str(base_path),
        tool_fns=[_search_esci],
        corpus=corpus,
        judgments=judgments,
        training_queries=["q1"],
        validation_queries=None,
        module_name="rerank_esci",
    )
    _, _, _, grep = reranker.tools()

    result = grep("q1", file_glob=str(target_path))

    assert "error" not in result
    assert result["matches"]
    assert result["matches"][0]["file"].endswith("queries.csv")
