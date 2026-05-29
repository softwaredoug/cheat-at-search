import pandas as pd
from cheat_at_search.strategy.best import BestPossibleResults


def test_best_search_returns_labeled_results():
    corpus = pd.DataFrame({
        "doc_id": ["doc1", "doc2", "doc3"],
        "title": ["Product A", "Product B", "Product C"],
    })
    judgments = pd.DataFrame({
        "query": ["test query", "test query", "test query"],
        "doc_id": ["doc1", "doc2", "doc3"],
        "grade": [2, 1, 0],
    })

    strategy = BestPossibleResults(corpus, judgments)
    top_k, scores = strategy.search("test query", k=2)

    assert len(top_k) == 2
    assert len(scores) == 2
    assert scores[0] == 2
    assert scores[1] == 1


def test_best_search_skips_missing_docs():
    corpus = pd.DataFrame({
        "doc_id": ["doc1"],
        "title": ["Product A"],
    })
    judgments = pd.DataFrame({
        "query": ["test query", "test query"],
        "doc_id": ["doc1", "doc_missing"],
        "grade": [2, 1],
    })

    strategy = BestPossibleResults(corpus, judgments)
    top_k, scores = strategy.search("test query", k=2)

    assert len(top_k) == 1
    assert len(scores) == 1
    assert scores[0] == 2


def test_best_search_returns_empty_for_unknown_query():
    corpus = pd.DataFrame({
        "doc_id": ["doc1"],
        "title": ["Product A"],
    })
    judgments = pd.DataFrame({
        "query": ["other query"],
        "doc_id": ["doc1"],
        "grade": [2],
    })

    strategy = BestPossibleResults(corpus, judgments)
    top_k, scores = strategy.search("test query", k=2)

    assert len(top_k) == 0
    assert len(scores) == 0


def test_best_search_respects_k_limit():
    corpus = pd.DataFrame({
        "doc_id": ["doc1", "doc2", "doc3"],
        "title": ["Product A", "Product B", "Product C"],
    })
    judgments = pd.DataFrame({
        "query": ["test query"] * 3,
        "doc_id": ["doc1", "doc2", "doc3"],
        "grade": [2, 2, 2],
    })

    strategy = BestPossibleResults(corpus, judgments)
    top_k, scores = strategy.search("test query", k=2)

    assert len(top_k) == 2
    assert len(scores) == 2
