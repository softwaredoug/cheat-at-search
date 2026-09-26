from unittest.mock import patch

import pandas as pd


@patch("cheat_at_search.scifact_data.load_or_build_lexical_corpus")
@patch("cheat_at_search.scifact_data._read_jsonl")
@patch("cheat_at_search.scifact_data._source_path")
@patch("cheat_at_search.scifact_data.download_scifact")
@patch("cheat_at_search.scifact_data.ensure_data_subdir")
def test_scifact_corpus_loads_empty_metadata(
    ensure_data_subdir,
    download_scifact,
    source_path,
    read_jsonl,
    load_or_build_lexical_corpus,
    tmp_path,
):
    from cheat_at_search import scifact_data

    ensure_data_subdir.return_value = tmp_path
    download_scifact.return_value = tmp_path
    source_path.return_value = tmp_path / "corpus.jsonl"
    read_jsonl.return_value = pd.DataFrame(
        {
            "doc_id": ["d1"],
            "title": ["Example"],
            "description": ["Example text"],
            "metadata": [{}],
        }
    )
    load_or_build_lexical_corpus.side_effect = lambda corpus, _name: corpus
    scifact_data.__dict__.pop("corpus", None)

    try:
        corpus = scifact_data.corpus
    finally:
        scifact_data.__dict__.pop("corpus", None)

    assert list(corpus["doc_id"]) == ["d1"]
    assert "metadata" not in corpus.columns


@patch("cheat_at_search.scifact_data.load_or_build_lexical_corpus")
@patch("cheat_at_search.scifact_data.download_scifact")
@patch("cheat_at_search.scifact_data.ensure_data_subdir")
def test_scifact_cached_ids_are_normalized(
    ensure_data_subdir,
    download_scifact,
    load_or_build_lexical_corpus,
    tmp_path,
):
    from cheat_at_search import scifact_data

    ensure_data_subdir.return_value = tmp_path
    download_scifact.return_value = tmp_path
    load_or_build_lexical_corpus.side_effect = lambda corpus, _name: corpus
    pd.DataFrame({"query_id": [1], "query": ["example"]}).to_parquet(
        tmp_path / "queries.parquet", index=False
    )
    pd.DataFrame(
        {"query_id": [1], "query": ["example"], "doc_id": [2], "grade": [1]}
    ).to_parquet(tmp_path / "judgments.parquet", index=False)
    pd.DataFrame(
        {"doc_id": [2], "title": ["Example"], "description": ["Text"]}
    ).to_parquet(tmp_path / "corpus.parquet", index=False)
    scifact_data.__dict__.pop("queries", None)
    scifact_data.__dict__.pop("judgments", None)
    scifact_data.__dict__.pop("corpus", None)

    try:
        queries = scifact_data.queries
        judgments = scifact_data.judgments
        corpus = scifact_data.corpus
    finally:
        for name in ("queries", "judgments", "corpus"):
            scifact_data.__dict__.pop(name, None)

    assert queries["query_id"].dtype == object
    assert judgments["query_id"].dtype == object
    assert judgments["doc_id"].dtype == object
    assert corpus["doc_id"].dtype == object
