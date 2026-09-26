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
