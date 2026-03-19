import importlib


def test_doug_blog_data_corpus_loads():
    module = importlib.import_module("cheat_at_search.doug_blog_data")
    corpus = module.corpus
    assert len(corpus) > 0
    assert "doc_id" in corpus.columns
    assert "title" in corpus.columns
    assert "description" in corpus.columns
    assert "publication_date" in corpus.columns
