import importlib
import time


def test_doug_blog_data_corpus_loads():
    start = time.monotonic()
    module = importlib.import_module("cheat_at_search.doug_blog_data")
    after_import = time.monotonic()
    corpus = module.corpus
    after_corpus = time.monotonic()
    print(f"doug_blog_data import: {after_import - start:.3f}s")
    print(f"doug_blog_data corpus access: {after_corpus - after_import:.3f}s")
    print(f"doug_blog_data total: {after_corpus - start:.3f}s")
    assert len(corpus) > 0
    assert "doc_id" in corpus.columns
    assert "title" in corpus.columns
    assert "description" in corpus.columns
    assert "publication_date" in corpus.columns
