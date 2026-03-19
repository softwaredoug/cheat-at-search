import importlib


def _is_stripped(value):
    if isinstance(value, str):
        return value == value.strip()
    return True


def test_wands_data_categories_trimmed():
    module = importlib.import_module("cheat_at_search.wands_data")
    corpus = module.corpus
    assert "category" in corpus.columns
    assert "sub_category" in corpus.columns
    assert corpus["category"].apply(_is_stripped).all()
    assert corpus["sub_category"].apply(_is_stripped).all()
