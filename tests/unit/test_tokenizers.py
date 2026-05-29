from cheat_at_search.tokenizers import snowball_tokenizer, taxonomy_tokenizer, ws_tokenizer, stem_word


def test_stem_word_basic():
    assert stem_word("running") == "run"
    assert stem_word("flies") == "fli"


def test_stem_word_empty():
    assert stem_word("") == ""


def test_snowball_tokenizer_basic():
    result = snowball_tokenizer("Hello, worlds!")
    assert isinstance(result, list)
    assert len(result) > 0


def test_snowball_tokenizer_punctuation_removed():
    result = snowball_tokenizer("Hello, worlds!")
    assert "hello" in result
    assert "," not in result
    assert "!" not in result


def test_snowball_tokenizer_stemming_applied():
    result = snowball_tokenizer("running runners")
    assert "run" in result


def test_snowball_tokenizer_none_returns_empty_string():
    result = snowball_tokenizer(None)
    assert result == ""


def test_snowball_tokenizer_float_returns_empty_string():
    result = snowball_tokenizer(float)
    assert result == ""


def test_snowball_tokenizer_unicode_folding():
    result = snowball_tokenizer("It's a 'test' with - dash")
    assert "'" not in result
    assert "it" in result
    assert "test" in result


def test_snowball_tokenizer_empty_string():
    result = snowball_tokenizer("")
    assert result == []


def test_taxonomy_tokenizer_slash_replaced():
    result = taxonomy_tokenizer("category/subcategory")
    assert isinstance(result, list)
    assert len(result) > 0


def test_taxonomy_tokenizer_none_returns_empty_string():
    result = taxonomy_tokenizer(None)
    assert result == ""


def test_taxonomy_tokenizer_float_returns_empty_string():
    result = taxonomy_tokenizer(float)
    assert result == ""


def test_taxonomy_tokenizer_basic():
    result = taxonomy_tokenizer("Hello, worlds!")
    assert isinstance(result, list)
    assert len(result) > 0


def test_ws_tokenizer_basic():
    result = ws_tokenizer("Hello, worlds!")
    assert isinstance(result, list)
    assert "hello" in result
    assert "worlds" in result


def test_ws_tokenizer_no_stemming():
    result = ws_tokenizer("running")
    assert "running" in result
    assert "run" not in result


def test_ws_tokenizer_none_returns_empty_string():
    result = ws_tokenizer(None)
    assert result == ""


def test_ws_tokenizer_float_returns_empty_string():
    result = ws_tokenizer(float)
    assert result == ""


def test_ws_tokenizer_punctuation_removed():
    result = ws_tokenizer("Hello, worlds!")
    assert "," not in result
    assert "!" not in result


def test_ws_tokenizer_empty_string():
    result = ws_tokenizer("")
    assert result == []
