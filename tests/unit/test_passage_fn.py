from cheat_at_search.passage_fn import default_passage_fn, make_passage_fn


def test_default_passage_fn_handles_title_and_description():
    assert default_passage_fn({"title": "  Title ", "description": " Description "}) == (
        "Title\n\nDescription"
    )


def test_default_passage_fn_handles_missing_or_non_string_fields():
    assert default_passage_fn({"title": "Title"}) == "Title"
    assert default_passage_fn({"description": "Description"}) == "Description"
    assert default_passage_fn({"title": 123, "description": None}) == ""
    assert default_passage_fn({"title": "  ", "description": " Description "}) == "Description"


def test_make_passage_fn_adds_prefix():
    assert make_passage_fn("document: ")({"title": "Title"}) == "document: Title"
    assert make_passage_fn("passage: ")({"description": "Description"}) == "passage: Description"


def test_make_passage_fn_default_is_default_passage_fn():
    assert make_passage_fn() is default_passage_fn
