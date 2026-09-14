import re
from collections.abc import Callable
from typing import Any


def _build_passage_text(row: Any) -> str:
    title = row.get("title")
    description = row.get("description")
    title_text = title.strip() if isinstance(title, str) else ""
    description_text = description.strip() if isinstance(description, str) else ""
    if title_text and description_text:
        return f"{title_text}\n\n{description_text}"
    if title_text:
        return title_text
    return description_text


def _passage_fn_default(row: Any) -> str:
    return _build_passage_text(row)


def _passage_fn_passage_prefix(row: Any) -> str:
    return f"passage: {_build_passage_text(row)}"


def _safe_qualname(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")
    return cleaned or "custom"


def make_passage_fn(document_prefix: str | None = None) -> Callable[[Any], str]:
    """Create a passage function with an optional prefix."""
    if document_prefix is None:
        return _passage_fn_default
    if document_prefix == "passage: ":
        return _passage_fn_passage_prefix

    def passage_fn(row: Any, _document_prefix=document_prefix) -> str:
        return f"{_document_prefix}{_build_passage_text(row)}"

    passage_fn.__qualname__ = f"passage_fn_{_safe_qualname(document_prefix)}"
    return passage_fn


default_passage_fn = make_passage_fn()
