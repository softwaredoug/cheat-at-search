"""Built-in SID-2 document and report tools."""

from typing import Annotated

from sid import DocumentCache, InvalidCharacterRange, parse_rendered_model_facing_id


SNIPPET_FIELD = "content"


def read(
    id: Annotated[
        str,
        "A model-facing document ID or ranged reference, such as 123456#800:1600.",
        "document_id_from_search",
    ],
    agent_state: dict,
) -> str:
    """Read up to 10,000 characters from a document returned by search."""

    cache = agent_state.get("_sid_document_cache")
    if cache is None:
        return "Error executing read: episode document cache is unavailable."

    try:
        model_id, char_range = parse_rendered_model_facing_id(id)

        if not cache.contains_model_facing_id(model_id):
            return "Unknown document ID. Use search or text_search first."

        data_id = cache.to_data_id(model_id)

        if char_range is not None:
            char_range = cache.resolve_char_range(
                data_id, SNIPPET_FIELD, char_range
            )
    except (ValueError, InvalidCharacterRange) as error:
        return f"Error executing read: {error}"

    if char_range is not None:
        start, end = char_range
        char_range = (start, min(end, start + 10_000))
    elif len(cache.get_document(data_id)[SNIPPET_FIELD]) > 10_000:
        char_range = (0, 10_000)

    view = cache.get_single_span_document_view(
        data_id,
        snippet_field=SNIPPET_FIELD,
        snippet_display_span=char_range,
    )
    output = view.render_xml()
    cache.update_seen(view)
    return output


REPORT_TOOL = {
    "type": "function",
    "name": "report_helpful_ids",
    "description": (
        "Report the most helpful document ids from the search results. This is required in "
        "the last step. Each id may be reported at most once; repeated ids are penalized. "
        "Ids may be whole documents ('123456') or ranged references ('123456#800:1600'); a "
        "ranged reference counts as its whole document, so there is no need to strip the "
        "'#start:end' part."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "description": "The ids of the most helpful documents from the search results.",
                "items": {"type": "string"},
            },
        },
        "required": ["ids"],
    },
}

REPORT_TOOL_EXAMPLE = {
    "caption": "Report the most helpful document IDs, ranked best first:",
    "name": "report_helpful_ids",
    "arguments": {"ids": ["document_id_from_search"]},
}


def resolve_reported_ids(
    references: list[str], cache: DocumentCache
) -> list[str]:
    """Map valid, unique model-facing report IDs back to backend IDs."""
    resolved = []
    seen_model_ids = set()
    for reference in references:
        try:
            model_id, _ = parse_rendered_model_facing_id(reference)
        except (TypeError, ValueError):
            continue
        if model_id in seen_model_ids or not cache.contains_model_facing_id(model_id):
            continue
        seen_model_ids.add(model_id)
        resolved.append(cache.to_data_id(model_id))
    return resolved
