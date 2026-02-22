import numpy as np

from cheat_at_search.enrich.entities import Entities


class DummyModel:
    def encode(self, text: str, **kwargs) -> np.ndarray:
        return np.array([float(len(text))])


def assert_embeddings_match_names(entities: Entities) -> None:
    embedding_count = 0 if entities._embeddings is None else len(entities._embeddings)
    assert embedding_count == len(entities.names)


def test_repr_includes_entity_strings() -> None:
    entities = Entities(DummyModel())
    entities.add(["Steve Jobs", "Bill Gates"])
    assert repr(entities) == "Entities(names=['Steve Jobs', 'Bill Gates'])"
    assert_embeddings_match_names(entities)


def test_add_single_string_skips_existing_name() -> None:
    entities = Entities(DummyModel())
    entities.add("Steve Jobs")
    entities.add("Steve Jobs")
    assert len(entities) == 1
    assert_embeddings_match_names(entities)


def test_contains_returns_true_for_existing_name() -> None:
    entities = Entities(DummyModel())
    entities.add("Steve Jobs")
    assert "Steve Jobs" in entities
    assert_embeddings_match_names(entities)


def test_contains_returns_false_for_missing_name() -> None:
    entities = Entities(DummyModel())
    entities.add("Steve Jobs")
    assert "Bill Gates" not in entities
    assert_embeddings_match_names(entities)
