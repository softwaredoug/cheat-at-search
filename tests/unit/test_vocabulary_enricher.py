from unittest.mock import patch

import numpy as np
import pytest

from cheat_at_search.enrich import VocabularyEnricher, VocabularyResponse


class FakeEmbeddingModel:
    vectors = {
        "red": [1.0, 0.0],
        "blue": [0.0, 1.0],
        "crimson": [0.9, 0.1],
    }

    def encode(self, values, convert_to_numpy=True):
        return np.array([self.vectors[value] for value in values])


class FakeAutoEnricher:
    def __init__(self, **kwargs):
        self.response = VocabularyResponse(value="crimson")
        self.call_count = 0

    def enrich(self, prompt, force_refresh=False):
        self.call_count += 1
        return self.response


@patch("cheat_at_search.enrich.vocabulary.load_model", return_value=FakeEmbeddingModel())
@patch("cheat_at_search.enrich.vocabulary.AutoEnricher", FakeAutoEnricher)
def test_vocabulary_enricher_resolves_closest_item(_load_model):
    enricher = VocabularyEnricher(
        model="openai/test",
        system_prompt="Extract a color",
        vocabulary=["red", "blue"],
    )

    assert enricher.resolve("crimson chair") == "red"


@patch("cheat_at_search.enrich.vocabulary.load_model", return_value=FakeEmbeddingModel())
@patch("cheat_at_search.enrich.vocabulary.AutoEnricher", FakeAutoEnricher)
def test_vocabulary_enricher_returns_none_for_empty_output(_load_model):
    enricher = VocabularyEnricher(
        model="openai/test",
        system_prompt="Extract a color",
        vocabulary=["red", "blue"],
    )
    enricher.enricher.response = None

    assert enricher.resolve("chair") is None


@patch("cheat_at_search.enrich.vocabulary.load_model", return_value=FakeEmbeddingModel())
@patch("cheat_at_search.enrich.vocabulary.AutoEnricher", FakeAutoEnricher)
def test_vocabulary_enricher_can_force_refresh(_load_model):
    enricher = VocabularyEnricher(
        model="openai/test",
        system_prompt="Extract a color",
        vocabulary=["red", "blue"],
    )

    assert enricher.resolve("chair") == "red"
    assert enricher.enricher.call_count == 1
    assert enricher.resolve("chair") == "red"
    assert enricher.enricher.call_count == 2
    assert enricher.resolve("chair", force_refresh=True) == "red"
    assert enricher.enricher.call_count == 3


def test_vocabulary_enricher_requires_vocabulary():
    with pytest.raises(ValueError, match="at least one"):
        VocabularyEnricher("openai/test", "Extract a color", [])
