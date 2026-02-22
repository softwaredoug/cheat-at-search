import pytest
import numpy as np

pytest.importorskip("sentence_transformers", reason="sentence_transformers not installed")
from sentence_transformers import SentenceTransformer

from cheat_at_search.enrich.entities import Entities


@pytest.fixture(scope="module")
def model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# --- empty / single entity ---

def test_empty_returns_nothing(model):
    entities = Entities(model)
    assert entities.most_similar("Steve Jobs") == []


def test_len_empty(model):
    entities = Entities(model)
    assert len(entities) == 0


def test_exact_match(model):
    entities = Entities(model)
    entities.add_entity("Steve Jobs")
    assert entities.most_similar("Steve Jobs") == ["Steve Jobs"]


def test_len_after_add(model):
    entities = Entities(model)
    entities.add_entity("Steve Jobs")
    entities.add_entity("Bill Gates")
    assert len(entities) == 2


# --- fuzzy / alternate forms ---
# Note: MiniLM is a semantic model, not character-level. Character typos
# score lower than semantic variants. Thresholds below reflect actual scores.

def test_typo_match(model):
    # "Steve Joooobs" ~0.66 cosine with "Steve Jobs"
    entities = Entities(model)
    entities.add_entity("Steve Jobs")
    result = entities.most_similar("Steve Joooobs", threshold=0.6)
    assert result == ["Steve Jobs"]


def test_alternate_company_form(model):
    # "Microsoft Corp" ~0.82 cosine with "Microsoft"
    entities = Entities(model)
    entities.add_entity("Microsoft")
    result = entities.most_similar("Microsoft Corp", threshold=0.8)
    assert result == ["Microsoft"]


def test_misspelled_name(model):
    # "Barak Obama" ~0.89 cosine with "Barack Obama"
    entities = Entities(model)
    entities.add_entity("Barack Obama")
    result = entities.most_similar("Barak Obama", threshold=0.85)
    assert result == ["Barack Obama"]


# --- multiple entities ---

def test_multiple_entities_exact(model):
    entities = Entities(model)
    entities.add_entity("Steve Jobs")
    entities.add_entity("Bill Gates")
    entities.add_entity("Elon Musk")
    assert entities.most_similar("Bill Gates") == ["Bill Gates"]


def test_multiple_entities_fuzzy(model):
    # "Elon Musk CEO" is semantically close to "Elon Musk" at ~0.85
    entities = Entities(model)
    entities.add_entity("Steve Jobs")
    entities.add_entity("Bill Gates")
    entities.add_entity("Elon Musk")
    result = entities.most_similar("Elon Musk CEO", threshold=0.8)
    assert result == ["Elon Musk"]


def test_top_k_limits_results(model):
    entities = Entities(model)
    for name in ["cat", "dog", "fish", "bird", "snake"]:
        entities.add_entity(name)
    result = entities.most_similar("cat", top_k=1, threshold=0.0)
    assert len(result) == 1


def test_results_ordered_by_similarity(model):
    entities = Entities(model)
    entities.add_entity("machine learning")
    entities.add_entity("gardening tips")
    entities.add_entity("deep learning")
    result = entities.most_similar("neural networks", top_k=3, threshold=0.0)
    # machine learning / deep learning should both rank above gardening
    gardening_idx = result.index("gardening tips")
    assert gardening_idx == len(result) - 1


# --- threshold behaviour ---

def test_no_match_below_threshold(model):
    entities = Entities(model)
    entities.add_entity("Steve Jobs")
    # "cat" should not be similar to "Steve Jobs" at a high threshold
    result = entities.most_similar("cat", threshold=0.95)
    assert result == []


def test_match_found_at_lower_threshold(model):
    entities = Entities(model)
    entities.add_entity("machine learning")
    # semantically related but not identical
    result = entities.most_similar("artificial intelligence", threshold=0.5)
    assert "machine learning" in result


def test_exact_match_always_exceeds_high_threshold(model):
    entities = Entities(model)
    entities.add_entity("quantum computing")
    result = entities.most_similar("quantum computing", threshold=0.99)
    assert result == ["quantum computing"]
