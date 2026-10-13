from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from typesafe_sdk import Choice

from cheat_at_search.enrich.jev_enrich_client import JevEnrichClient


def choice_answer(choice="furniture", probabilities=None, confidence=0.8):
    if probabilities is None:
        probabilities = {"furniture": 0.8, "lighting": 0.2}
    return SimpleNamespace(choice=choice, probabilities=probabilities, confidence=confidence)


@patch("cheat_at_search.enrich.jev_enrich_client.key_for_provider", return_value="test-typesafe-key")
@patch("cheat_at_search.enrich.jev_enrich_client.TypeSafeClient")
def test_jev_enrich_client_submits_state_question_and_choices(
    mock_type_safe_client, mock_key_for_provider
):
    state = {"query": "red chair", "product": {"title": "Armchair"}}
    question = "Which product category best matches this query?"
    choices = {"furniture": "Furniture items", "lighting": "Lighting items"}
    sdk_client = mock_type_safe_client.return_value
    sdk_client.system_one.return_value = SimpleNamespace(
        choices={"decision": choice_answer()}
    )

    client = JevEnrichClient(model="jev-latest")

    result = client.decide(state=state, question=question, choices=choices)

    assert result == "furniture"
    mock_key_for_provider.assert_called_once_with("typesafe")
    mock_type_safe_client.assert_called_once_with(
        model="jev-latest",
        api_key="test-typesafe-key",
    )
    sdk_client.system_one.assert_called_once_with(
        state=state,
        questions={"decision": Choice(instructions=question, criteria=choices)},
    )


def test_jev_enrich_client_score_returns_probabilities_and_confidence():
    sdk_client = Mock()
    sdk_client.system_one.return_value = SimpleNamespace(
        choices={
            "decision": choice_answer(
                probabilities={"furniture": 0.75, "lighting": 0.25},
                confidence=0.4,
            )
        }
    )
    client = JevEnrichClient(
        client=sdk_client,
        probability_threshold=0.9,
        confidence_threshold=0.8,
    )

    result = client.score(
        state="red chair",
        question="Which category?",
        choices={"furniture": None, "lighting": None},
    )

    assert result == ({"furniture": 0.75, "lighting": 0.25}, 0.4)


def test_jev_enrich_client_accepts_an_injected_sdk_client():
    sdk_client = Mock()
    sdk_client.system_one.return_value = SimpleNamespace(
        choices={"decision": choice_answer("yes", {"yes": 1.0, "no": 0.0})}
    )
    client = JevEnrichClient(client=sdk_client)

    result = client.decide(
        state="The item is a red chair.",
        question="Is this furniture?",
        choices={"yes": None, "no": None},
    )

    assert result == "yes"
    assert client.str_hash()


def test_jev_enrich_client_hash_includes_model():
    default_client = JevEnrichClient(client=Mock())
    custom_client = JevEnrichClient(model="custom-jev", client=Mock())
    threshold_client = JevEnrichClient(
        probability_threshold=0.5,
        client=Mock(),
    )

    assert default_client.str_hash() != custom_client.str_hash()
    assert default_client.str_hash() != threshold_client.str_hash()


def test_jev_enrich_client_accepts_choice_when_both_thresholds_are_met():
    sdk_client = Mock()
    sdk_client.system_one.return_value = SimpleNamespace(
        choices={"decision": choice_answer(confidence=0.7)}
    )
    client = JevEnrichClient(
        client=sdk_client,
        probability_threshold=0.8,
        confidence_threshold=0.7,
    )

    result = client.decide("state", "question", {"furniture": None, "lighting": None})

    assert result == "furniture"


def test_jev_enrich_client_rejects_choice_below_probability_threshold():
    sdk_client = Mock()
    sdk_client.system_one.return_value = SimpleNamespace(
        choices={"decision": choice_answer(probabilities={"furniture": 0.79, "lighting": 0.21})}
    )
    client = JevEnrichClient(client=sdk_client, probability_threshold=0.8)

    result = client.decide("state", "question", {"furniture": None, "lighting": None})

    assert result is None


def test_jev_enrich_client_rejects_choice_below_confidence_threshold():
    sdk_client = Mock()
    sdk_client.system_one.return_value = SimpleNamespace(
        choices={"decision": choice_answer(confidence=0.69)}
    )
    client = JevEnrichClient(client=sdk_client, confidence_threshold=0.7)

    result = client.decide("state", "question", {"furniture": None, "lighting": None})

    assert result is None


@patch("cheat_at_search.enrich.jev_enrich_client.TypeSafeClient")
@pytest.mark.parametrize("threshold_name", ["probability_threshold", "confidence_threshold"])
@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), "0.5"])
def test_jev_enrich_client_validates_threshold_range(mock_type_safe_client, threshold_name, threshold):
    with pytest.raises(ValueError, match=threshold_name):
        JevEnrichClient(**{threshold_name: threshold})
