from unittest.mock import patch

import pytest

from cheat_at_search.enrich import AutoDecisionEnricher, AutoEnricher, AutoLLMEnricher


def test_auto_enricher_remains_an_alias_for_auto_llm_enricher():
    assert AutoEnricher is AutoLLMEnricher


@patch("cheat_at_search.enrich.enrich.CachedDecisionEnrichClient")
@patch("cheat_at_search.enrich.enrich.JevEnrichClient")
def test_auto_decision_enricher_routes_one_decision_to_jev(mock_jev_client, mock_cached_client):
    state = {"query": "red chair", "title": "Armchair"}
    question = "Which category matches the query?"
    choices = {"furniture": "Furniture", "lighting": "Lighting"}
    mock_cached_client.return_value.decide.return_value = "furniture"

    enricher = AutoDecisionEnricher(
        choices=choices,
        model="jev/jev-latest",
        probability_threshold=0.75,
        confidence_threshold=0.6,
    )

    result = enricher.decide(state=state, question=question)

    assert result == "furniture"
    mock_jev_client.assert_called_once_with(
        model="jev-latest",
        probability_threshold=0.75,
        confidence_threshold=0.6,
    )
    mock_cached_client.assert_called_once_with(mock_jev_client.return_value)
    mock_cached_client.return_value.decide.assert_called_once_with(
        state=state,
        question=question,
        choices=choices,
        force_refresh=False,
    )


@patch("cheat_at_search.enrich.enrich.CachedDecisionEnrichClient")
@patch("cheat_at_search.enrich.enrich.JevEnrichClient")
def test_auto_decision_enricher_plumbs_score_through_cache(mock_jev_client, mock_cached_client):
    choices = {"furniture": "Furniture", "lighting": "Lighting"}
    score = ({"furniture": 0.8, "lighting": 0.2}, 0.75)
    mock_cached_client.return_value.score.return_value = score
    enricher = AutoDecisionEnricher(choices=choices)

    result = enricher.score(state="chair", question="Which category?")

    assert result == score
    mock_cached_client.return_value.score.assert_called_once_with(
        state="chair",
        question="Which category?",
        choices=choices,
        force_refresh=False,
    )


@patch("cheat_at_search.enrich.enrich.CachedDecisionEnrichClient")
@patch("cheat_at_search.enrich.enrich.JevEnrichClient")
def test_auto_decision_enricher_applies_one_question_to_all_states(mock_jev_client, mock_cached_client):
    decision_client = mock_cached_client.return_value
    decision_client.decide.side_effect = lambda state, question, choices, force_refresh: state["id"]
    enricher = AutoDecisionEnricher(
        choices={"yes": None, "no": None},
    )
    states = [{"id": "first"}, {"id": "second"}]

    decisions = enricher.decide_all(states, questions="Choose a label", workers=2)

    assert decisions == ["first", "second"]
    assert [call.kwargs["question"] for call in decision_client.decide.call_args_list] == [
        "Choose a label",
        "Choose a label",
    ]
    decision_client.save_cache.assert_called_once_with()


@patch("cheat_at_search.enrich.enrich.CachedDecisionEnrichClient")
@patch("cheat_at_search.enrich.enrich.JevEnrichClient")
def test_auto_decision_enricher_supports_per_state_questions(mock_jev_client, mock_cached_client):
    decision_client = mock_cached_client.return_value
    decision_client.decide.side_effect = lambda state, question, choices, force_refresh: question
    enricher = AutoDecisionEnricher(
        choices={"yes": None, "no": None},
    )

    decisions = enricher.decide_all(
        states=[{}, {}],
        questions=["First question", "Second question"],
        workers=1,
    )

    assert decisions == ["First question", "Second question"]


@patch("cheat_at_search.enrich.enrich.CachedDecisionEnrichClient")
@patch("cheat_at_search.enrich.enrich.JevEnrichClient")
def test_auto_decision_enricher_requires_question_per_state(mock_jev_client, mock_cached_client):
    enricher = AutoDecisionEnricher(
        choices={"yes": None, "no": None},
    )

    with pytest.raises(ValueError, match="same length as states"):
        enricher.decide_all(states=[{}, {}], questions=["Only one question"])


@pytest.mark.parametrize("choices", [{}, None])
def test_auto_decision_enricher_requires_choices(choices):
    with pytest.raises(ValueError, match="choices.*at least one"):
        AutoDecisionEnricher(choices=choices)


def test_auto_decision_enricher_requires_a_jev_model():
    with pytest.raises(ValueError, match="requires a Jev model"):
        AutoDecisionEnricher(choices={"yes": None}, model="openai/gpt-4.1-nano")


@pytest.mark.parametrize("threshold_name", ["probability_threshold", "confidence_threshold"])
@patch("cheat_at_search.enrich.enrich.JevEnrichClient")
def test_auto_decision_enricher_validates_thresholds(mock_jev_client, threshold_name):
    with pytest.raises(ValueError, match=threshold_name):
        AutoDecisionEnricher(
            choices={"yes": None},
            **{threshold_name: 1.01},
        )
