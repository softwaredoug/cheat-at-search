from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from cheat_at_search.data_dir import mount
from cheat_at_search.enrich.enrich_client import DecisionEnrichClient
from cheat_at_search.enrich.cached_decision_enrich_client import CachedDecisionEnrichClient


@pytest.fixture
def mounted_data_dir(tmp_path):
    mount(use_gdrive=False, manual_path=str(tmp_path), load_keys=False)
    return tmp_path


class FakeDecisionEnrichClient(DecisionEnrichClient):
    def __init__(self, probabilities=None, confidence_score=0.8,
                 client_hash="fake-decision-client", probability_threshold=0.0,
                 confidence_threshold=0.0):
        super().__init__(
            probability_threshold=probability_threshold,
            confidence_threshold=confidence_threshold,
        )
        self.probabilities = probabilities or {"furniture": 0.8, "lighting": 0.2}
        self.confidence_score = confidence_score
        self.client_hash = client_hash
        self.call_count = 0

    def score(self, state, question, choices, force_refresh=False):
        self.call_count += 1
        return self.probabilities, self.confidence_score

    def str_hash(self):
        return self.client_hash


def test_cached_decision_client_caches_and_persists_decisions(mounted_data_dir):
    client = FakeDecisionEnrichClient()
    cached_client = CachedDecisionEnrichClient(client)
    request = {
        "state": {"query": "red chair"},
        "question": "Which category?",
        "choices": {"furniture": "Furniture", "lighting": "Lighting"},
    }

    expected_score = ({"furniture": 0.8, "lighting": 0.2}, 0.8)
    assert cached_client.score(**request) == expected_score
    assert cached_client.score(**request) == expected_score
    assert cached_client.decide(**request) == "furniture"
    assert client.call_count == 1
    assert cached_client.save_cache()

    reloaded_client = FakeDecisionEnrichClient(
        probabilities={"lighting": 0.9, "furniture": 0.1},
        confidence_score=0.9,
    )
    reloaded_cache = CachedDecisionEnrichClient(reloaded_client)
    assert reloaded_cache.score(**request) == expected_score
    assert reloaded_cache.decide(**request) == "furniture"
    assert reloaded_client.call_count == 0


def test_cached_decision_client_caches_none_results(mounted_data_dir):
    client = FakeDecisionEnrichClient(
        probabilities={"x": 0.6, "other": 0.4},
        confidence_score=0.4,
        probability_threshold=0.5,
        confidence_threshold=0.7,
    )
    cached_client = CachedDecisionEnrichClient(client)
    request = {
        "state": "unclassifiable",
        "question": "Which category?",
        "choices": {"x": None, "other": None},
    }

    assert cached_client.decide(**request) is None
    assert cached_client.decide(**request) is None
    assert client.call_count == 1
    assert cached_client.save_cache()

    reloaded_client = FakeDecisionEnrichClient(
        probabilities={"x": 1.0, "other": 0.0},
        confidence_score=1.0,
        probability_threshold=0.5,
        confidence_threshold=0.7,
    )
    reloaded_cache = CachedDecisionEnrichClient(reloaded_client)
    assert reloaded_cache.decide(**request) is None
    assert reloaded_client.call_count == 0


def test_cached_decision_client_key_is_canonical_and_includes_request_inputs(mounted_data_dir):
    client = FakeDecisionEnrichClient()
    cached_client = CachedDecisionEnrichClient(client)
    choices = {"furniture": "Furniture", "lighting": None}

    assert cached_client.decide(
        state={"query": "chair", "title": "Armchair"},
        question="Which category?",
        choices=choices,
    ) == "furniture"
    assert cached_client.decide(
        state={"title": "Armchair", "query": "chair"},
        question="Which category?",
        choices={"lighting": None, "furniture": "Furniture"},
    ) == "furniture"
    assert client.call_count == 1

    cached_client.decide(state={"query": "lamp"}, question="Which category?", choices=choices)
    cached_client.decide(state={"query": "chair"}, question="Which material?", choices=choices)
    cached_client.decide(
        state={"query": "chair"},
        question="Which category?",
        choices={"furniture": "Furniture", "decor": None},
    )
    assert client.call_count == 4


def test_cached_decision_client_can_force_refresh(mounted_data_dir):
    client = FakeDecisionEnrichClient()
    cached_client = CachedDecisionEnrichClient(client)
    request = {"state": "chair", "question": "Which category?", "choices": {"furniture": None}}

    assert cached_client.decide(**request) == "furniture"
    client.probabilities = {"furniture": 0.2, "decor": 0.8}
    assert cached_client.decide(**request) == "furniture"
    assert cached_client.decide(**request, force_refresh=True) == "decor"
    assert cached_client.decide(**request) == "decor"
    assert client.call_count == 2


def test_cached_decision_client_uses_delegate_configuration_hash(mounted_data_dir):
    first = CachedDecisionEnrichClient(FakeDecisionEnrichClient(client_hash="jev-model-a-thresholds"))
    second = CachedDecisionEnrichClient(FakeDecisionEnrichClient(client_hash="jev-model-b-thresholds"))

    assert first.cache_file != second.cache_file


def test_cached_decision_client_key_includes_thresholds(mounted_data_dir):
    request = {"state": "chair", "question": "Which category?", "choices": {"furniture": None}}
    first = CachedDecisionEnrichClient(FakeDecisionEnrichClient(probability_threshold=0.2))
    second = CachedDecisionEnrichClient(FakeDecisionEnrichClient(probability_threshold=0.3))

    assert first.decision_key(**request) != second.decision_key(**request)


def test_cached_decision_client_rejects_non_json_state(mounted_data_dir):
    cached_client = CachedDecisionEnrichClient(FakeDecisionEnrichClient())

    with pytest.raises(TypeError):
        cached_client.decide(
            state=object(),
            question="Which category?",
            choices={"furniture": None},
        )


def test_cached_decision_client_coalesces_concurrent_identical_requests(mounted_data_dir):
    decision_started = Event()
    allow_decision_to_finish = Event()

    class BlockingDecisionClient(FakeDecisionEnrichClient):
        def score(self, state, question, choices, force_refresh=False):
            self.call_count += 1
            decision_started.set()
            assert allow_decision_to_finish.wait(timeout=5)
            return self.probabilities, self.confidence_score

    client = BlockingDecisionClient()
    cached_client = CachedDecisionEnrichClient(client)
    request = {"state": "chair", "question": "Which category?", "choices": {"furniture": None}}

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(cached_client.decide, **request)
        assert decision_started.wait(timeout=5)
        second = executor.submit(cached_client.decide, **request)
        allow_decision_to_finish.set()

    assert first.result(timeout=5) == "furniture"
    assert second.result(timeout=5) == "furniture"

    assert client.call_count == 1
