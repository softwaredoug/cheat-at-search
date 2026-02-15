import tempfile
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel

from cheat_at_search.data_dir import mount
from cheat_at_search.enrich import AutoEnricher, DataframeEnricher
from cheat_at_search.enrich.enrich_client import DebugMetaData, EnrichClient


@pytest.fixture()
def mounted_data_dir():
    data_dir = tempfile.mkdtemp()
    mount(use_gdrive=False, manual_path=data_dir, load_keys=False)
    return data_dir


class RowClassification(BaseModel):
    label: str


class FakeOpenAIEnricher(EnrichClient):
    backend_call_count = 0

    def __init__(self, response_model: BaseModel, model: str, system_prompt: str = None,
                 temperature: float = 0.0, verbosity: str = "low",
                 reasoning_effort: str = "minimal"):
        super().__init__(response_model=response_model)
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.verbosity = verbosity
        self.reasoning_effort = reasoning_effort

    def enrich(self, prompt: str):
        FakeOpenAIEnricher.backend_call_count += 1
        value = prompt.split("::", maxsplit=1)[1]
        return self.response_model(label=value.upper())

    def debug(self, prompt: str):
        return DebugMetaData(
            model=self.model,
            prompt_tokens=1,
            completion_tokens=1,
            reasoning_tokens=0,
            output=self.enrich(prompt),
        )

    def str_hash(self) -> str:
        return "fake-openai-enricher-v1"


@patch("cheat_at_search.enrich.enrich.OpenAIEnricher", FakeOpenAIEnricher)
def test_dataframe_enricher_uses_cached_backend(mounted_data_dir):
    FakeOpenAIEnricher.backend_call_count = 0

    rows = pd.DataFrame(
        {
            "doc_id": [1, 2, 3],
            "text": ["alpha", "beta", "gamma"],
        }
    )

    def prompt_fn(row):
        return f"classify::{row['text']}"

    first_auto = AutoEnricher(
        model="openai/gpt-4.1-nano",
        system_prompt="Classify each row",
        response_model=RowClassification,
    )
    first_df_enricher = DataframeEnricher(
        enricher=first_auto,
        prompt_fn=prompt_fn,
        attrs=["label"],
    )
    first_enriched = first_df_enricher.enrich_all(rows.copy(), workers=1, batch_size=2)

    assert first_enriched["label"].tolist() == ["ALPHA", "BETA", "GAMMA"]
    assert FakeOpenAIEnricher.backend_call_count == 3

    second_auto = AutoEnricher(
        model="openai/gpt-4.1-nano",
        system_prompt="Classify each row",
        response_model=RowClassification,
    )
    second_df_enricher = DataframeEnricher(
        enricher=second_auto,
        prompt_fn=prompt_fn,
        attrs=["label"],
    )
    second_enriched = second_df_enricher.enrich_all(rows.copy(), workers=1, batch_size=2)

    assert second_enriched["label"].tolist() == ["ALPHA", "BETA", "GAMMA"]
    assert FakeOpenAIEnricher.backend_call_count == 3


@patch("cheat_at_search.enrich.enrich.OpenAIEnricher", FakeOpenAIEnricher)
def test_dataframe_enricher_second_call_uses_cache(mounted_data_dir):
    FakeOpenAIEnricher.backend_call_count = 0

    auto = AutoEnricher(
        model="openai/gpt-4.1-nano",
        system_prompt="Classify each row",
        response_model=RowClassification,
    )
    df_enricher = DataframeEnricher(
        enricher=auto,
        prompt_fn=lambda row: f"classify::{row['text']}",
        attrs=["label"],
    )

    first = df_enricher.enrich_one({"doc_id": 1, "text": "alpha"})
    calls_after_first = FakeOpenAIEnricher.backend_call_count
    second = df_enricher.enrich_one({"doc_id": 1, "text": "alpha"})
    calls_after_second = FakeOpenAIEnricher.backend_call_count

    assert first.label == "ALPHA"
    assert second.label == "ALPHA"
    assert calls_after_first == 1
    assert calls_after_second == 1


@pytest.mark.benchmarks
@patch("cheat_at_search.enrich.enrich.OpenAIEnricher", FakeOpenAIEnricher)
def test_dataframe_enricher_benchmark_1000_rows(mounted_data_dir):
    FakeOpenAIEnricher.backend_call_count = 0

    rows = pd.DataFrame(
        {
            "doc_id": list(range(1000)),
            "text": [f"row-{idx}" for idx in range(1000)],
        }
    )

    auto = AutoEnricher(
        model="openai/gpt-4.1-nano",
        system_prompt="Classify each row",
        response_model=RowClassification,
    )
    df_enricher = DataframeEnricher(
        enricher=auto,
        prompt_fn=lambda row: f"classify::{row['text']}",
        attrs=["label"],
    )

    enriched = df_enricher.enrich_all(rows.copy(), workers=8, batch_size=100)

    assert len(enriched) == 1000
    assert enriched["label"].iloc[0] == "ROW-0"
    assert enriched["label"].iloc[-1] == "ROW-999"
    assert FakeOpenAIEnricher.backend_call_count == 1000
