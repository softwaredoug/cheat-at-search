from cheat_at_search.enrich import CachedEnrichClient, EnrichClient, DebugMetaData
from pydantic import BaseModel
from pydantic import Field
import pytest
from cheat_at_search.data_dir import mount
import tempfile
from typing import Optional
import json
from concurrent.futures import ThreadPoolExecutor
from json import dump as real_json_dump
from threading import Event
from unittest.mock import patch

from cheat_at_search.logger import log_to_stdout

logger = log_to_stdout(logger_name="test_enrich_cache_client")


class ColorEnrich(BaseModel):
    color: str = Field(
        ...,
        description="The color to enrich",
    )


@pytest.fixture(scope="module")
def mounted_data_dir():
    """
    Fixture to mount the data directory before running tests.
    """
    data_dir = tempfile.mkdtemp()
    print(f"Mounting data directory at: {data_dir}")
    mount(use_gdrive=False, manual_path=data_dir, load_keys=False)
    return True


class MockEnrichClient(EnrichClient):

    def __init__(self, colors):
        self.colors = colors
        self.call_count = 0

    def enrich(self, prompt: str) -> Optional[BaseModel]:
        color = self.colors[self.call_count % len(self.colors)]
        self.call_count += 1
        return ColorEnrich(color=color)

    def debug(self, prompt: str) -> Optional[DebugMetaData]:
        return DebugMetaData(model="test-model", prompt_tokens=10, completion_tokens=5, reasoning_tokens=0, output=ColorEnrich(color="blue"))

    def str_hash(self) -> str:
        return "test-hash"

    @property
    def response_model(self):
        return ColorEnrich


def test_cached_enrich_client(mounted_data_dir):
    client = MockEnrichClient(colors=["red", "blue", "green"])
    cached_client = CachedEnrichClient(client)

    colors_first_pass = []
    for i in range(5):
        result = cached_client.enrich(f"What color is this? {i}")
        colors_first_pass.append(result.color)

    cached_client.save_cache()

    client2 = MockEnrichClient(colors=["yellow", "purple", "orange"])  # thes ecolors should be ignored
    cached_client_loaded = CachedEnrichClient(client2)
    colors_second_pass = []
    for i in range(5):
        result = cached_client_loaded.enrich(f"What color is this? {i}")
        colors_second_pass.append(result.color)
    assert colors_first_pass == colors_second_pass, "Cached results do not match after reloading cache"


def test_cached_enrich_client_can_force_refresh(mounted_data_dir):
    client = MockEnrichClient(colors=["red", "blue"])
    cached_client = CachedEnrichClient(client)

    assert cached_client.enrich("What color?").color == "red"
    assert cached_client.enrich("What color?").color == "red"
    assert cached_client.enrich("What color?", force_refresh=True).color == "blue"
    assert cached_client.enrich("What color?").color == "blue"
    assert client.call_count == 2


@patch("cheat_at_search.enrich.cached_enrich_client.json.dump")
def test_save_cache_uses_snapshot_and_atomic_replace(mock_dump, mounted_data_dir):
    cached_client = CachedEnrichClient(MockEnrichClient(colors=["red"]))
    cached_client.cache = {"before": ColorEnrich(color="red").model_dump_json()}
    old_cache = '{"old": "cache"}'
    with open(cached_client.cache_file, "w") as f:
        f.write(old_cache)

    dump_started = Event()
    continue_dump = Event()

    def delayed_dump(cache_snapshot, f):
        dump_started.set()
        if not continue_dump.wait(timeout=5):
            raise TimeoutError("Timed out waiting to continue cache dump")
        real_json_dump(cache_snapshot, f)

    mock_dump.side_effect = delayed_dump
    with ThreadPoolExecutor(max_workers=1) as executor:
        save_future = executor.submit(cached_client.save_cache)
        assert dump_started.wait(timeout=5)
        try:
            with open(cached_client.cache_file) as f:
                assert f.read() == old_cache
            cached_client.cache["during-save"] = ColorEnrich(color="blue").model_dump_json()
        finally:
            continue_dump.set()
        assert save_future.result(timeout=5) is True

    with open(cached_client.cache_file) as f:
        saved_cache = json.load(f)
    assert saved_cache == {"before": ColorEnrich(color="red").model_dump_json()}
    assert not list(cached_client.cache_path.glob(".*.tmp"))


@patch("cheat_at_search.enrich.cached_enrich_client.json.dump", side_effect=OSError("disk full"))
def test_cache_save_failure_preserves_existing_file_and_enrichment(mock_dump, mounted_data_dir):
    cached_client = CachedEnrichClient(MockEnrichClient(colors=["red"]))
    old_cache = '{"old": "cache"}'
    with open(cached_client.cache_file, "w") as f:
        f.write(old_cache)
    cached_client.last_cache_save_time = 0

    result = cached_client.enrich("What color?")

    assert result.color == "red"
    with open(cached_client.cache_file) as f:
        assert f.read() == old_cache
    assert not list(cached_client.cache_path.glob(".*.tmp"))
