from cheat_at_search.agent.cached_enrich_client import CachedEnrichClient
from cheat_at_search.agent.enrich_client import EnrichClient, DebugMetaData
from pydantic import BaseModel
from pydantic import Field
import pytest
from cheat_at_search.data_dir import mount
import tempfile
from typing import Optional

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
