import os
import tempfile
from unittest.mock import patch

from cheat_at_search.cache import StoredLruCache


@patch("cheat_at_search.cache.stored_cache_path")
def test_stored_lru_cache_basic_usage(mock_cache_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_cache_path.__fspath__.return_value = tmpdir
        cache = StoredLruCache(maxsize=2)
        call_counter = {"count": 0}

        def add_one(value):
            call_counter["count"] += 1
            return value + 1

        cached_add_one = cache(add_one)

        assert cached_add_one(1) == 2
        assert cached_add_one(1) == 2
        assert call_counter["count"] == 1
        assert cache.cache_file is not None
        assert os.path.exists(cache.cache_file)

        reloaded_cache = StoredLruCache(maxsize=2)
        cached_add_one_reloaded = reloaded_cache(add_one)

        assert cached_add_one_reloaded(1) == 2
        assert call_counter["count"] == 1


@patch("cheat_at_search.cache.stored_cache_path")
def test_stored_lru_cache_malformed_pickle_resets_cache(mock_cache_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_cache_path.__fspath__.return_value = tmpdir
        cache = StoredLruCache(maxsize=2)

        def add_one(value):
            return value + 1

        cache(add_one)

        with open(cache.cache_file, "wb") as handle:
            handle.write(b"not-a-pickle")

        cache.cache_loaded = False
        cache.load_cache()

        assert cache.cache == {}
        assert cache.cache_loaded is False
