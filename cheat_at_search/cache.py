from cheat_at_search.data_dir import DATA_DIR
import os
import pickle

import logging

logger = logging.getLogger(__name__)


class StoredLruCache:
    """An LRU cache decorator that stores its contents in a file."""

    def __init__(self, maxsize=128, cache_file='cache.pkl'):
        self.maxsize = maxsize
        self.cache_file = os.path.join(DATA_DIR, cache_file)
        self.cache = {}
        self.load_cache()

    def load_cache(self):
        """Load the cache from the file."""
        try:
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
        except (FileNotFoundError, EOFError):
            self.cache = {}

    def save_cache(self):
        """Save the cache to the file."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except KeyboardInterrupt:
            logger.warning("Cache saving interrupted. Retrying save before raising exception.")
            self.save_cache()
            raise

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            key = (args, tuple(kwargs.items()))
            if key in self.cache:
                return self.cache[key]
            result = func(*args, **kwargs)
            if len(self.cache) >= self.maxsize:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = result
            self.save_cache()
            return result
        return wrapper
