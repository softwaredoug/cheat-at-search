from cheat_at_search.data_dir import ensure_data_subdir
import os
import pickle
import logging
from hashlib import md5
import inspect


logger = logging.getLogger(__name__)


stored_cache_path = ensure_data_subdir('stored_lru_cache')


def get_function_signature_key(func):
    qualname = func.__qualname__
    module = func.__module__
    signature = str(inspect.signature(func))
    unique_string = f"{module}.{qualname}{signature}"
    return unique_string


class StoredLruCache:
    """An LRU cache decorator that stores its contents in a file."""

    def __init__(self, maxsize=128):
        self.maxsize = maxsize
        self.cache = {}
        self.cache_loaded = False
        self.cache_file = None

    def load_cache(self):
        """Load the cache from the file."""
        try:
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
                logger.info(f"Cache loaded from {self.cache_file} with {len(self.cache)} entries.")
            self.cache_loaded = True
        except (FileNotFoundError, EOFError):
            self.cache = {}
            self.cache_loaded = False

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
        signature = get_function_signature_key(func)
        self.cache_file = os.path.join(stored_cache_path, f"{md5(signature.encode()).hexdigest()}.pkl")
        logger.info(f"Using cache file: {self.cache_file}")
        if not self.cache_loaded:
            self.load_cache()

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
