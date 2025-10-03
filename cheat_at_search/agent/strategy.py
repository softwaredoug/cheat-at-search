from cheat_at_search.strategy.strategy import SearchStrategy
from cheat_at_search.agent.search_client import SearchClient
from cheat_at_search.data_dir import ensure_data_subdir
import pickle
from hashlib import md5


cached_results_dir = ensure_data_subdir("cached_agent_search_results")


class ReasoningSearchStrategy(SearchStrategy):
    """A search strategy that uses reasoning to improve search results."""

    def __init__(self, products, search_client: SearchClient, prompt: str, cache=True):
        super().__init__(products)
        self.search_client = search_client
        self.prompt = prompt
        self.cache = None
        if cache:
            system_prompt = self.search_cliest.system_prompt
            self.prompt_hash = md5((prompt + system_prompt).encode('utf-8')).hexdigest()[:8]
            self.cache_path = cached_results_dir / f"reasoning_search_cache_{self.prompt_hash}.pkl"
            self.cache = {}
            try:
                with open(self.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
            except FileNotFoundError:
                pass

    def search(self, query, k=10):
        if self.cache and query in self.cache:
            return self.cache[query]
        print("----")
        print(f"Searching for: {query}")
        print("----")
        prompt = self.prompt + f"\nSearch query: {query}"
        search_results = self.search_client.search(prompt)
        top_k = []
        scores = []
        for result in search_results.results[:k]:
            top_k.append(int(result.id))
            scores.append(result.score)
        if self.cache is not None:
            self.cache[query] = (top_k, scores)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.cache, f)
        return top_k, scores
