from cheat_at_search.strategy.strategy import SearchStrategy
from cheat_at_search.agent.search_client import SearchClient
from cheat_at_search.data_dir import ensure_data_subdir
import pickle
from hashlib import md5


cached_results_dir = ensure_data_subdir("cached_agent_search_results")


class ReasoningSearchStrategy(SearchStrategy):
    """A search strategy that uses reasoning to improve search results."""

    def __init__(self, corpus, search_client: SearchClient, prompt: str, cache=True,
                 workers=1):
        super().__init__(corpus, workers=workers)
        self.search_client = search_client
        self.prompt = prompt
        self.cache = None
        self.total_tokens = 0
        if cache:
            system_prompt = self.search_client.system_prompt
            prompt_hash = md5((prompt + system_prompt).encode('utf-8')).hexdigest()[:8]
            corpus_hash = md5(corpus.columns.to_series().astype(str).sum().encode('utf-8')).hexdigest()[:8]
            length = len(corpus)
            self.hash = prompt_hash + "_" + corpus_hash + "_" + str(length)
            self.cache_path = cached_results_dir / f"reasoning_search_cache_{self.hash}.pkl"
            self.cache = {}
            try:
                with open(self.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
                    print(f"Loaded {len(self.cache)} cached search results from", self.cache_path)
            except FileNotFoundError:
                print("No existing cache found at", self.cache_path, "starting fresh.")
                pass

    def search(self, query, k=10):
        if self.cache and query in self.cache:
            return self.cache[query]
        print("----")
        print(f"Searching for: {query}")
        print("----")
        prompt = self.prompt + f"\nSearch query: {query}"
        search_results, total_tokens = self.search_client.search(prompt, return_usage=True)
        self.total_tokens += total_tokens
        top_k = []
        scores = []
        for result in search_results.results[:k]:
            top_k.append(int(result.id))
            scores.append(result.score)

        if self.cache is not None:
            self.cache[query] = (top_k, scores)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.cache, f)
        else:
            print("Not caching")
        return top_k, scores
