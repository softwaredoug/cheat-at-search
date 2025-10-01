from cheat_at_search.strategy.strategy import SearchStrategy
from cheat_at_search.agent.search_client import SearchClient


class ReasoningSearchStrategy(SearchStrategy):
    """A search strategy that uses reasoning to improve search results."""

    def __init__(self, products, search_client: SearchClient, prompt):
        super().__init__(products)
        self.search_client = search_client
        self.prompt = prompt

    def search(self, query, k=10):
        prompt = self.prompt + f"\nSearch query: {query}"
        search_results = self.search_client.search(prompt)
        top_k = []
        scores = []
        for result in search_results.results[:k]:
            top_k.append(int(result.id))
            if result.relevance == 'exact':
                scores.append(1.0)
            elif result.relevance == 'partial':
                scores.append(0.5)
            else:
                scores.append(0.0)
        return top_k, scores
