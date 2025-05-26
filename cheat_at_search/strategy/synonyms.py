from searcharray import SearchArray
from cheat_at_search.tokenizers import snowball_tokenizer
from cheat_at_search.strategy.strategy import SearchStrategy
import numpy as np

from cheat_at_search.agent.enrich import CachedEnricher, OllamaEnricher
from cheat_at_search.model import QueryWithSynonyms


class SynonymSearch(SearchStrategy):
    def __init__(self, products):
        super().__init__(products)
        self.index = products
        self.index['product_name_snowball'] = SearchArray.index(
            products['product_name'], snowball_tokenizer)
        self.index['product_description_snowball'] = SearchArray.index(
            products['product_description'], snowball_tokenizer)
        self.enricher = CachedEnricher(OllamaEnricher(
            model="llama3.2",
            system_prompt="You are a helpful AI assistant extracting synonyms from queries.",
            cls=QueryWithSynonyms
        ))

    def _synonyms(self, query: str) -> QueryWithSynonyms:
        """Extract synonyms from the query using an enricher"""
        prompt = f"""
            Extract synonyms from the following query:

            {query}
        """

        return self.enricher.enrich(prompt)

    def search(self, query, k=10):
        """Dumb baseline lexical search"""
        tokenized = snowball_tokenizer(query)
        bm25_scores = np.zeros(len(self.index))
        for token in tokenized:
            bm25_scores += self.index['product_name_snowball'].array.score(token)
            bm25_scores += self.index['product_description_snowball'].array.score(
                token)
        # Boost by each synonym phrase
        synonyms = self._synonyms(query)
        for _, phrases in synonyms.synonyms.items():
            for phrase in phrases:
                tokenized = snowball_tokenizer(phrase)
                bm25_scores += self.index['product_name_snowball'].array.score(tokenized)
                bm25_scores += self.index['product_description_snowball'].array.score(tokenized)
        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]

        return top_k, scores
