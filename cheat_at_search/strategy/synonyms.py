from searcharray import SearchArray
from cheat_at_search.tokenizers import snowball_tokenizer
from cheat_at_search.strategy.strategy import SearchStrategy
import numpy as np

from cheat_at_search.agent.enrich import CachedEnricher, OllamaEnricher, OpenAIEnricher
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
            system_prompt="You are a helpful AI assistant extracting synonyms from queries.",
            cls=QueryWithSynonyms
        ))

    def _synonyms(self, query: str) -> QueryWithSynonyms:
        """Extract synonyms from the query using an enricher"""
        prompt = f"""
            Extract synonyms from the following query that will help us find relevant products for the query.

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
        print(synonyms)
        for mapping in synonyms.synonyms:
            for phrase in mapping.synonyms:
                tokenized = snowball_tokenizer(phrase)
                bm25_scores += self.index['product_name_snowball'].array.score(tokenized)
                bm25_scores += self.index['product_description_snowball'].array.score(tokenized)
        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]

        return top_k, scores


class AlternateLabelSearch(SearchStrategy):
    def __init__(self, products):
        super().__init__(products)
        self.index = products
        self.index['product_name_snowball'] = SearchArray.index(
            products['product_name'], snowball_tokenizer)
        self.index['product_description_snowball'] = SearchArray.index(
            products['product_description'], snowball_tokenizer)
        self.enricher = CachedEnricher(OpenAIEnricher(
            system_prompt="You are a helpful AI assistant extracting synonyms from queries.",
            cls=QueryWithSynonyms
        ))

    def _synonyms(self, query: str) -> QueryWithSynonyms:
        """Extract synonyms from the query using an enricher"""
        prompt = f"""
            I define a STRICT alternate label where two phrases share 99% of the meaning. IE USA ~ United States of America. Some examples include:

            * Compound variants (bedside == "bed side")
            * Acronyms (LED == Light Emitting Diode)
            * Spelling variants (grey vs gray)
            * Abberviations (refrigerator == fridge, in = "inches", ft = "feet" in the right context)

            Some examples that are synonyms, but not alternate labels:
            * Synonyms (couch != sofa, table != desk, chair != stool)
            * Hypernyms (couch != furniture, sofa != furniture)
            * Hyponyms (couch != chair, sofa != chair)

            Now map the phrases of the query below to their alternate labels. Its ok to return no alternate labels, as
            we care about strict matches only.

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
        for mapping in synonyms.synonyms:
            original_phrase = mapping.phrase
            original_tokenized = snowball_tokenizer(original_phrase)
            for synonym in mapping.synonyms:
                tokenized = snowball_tokenizer(synonym)
                if tokenized == original_tokenized:
                    # Skip the original phrase, as it is already counted
                    continue
                if not tokenized:
                    continue
                print(f"Mapping {original_phrase} -> {synonym}")
                if len(tokenized) == 1:
                    tokenized = tokenized[0]
                bm25_scores += self.index['product_name_snowball'].array.score(tokenized)
                bm25_scores += self.index['product_description_snowball'].array.score(tokenized)
        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]

        return top_k, scores


class HyponymSearch(SearchStrategy):
    def __init__(self, products):
        super().__init__(products)
        self.index = products
        self.index['product_name_snowball'] = SearchArray.index(
            products['product_name'], snowball_tokenizer)
        self.index['product_description_snowball'] = SearchArray.index(
            products['product_description'], snowball_tokenizer)
        self.enricher = CachedEnricher(OpenAIEnricher(
            system_prompt="You are a helpful AI assistant extracting synonyms from queries.",
            cls=QueryWithSynonyms
        ))

    def _synonyms(self, query: str) -> QueryWithSynonyms:
        """Extract synonyms from the query using an enricher"""
        prompt = f"""
            Extract hyponyms from the following query that will help us find relevant products for the query.

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
        for mapping in synonyms.synonyms:
            original_phrase = mapping.phrase
            original_tokenized = snowball_tokenizer(original_phrase)
            for synonym in mapping.synonyms:
                tokenized = snowball_tokenizer(synonym)
                if tokenized == original_tokenized:
                    # Skip the original phrase, as it is already counted
                    continue
                if not tokenized:
                    continue
                print(f"Mapping {original_phrase} -> {synonym}")
                if len(tokenized) == 1:
                    tokenized = tokenized[0]
                bm25_scores += self.index['product_name_snowball'].array.score(tokenized)
                bm25_scores += self.index['product_description_snowball'].array.score(tokenized)
        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]

        return top_k, scores


class HypernymSearch(SearchStrategy):
    def __init__(self, products):
        super().__init__(products)
        self.index = products
        self.index['product_name_snowball'] = SearchArray.index(
            products['product_name'], snowball_tokenizer)
        self.index['product_description_snowball'] = SearchArray.index(
            products['product_description'], snowball_tokenizer)
        self.enricher = CachedEnricher(OpenAIEnricher(
            system_prompt="You are a helpful AI assistant extracting synonyms from queries.",
            cls=QueryWithSynonyms
        ))

    def _synonyms(self, query: str) -> QueryWithSynonyms:
        """Extract synonyms from the query using an enricher"""
        prompt = f"""
            Extract hypernyms from the following query that will help us find relevant products for the query.

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
        for mapping in synonyms.synonyms:
            original_phrase = mapping.phrase
            original_tokenized = snowball_tokenizer(original_phrase)
            for synonym in mapping.synonyms:
                tokenized = snowball_tokenizer(synonym)
                if tokenized == original_tokenized:
                    # Skip the original phrase, as it is already counted
                    continue
                if not tokenized:
                    continue
                print(f"Mapping {original_phrase} -> {synonym}")
                if len(tokenized) == 1:
                    tokenized = tokenized[0]
                bm25_scores += self.index['product_name_snowball'].array.score(tokenized)
                bm25_scores += self.index['product_description_snowball'].array.score(tokenized)
        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]

        return top_k, scores
