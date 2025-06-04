from searcharray import SearchArray
from cheat_at_search.tokenizers import snowball_tokenizer
from cheat_at_search.strategy.strategy import SearchStrategy
import numpy as np

from cheat_at_search.agent.enrich import CachedEnricher, OpenAIEnricher
from cheat_at_search.model import SpellingCorrectedQuery


class SpellingCorrectedSearch(SearchStrategy):
    def __init__(self, products):
        super().__init__(products)
        self.index = products
        self.index['product_name_snowball'] = SearchArray.index(
            products['product_name'], snowball_tokenizer)
        self.index['product_description_snowball'] = SearchArray.index(
            products['product_description'], snowball_tokenizer)
        self.enricher = CachedEnricher(OpenAIEnricher(
            system_prompt="You are a helpful AI assistant that very lightly spell-checks furniture e-commerce queries.",
            cls=SpellingCorrectedQuery
        ))

    def _corrected(self, query: str) -> SpellingCorrectedQuery:
        """Extract synonyms from the query using an enricher"""
        prompt = f"""
            Take this query and correct any obvious spelling mistakes.

            {query}

            * Dont add hyphens (ie "anti scratch" not "anti-scratch")
            * Dont correct stylized product names (ie "merlyn" not "merlin", "oller" not "oiler/oller")
        """

        return self.enricher.enrich(prompt)

    def search(self, query, k=10):
        """Dumb baseline lexical search"""
        bm25_scores = np.zeros(len(self.index))
        corrected = self._corrected(query)
        different = corrected.corrected_keywords.lower().split() != query.lower().split()
        asterisk = "*" if different else ""
        if different:
            print(f"Query: {query} -> Corrected: {corrected.corrected_keywords}{asterisk}")
        tokenized = snowball_tokenizer(corrected.corrected_keywords)
        for token in tokenized:
            bm25_scores += self.index['product_name_snowball'].array.score(token)
            bm25_scores += self.index['product_description_snowball'].array.score(
                token)
        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]

        return top_k, scores


class SpellingCorrectedSearch2(SearchStrategy):
    def __init__(self, products):
        super().__init__(products)
        self.index = products
        self.index['product_name_snowball'] = SearchArray.index(
            products['product_name'], snowball_tokenizer)
        self.index['product_description_snowball'] = SearchArray.index(
            products['product_description'], snowball_tokenizer)
        self.enricher = CachedEnricher(OpenAIEnricher(
            system_prompt="You are a helpful AI assistant that very lightly spell-checks furniture e-commerce queries.",
            cls=SpellingCorrectedQuery
        ))

    def _corrected(self, query: str) -> SpellingCorrectedQuery:
        """Extract synonyms from the query using an enricher"""
        prompt = f"""

        Your job is to:
        - Correct **only clear and obvious typos**
        - Avoid changing the number of words or their structure
        - Preserve known stylized brand or product names (e.g., "merlyn", "oller", "kohen")
        - Do not change proper names
        - Do not add hyphens, even when grammatically correct (e.g., use "anti scratch", not "anti-scratch")
        - Do not merge or split known product compound words (e.g., "loveseat" should remain "loveseat")

        Only make a correction if it clearly improves readability without altering the intended meaning or structure.

        Here is the query:

        {query}

        Provide only the corrected query, or the original if no changes are needed.
        """

        return self.enricher.enrich(prompt)

    def search(self, query, k=10):
        """Dumb baseline lexical search"""
        bm25_scores = np.zeros(len(self.index))
        corrected = self._corrected(query)
        different = corrected.corrected_keywords.lower().split() != query.lower().split()
        asterisk = "*" if different else ""
        if different:
            print(f"Query: {query} -> Corrected: {corrected.corrected_keywords}{asterisk}")
        tokenized = snowball_tokenizer(corrected.corrected_keywords)
        for token in tokenized:
            bm25_scores += self.index['product_name_snowball'].array.score(token)
            bm25_scores += self.index['product_description_snowball'].array.score(
                token)
        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]

        return top_k, scores


class SpellingCorrectedSearch3(SearchStrategy):
    def __init__(self, products):
        super().__init__(products)
        self.index = products
        self.index['product_name_snowball'] = SearchArray.index(
            products['product_name'], snowball_tokenizer)
        self.index['product_description_snowball'] = SearchArray.index(
            products['product_description'], snowball_tokenizer)
        self.enricher = CachedEnricher(OpenAIEnricher(
            system_prompt="You are a helpful AI assistant that very lightly spell-checks furniture e-commerce queries.",
            cls=SpellingCorrectedQuery
        ))

    def _corrected(self, query: str) -> SpellingCorrectedQuery:
        """Extract synonyms from the query using an enricher"""
        prompt = f"""

        First, enerate a list of 100 known Wayfair furniture stylized product names that look like misspelled words (ie merlyn, ollie, etc) in a python list. Call that the exception list (ie don't spellcheck these)

        Your job is to:
        - Correct **only clear and obvious typos** not in the exception list
        - Avoid changing the number of words or their structure
        - Preserve known stylized brand or product names (e.g., "merlyn", "oller", "kohen", "brendon" and others in the exception list)
        - Do not change proper names
        - Do not add or remove hyphens, even when grammatically correct (e.g., use "anti scratch", not "anti-scratch")
        - Do not merge or split known product compound words (e.g., "loveseat" should remain "loveseat", "bed frame" should remain "bed frame" not "bedframe", "bed side table" should remain "bed side table", not "bedside table", etc)

        Only make a correction if it clearly improves readability without altering the intended meaning or structure.

        Here is the query:

        {query}

        Provide only the corrected query, or the original if no changes are needed.
        """

        return self.enricher.enrich(prompt)

    def search(self, query, k=10):
        """Dumb baseline lexical search"""
        bm25_scores = np.zeros(len(self.index))
        corrected = self._corrected(query)
        different = corrected.corrected_keywords.lower().split() != query.lower().split()
        asterisk = "*" if different else ""
        if different:
            print(f"Query: {query} -> Corrected: {corrected.corrected_keywords}{asterisk}")
        tokenized_orig = snowball_tokenizer(query)
        for token in tokenized_orig:
            bm25_scores += self.index['product_name_snowball'].array.score(token)
            bm25_scores += self.index['product_description_snowball'].array.score(
                token)
        tokenized = snowball_tokenizer(corrected.corrected_keywords)
        for token in tokenized:
            bm25_scores += self.index['product_name_snowball'].array.score(token)
            bm25_scores += self.index['product_description_snowball'].array.score(
                token)
        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]

        return top_k, scores
