from searcharray import SearchArray
from cheat_at_search.tokenizers import snowball_tokenizer
from cheat_at_search.strategy.strategy import SearchStrategy
import numpy as np

from cheat_at_search.agent.enrich import CachedEnricher, OpenAIEnricher
from cheat_at_search.model import StructuredQuery


class StructuredSearch(SearchStrategy):
    def __init__(self, products):
        super().__init__(products)
        self.index = products
        self.index['product_name_snowball'] = SearchArray.index(
            products['product_name'], snowball_tokenizer)
        self.index['product_description_snowball'] = SearchArray.index(
            products['product_description'], snowball_tokenizer)

        cat_split = products['category hierarchy'].fillna('').str.split("/")

        products['category'] = cat_split.apply(
            lambda x: x[0].strip() if len(x) > 0 else ""
        )
        products['subcategory'] = cat_split.apply(
            lambda x: x[1].strip() if len(x) > 1 else ""
        )
        self.index['category_snowball'] = SearchArray.index(
            products['category'], snowball_tokenizer
        )
        self.index['subcategory_snowball'] = SearchArray.index(
            products['subcategory'], snowball_tokenizer
        )

        self.enricher = CachedEnricher(OpenAIEnricher(
            system_prompt="You are a helpful furniture shopping agent that helps users construct search queries.",
            cls=StructuredQuery
        ))

    def _structured(self, query: str) -> StructuredQuery:
        """Extract synonyms from the query using an enricher"""
        prompt = f"""
            As a helpful agent, you'll recieve requests from users looking for furniture products.

            Your task is to search with a structured query against a furniture product catalog.

            Here is the users request:

            Return
            * Category - the allowed categories (as listed in schema) for the product. Leave blank

            {query}
        """

        return self.enricher.enrich(prompt)

    def search(self, query, k=10):
        """Dumb baseline lexical search"""
        bm25_scores = np.zeros(len(self.index))
        structured = self._structured(query)
        tokenized = snowball_tokenizer(query)
        for token in tokenized:
            bm25_scores += self.index['product_name_snowball'].array.score(token)
            bm25_scores += self.index['product_description_snowball'].array.score(
                token)

        if structured.sub_category and structured.sub_category != "No SubCategory Fits":
            tokenized_subcategory = snowball_tokenizer(structured.sub_category)
            subcategory_match = np.ones(len(self.index))
            if tokenized_subcategory:
                subcategory_match = self.index['subcategory_snowball'].array.score(tokenized_subcategory) > 0
            bm25_scores[subcategory_match] += 1

        if structured.category and structured.category != "No Category Fits":
            tokenized_category = snowball_tokenizer(structured.category)
            category_match = np.ones(len(self.index))
            if tokenized_category:
                category_match = self.index['category_snowball'].array.score(tokenized_category) > 0
            bm25_scores[category_match] += 2
        else:
            print("No category or subcategory specified, returning all results.")

        print("******")
        print(f"Query: {query}")
        print(f"Structured query: {structured.category} / {structured.sub_category}")

        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]

        return top_k, scores
