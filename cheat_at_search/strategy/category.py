from typing import Optional
from searcharray import SearchArray
from cheat_at_search.tokenizers import snowball_tokenizer
from cheat_at_search.strategy.strategy import SearchStrategy
import numpy as np
import pandas as pd

from cheat_at_search.agent.enrich import CachedEnricher, OpenAIEnricher, BatchOpenAIEnricher
from cheat_at_search.model import QueryCategory, QueryCategoryReversed, ProductCategory, \
    QueryCategoryFullyQualified, ProductCategoryFullyQualified


class CategorySearch(SearchStrategy):
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
            cls=QueryCategory
        ))

    def _category(self, query: str) -> QueryCategory:
        """Extract synonyms from the query using an enricher"""
        prompt = f"""
            As a helpful agent, you'll recieve requests from users looking for furniture products.

            Your task is to search with a structured query against a furniture product catalog.

            Here is the users request:

            Return
            * Category - the allowed categories (as listed in schema) for the product. Leave blank
            * SubCategory - the allowed subcategories (as listed in schema) for the product. Leave blank

            {query}
        """

        return self.enricher.enrich(prompt)

    def search(self, query, k=10):
        """Dumb baseline lexical search"""
        bm25_scores = np.zeros(len(self.index))
        structured = self._category(query)
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


# We try to avoid problems by double checking, and being OK giving up
# if we dont get agreement between two LLM calls
class CategorySearchDoubleCheck(SearchStrategy):
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

        self.enricher1 = CachedEnricher(OpenAIEnricher(
            system_prompt="You are a helpful furniture shopping agent that helps users construct search queries.",
            cls=QueryCategory
        ))
        self.enricher2 = CachedEnricher(OpenAIEnricher(
            system_prompt="You are a helpful furniture shopping agent that helps users construct search queries.",
            cls=QueryCategoryReversed
        ), identifier="double_check")

    def _category(self, query: str) -> QueryCategory:
        """Extract synonyms from the query using an enricher"""
        prompt = f"""
            As a helpful agent, you'll recieve requests from users looking for furniture products.

            Your task is to search with a structured query against a furniture product catalog.

            Here is the users request:

            Return
            * Category - the allowed categories (as listed in schema) for the product. Leave blank
            * SubCategory - the allowed subcategories (as listed in schema) for the product. Leave blank

            {query}
        """

        first_check = self.enricher1.enrich(prompt)
        second_check = self.enricher2.enrich(prompt)
        if first_check.category != second_check.category:
            print(f"MISMATCH - Category for {query}: {first_check.category} != {second_check.category}")
            first_check.category = "No Category Fits"
        if first_check.sub_category != second_check.sub_category:
            print(f"SubCategory mismatch for {query}: {first_check.sub_category} != {second_check.sub_category}")
            first_check.sub_category = "No SubCategory Fits"
        return first_check

    def search(self, query, k=10):
        """Dumb baseline lexical search"""
        print("******")
        print(f"Query: {query}")
        bm25_scores = np.zeros(len(self.index))
        structured = self._category(query)
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

        print(f"Cats: {structured.category} / {structured.sub_category}")

        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]

        return top_k, scores


# We notice some realistic exceptions in the data, you
# might argue this is 'overfit' but its also reflective of real
# scenarios we need to share exceptions
class CategorySearchOverfit(SearchStrategy):
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

        self.enricher1 = CachedEnricher(OpenAIEnricher(
            system_prompt="You are a helpful furniture shopping agent that helps users construct search queries.",
            cls=QueryCategory
        ))
        self.enricher2 = CachedEnricher(OpenAIEnricher(
            system_prompt="You are a helpful furniture shopping agent that helps users construct search queries.",
            cls=QueryCategoryReversed
        ), identifier="double_check")

    def _category(self, query: str) -> QueryCategory:
        """Extract synonyms from the query using an enricher"""
        prompt = f"""
            As a helpful agent, you'll recieve requests from users looking for furniture products.

            Your task is to search with a structured query against a furniture product catalog.

            Here is the users request:

            Return
            * Category - the allowed categories (as listed in schema) for the product. Leave blank
            * SubCategory - the allowed subcategories (as listed in schema) for the product. Leave blank

            {query}

            Also here's some strange exceptions to pay attention to:
            * Bistro tables are actually outdoor / patio furniture
            * Outdoor lounge cushions are outdoor / outdoor decor
            * Sometimes storage goes under office furniture. But there are other types of storage
              to pay attention to (like subcategory)
        """

        first_check = self.enricher1.enrich(prompt)
        second_check = self.enricher2.enrich(prompt)
        if first_check.category != second_check.category:
            print(f"MISMATCH - Category for {query}: {first_check.category} != {second_check.category}")
            first_check.category = "No Category Fits"
        if first_check.sub_category != second_check.sub_category:
            print(f"SubCategory mismatch for {query}: {first_check.sub_category} != {second_check.sub_category}")
            first_check.sub_category = "No SubCategory Fits"
        return first_check

    def search(self, query, k=10):
        """Dumb baseline lexical search"""
        print("******")
        print(f"Query: {query}")
        bm25_scores = np.zeros(len(self.index))
        structured = self._category(query)
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

        print(f"Cats: {structured.category} / {structured.sub_category}")

        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]

        return top_k, scores


# Attempt to fill in missing categories in the product data
class CategorySearchFillNans(SearchStrategy):
    def __init__(self, products):

        # ADDED
        # *****
        self.product_enricher = BatchOpenAIEnricher(
            OpenAIEnricher(
                system_prompt="You are a helpful furniture shopping cataloger that maps products to categories and subcategories given metadata about them.",
                cls=ProductCategory
            )
        )

        for idx, product in products.iterrows():
            if pd.isna(product['category hierarchy']):
                task_id = f"product_category_{product['product_id']}"
                category = self._product_category(
                    task_id,
                    product['product_name'], product['product_description'])
                if category is not None:
                    products.at[idx, 'category hierarchy'] = f"{category.category} / {category.sub_category}"
        self.product_enricher.submit(block=True)

        # *****

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

        self.enricher1 = CachedEnricher(OpenAIEnricher(
            system_prompt="You are a helpful furniture shopping agent that helps users construct search queries.",
            cls=QueryCategory
        ))
        self.enricher2 = CachedEnricher(OpenAIEnricher(
            system_prompt="You are a helpful furniture shopping agent that helps users construct search queries.",
            cls=QueryCategoryReversed
        ), identifier="double_check")

    def _product_category(self, product_id: str,
                          product_name: str, product_description: str) -> Optional[ProductCategory]:
        """Extract product category from the product name and description using an enricher"""
        prompt = f"""
            As a helpful agent, you'll receive product metadata and need to categorize it.

            Here is the product metadata:

            Product Name: {product_name}
            Product Description: {product_description}

            Return
            * Category - the allowed categories (as listed in schema) for the product. Leave blank if no category fits.
            * SubCategory - the allowed subcategories (as listed in schema) for the product. Leave blank if no subcategory fits.
        """

        return self.product_enricher.enrich(prompt, task_id=product_id)

    def _category(self, query: str) -> QueryCategory:
        """Extract synonyms from the query using an enricher"""
        prompt = f"""
            As a helpful agent, you'll recieve requests from users looking for furniture products.

            Your task is to search with a structured query against a furniture product catalog.

            Here is the users request:

            Return
            * Category - the allowed categories (as listed in schema) for the product. Leave blank
            * SubCategory - the allowed subcategories (as listed in schema) for the product. Leave blank

            {query}

            Also here's some strange exceptions to pay attention to:
            * Bistro tables are actually outdoor / patio furniture
            * Outdoor lounge cushions are outdoor / outdoor decor
            * Storage cabinets are actually furniture / office furniture
        """

        first_check = self.enricher1.enrich(prompt)
        second_check = self.enricher2.enrich(prompt)
        if first_check.category != second_check.category:
            print(f"MISMATCH - Category for {query}: {first_check.category} != {second_check.category}")
            first_check.category = "No Category Fits"
        if first_check.sub_category != second_check.sub_category:
            print(f"SubCategory mismatch for {query}: {first_check.sub_category} != {second_check.sub_category}")
            first_check.sub_category = "No SubCategory Fits"
        return first_check

    def search(self, query, k=10):
        """Dumb baseline lexical search"""
        print("******")
        print(f"Query: {query}")
        bm25_scores = np.zeros(len(self.index))
        structured = self._category(query)
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

        print(f"Cats: {structured.category} / {structured.sub_category}")

        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]

        return top_k, scores


class CategorySearchFullyQualified(SearchStrategy):

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
            cls=QueryCategoryFullyQualified
        ))

    def _category(self, query: str) -> QueryCategoryFullyQualified:
        """Extract synonyms from the query using an enricher"""
        prompt = f"""
            As a helpful agent, you'll recieve requests from users looking for furniture products.

            Your task is to search with a structured query against a furniture product catalog.

            Here is the users request:

            {query}

            Return the full category / subcategory in the format "Category / SubCategory" from the
            list of allowed categories and subcategories.
        """
        return self.enricher.enrich(prompt)

    def search(self, query, k=10):
        """Dumb baseline lexical search"""
        bm25_scores = np.zeros(len(self.index))
        structured = self._category(query)
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


class CategorySearchFullyQualifiedLabelAll(SearchStrategy):

    def __init__(self, products):

        # ADDED
        # *****
        self.product_enricher = BatchOpenAIEnricher(
            OpenAIEnricher(
                system_prompt="You are a helpful furniture shopping cataloger that maps products to categories and subcategories given metadata about them.",
                cls=ProductCategoryFullyQualified
            )
        )

        for idx, product in products.iterrows():
            task_id = f"product_category_{product['product_id']}"
            category = self._product_category(
                task_id,
                product['product_name'], product['product_description'])
            if isinstance(category, dict):
                print(f"Product {product['product_id']} category enrichment returned a dict, skipping: {category}")
                continue
            if category is not None:
                products.at[idx, 'llm category hierarchy'] = f"{category.category} / {category.sub_category}"
        print("Waiting for batch...")
        self.product_enricher.submit(block=True)

        # *****
        super().__init__(products)
        self.index = products
        self.index['product_name_snowball'] = SearchArray.index(
            products['product_name'], snowball_tokenizer)
        self.index['product_description_snowball'] = SearchArray.index(
            products['product_description'], snowball_tokenizer)

        cat_split = products['llm category hierarchy'].fillna('').str.split("/")

        products['category'] = cat_split.apply(
            lambda x: x[0].strip() if len(x) > 0 else ""
        )
        products['subcategory'] = cat_split.apply(
            lambda x: x[1].strip() if len(x) > 1 else ""
        )
        products['cat_subcat'] = products['category'] + ' / ' + products['subcategory']
        self.index['category_snowball'] = SearchArray.index(
            products['category'], snowball_tokenizer
        )
        self.index['subcategory_snowball'] = SearchArray.index(
            products['subcategory'], snowball_tokenizer
        )
        self.index['cat_subcat_snowball'] = SearchArray.index(
            products['cat_subcat'], snowball_tokenizer
        )

        self.enricher = CachedEnricher(OpenAIEnricher(
            system_prompt="You are a helpful furniture shopping agent that helps users construct search queries.",
            cls=QueryCategoryFullyQualified
        ))

    def _category(self, query: str) -> QueryCategoryFullyQualified:
        """Extract synonyms from the query using an enricher"""
        prompt = f"""
            As a helpful agent, you'll recieve requests from users looking for furniture products.

            Your task is to search with a structured query against a furniture product catalog.

            Here is the users request:

            {query}

            Return the full category / subcategory in the format "Category / SubCategory" from the
            list of allowed categories and subcategories.
        """
        return self.enricher.enrich(prompt)

    def _product_category(self, product_id: str,
                          product_name: str, product_description: str) -> Optional[ProductCategoryFullyQualified]:
        """Extract product category from the product name and description using an enricher"""
        prompt = f"""
            As a helpful agent, you'll receive product metadata and need to categorize it.

            Here is the product metadata:

            Product Name: {product_name}
            Product Description: {product_description}

            Return
            * Category - the allowed categories (as listed in schema) for the product. Leave blank if no category fits.
            * SubCategory - the allowed subcategories (as listed in schema) for the product. Leave blank if no subcategory fits.
        """

        return self.product_enricher.enrich(prompt, task_id=product_id)

    def search(self, query, k=10):
        """Dumb baseline lexical search"""
        bm25_scores = np.zeros(len(self.index))
        structured = self._category(query)
        tokenized = snowball_tokenizer(query)
        for token in tokenized:
            bm25_scores += self.index['product_name_snowball'].array.score(token)
            bm25_scores += self.index['product_description_snowball'].array.score(
                token)

        has_subcategory = False
        has_cat = False
        if structured.sub_category and structured.sub_category != "No SubCategory Fits":
            tokenized_subcategory = snowball_tokenizer(structured.sub_category)
            subcategory_match = np.ones(len(self.index))
            if tokenized_subcategory:
                subcategory_match = self.index['subcategory_snowball'].array.score(tokenized_subcategory) > 0
            bm25_scores[subcategory_match] += 1
            has_subcategory = True

        if structured.category and structured.category != "No Category Fits":
            tokenized_category = snowball_tokenizer(structured.category)
            category_match = np.ones(len(self.index))
            if tokenized_category:
                category_match = self.index['category_snowball'].array.score(tokenized_category) > 0
            bm25_scores[category_match] += 2
            has_cat = True

        if has_cat and has_subcategory:
            tokenized_cat_subcat = snowball_tokenizer(
                structured.category + ' / ' + structured.sub_category)
            cat_subcat_match = np.ones(len(self.index))
            if tokenized_cat_subcat:
                cat_subcat_match = self.index['cat_subcat_snowball'].array.score(tokenized_cat_subcat) > 0
            bm25_scores[cat_subcat_match] += 3

        print("******")
        print(f"Query: {query}")
        print(f"Structured query: {structured.category} / {structured.sub_category}")

        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]

        return top_k, scores
