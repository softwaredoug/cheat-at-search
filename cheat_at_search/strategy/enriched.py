from searcharray import SearchArray
from cheat_at_search.tokenizers import snowball_tokenizer
from cheat_at_search.strategy.strategy import SearchStrategy
from cheat_at_search.agent.enrich import enrich_query, enrich_product
from cheat_at_search.model import EnrichedProduct, UnderstoodQuery
import numpy as np
from tqdm import tqdm


query_prompt = """
    You are a search query analyzer for a furniture e-commerce site that extracts structured information from user queries.

    Given the search query: "{query}"

    Extract the following information:
    1. Material - What material is the furniture made of? (e.g., wood, metal, glass, leather)
    2. Color - What color is mentioned? (e.g., blue, red, black)
    3. Furniture Type - What type of furniture is being searched for? (e.g., chair, table, sofa)
    4. Room - What room is mentioned, if any? (e.g., living room, bedroom, kitchen)
    5. Dimensions - Any dimensions mentioned (e.g., "72 inch", "king size")

    If any information is not present, use an empty string or empty list as appropriate.

    Examples:
    - "wooden dining table for small kitchen" → material: "wooden", color: "", furniture_type: "dining table", room: "kitchen", dimensions: ["small"]
    - "blue leather sofa for living room" → material: "leather", color: "blue", furniture_type: "sofa", room: "living room", dimensions: []
    - "72 inch oak bookshelf" → material: "oak", color: "", furniture_type: "bookshelf", room: "", dimensions: ["72 inch"]
"""

product_prompt = """
    You are a product analyzer for a furniture e-commerce site that extracts structured information from product names and descriptions.
    Given the product name:

    "{product_name}"

    product description:

    "{product_description}"

    product_category:

    "{product_category}"

    Extract the following information:
    1. Material - What material is the furniture made of? (e.g., wood, metal, glass, leather)
    2. Color - What color is the furniture? (e.g., blue, red, black)
    3. Furniture Type - What type of furniture is being described? (e.g., chair, table, sofa)
    4. Room - What room would this be used for (allowed rooms are "living room", "bedroom", "kitchen", "dining room", "office", "bathroom", "outdoor", "hallway", "entryway", "garage", "basement", "attic", "any", "unknown")
    5. Dimensions - Any dimensions mentioned (e.g., "72 inch", "king size")
"""


class EnrichedBM25Search(SearchStrategy):
    def __init__(self, products):
        enriched_products = []
        for idx, product in tqdm(products.iterrows(), desc="Enriching products", total=len(products)):
            name = product['product_name']
            description = product['product_description']
            category = product['category hierarchy']
            prompt = product_prompt.format(
                product_name=name, product_description=description,
                product_category=category
            )
            enriched_product = enrich_product(
                EnrichedProduct, prompt)
            if enriched_product:
                enriched_products.append(
                    enriched_product.dict(exclude_unset=False)
                )
                enriched_product.id = product['product_id']
                enriched_product.name = name
                enriched_product.description = description
                enriched_product.category = category
                enriched_product.classification = product['product_class']
                print(enriched_product.name)
                print(enriched_product.description)
                print(enriched_product.category)
                print("ROOM -- ", enriched_product.room)
        import pdb; pdb.set_trace()

        super().__init__(enriched_products)
        self.index = enriched_products
        self.index['product_name_snowball'] = SearchArray.index(
            products['product_name'], snowball_tokenizer)
        self.index['product_description_snowball'] = SearchArray.index(
            products['product_description'], snowball_tokenizer)

    def search(self, query, k=10):
        """Dumb baseline lexical search"""
        tokenized = snowball_tokenizer(query)
        bm25_scores = np.zeros(len(self.index))
        for token in tokenized:
            bm25_scores += self.index['product_name_snowball'].array.score(token)
            bm25_scores += self.index['product_description_snowball'].array.score(
                token)
        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]
        return top_k, scores
