from searcharray import SearchArray
from cheat_at_search.tokenizers import snowball_tokenizer
from cheat_at_search.strategy.strategy import SearchStrategy
from cheat_at_search.agent.enrich import OllamaEnricher, OpenAIEnricher, CachedEnricher, BatchOpenAIEnricher
from cheat_at_search.model import EnrichedProduct, StructuredQuery, ProductRoom
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

room_prompt = """
    What room is this product for?

    Given the product name:

    "{product_name}"

    product description:

    "{product_description}"

    product category:

    "{product_category}"

    Notes:
    * Use outdoor for any outdoor spaces
    * Certain items that might be used in multiple rooms (e.g., a chair) should be classified as "any" or "unknown", but
      if its typically used in one room type (ie couch in a living room) then use the typical room type

"""


class EnrichedJustRoomBM25Search(SearchStrategy):
    def __init__(self, products):
        enriched_products = []
        enricher = CachedEnricher(OllamaEnricher(
            model="llama3.2",
            system_prompt="You are a helpful AI assistant enriching product data responding in JSON. ",
            cls=ProductRoom
        ))
        openai_enricher = OpenAIEnricher(
            system_prompt="You are a helpful AI assistant enriching product data responding in JSON. ",
            cls=ProductRoom
        )
        batch_enricher = BatchOpenAIEnricher(
            enricher=openai_enricher,
        )

        progress = tqdm(products.iterrows(),
                        desc="Enriching products", total=len(products))
        count = 0
        for idx, product in progress:
            product_id = product['product_id']
            name = product['product_name']
            description = product['product_description']
            category = product['category hierarchy']
            prompt = room_prompt.format(
                product_name=name, product_description=description,
                product_category=category
            )
            enriched = batch_enricher.enrich(
                prompt,
                task_id=f"room_enrich_{product_id}"
            )
            if count > 10:
                break
            count += 1
        batch_enricher.submit()

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
