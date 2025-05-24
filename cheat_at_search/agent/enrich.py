from ollama import chat
from cheat_at_search.logger import log_to_stdout
from cheat_at_search.model import Query, Product, UnderstoodQuery, EnrichedProduct
from typing import Optional


logger = log_to_stdout(logger_name="query_parser")


def enrich_query(query_cls, prompt, model="llama3.2") -> Optional[Query]:
    try:
        response = chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format=query_cls.model_json_schema()
        )
        if response.message.content:
            return query_cls.model_validate_json(response.message.content)
    except Exception as e:
        logger.error(f"Error parsing query '{prompt}': {str(e)}")
        # Return a default object with keywords in case of errors
        raise e
    return None


def enrich_product(product_cls, prompt, model="llama3.2") -> Optional[Product]:
    try:
        response = chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format=product_cls.model_json_schema()
        )
        if response.message.content:
            return product_cls.model_validate_json(response.message.content)
    except Exception as e:
        logger.error(f"Error parsing query '{prompt}': {str(e)}")
        # Return a default object with keywords in case of errors
        raise e
    return None


if __name__ == "__main__":
    query = "wooden dining table for small kitchen"
    product = "wooden dining table"
    prompt = f"""
    You are a search query analyzer for a furniture e-commerce site that extracts structured information from user queries.

    Given the search query: "{query}"

    Extract the following information:
    1. Material - What material is the furniture made of? (e.g., wood, metal, glass, leather)
    2. Color - What color is mentioned? (e.g., blue, red, black)
    3. Furniture Type - What type of furniture is being searched for? (e.g., chair, table, sofa)
    4. Room - What room is mentioned, if any? (e.g., living room, bedroom, kitchen)
    5. Dimensions - Any dimensions mentioned (e.g., "72 inch", "king size")

    If any information is not present, use an empty string or empty list as appropriate.
    """
    enriched = enrich_query(UnderstoodQuery, prompt)
