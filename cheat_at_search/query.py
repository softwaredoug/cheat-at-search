from ollama import chat
from cheat_at_search.wands_data import queries
from cheat_at_search.logger import log_to_stdout
from typing import Optional
from model.query import UnderstoodQuery




def parse_query(query: str, model: str = "llama3.2") -> Optional[UnderstoodQuery]:
    """
    Parse a search query to extract structured information using Ollama's structured output.

    Args:
        query: The search query to parse
        model: The Ollama model to use (default: llama3)

    Returns:
        UnderstoodQuery object containing the extracted information and keywords
    """
    logger = log_to_stdout(logger_name="query_parser")
    logger.info(f"Parsing query: {query}")

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

    Examples:
    - "wooden dining table for small kitchen" → material: "wooden", color: "", furniture_type: "dining table", room: "kitchen", dimensions: ["small"]
    - "blue leather sofa for living room" → material: "leather", color: "blue", furniture_type: "sofa", room: "living room", dimensions: []
    - "72 inch oak bookshelf" → material: "oak", color: "", furniture_type: "bookshelf", room: "", dimensions: ["72 inch"]
    """

    try:
        response = chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format=UnderstoodQuery.model_json_schema()
        )
        if response.message.content:
            return UnderstoodQuery.model_validate_json(response.message.content)


    except Exception as e:
        logger.error(f"Error parsing query '{query}': {str(e)}")
        # Return a default object with keywords in case of errors
        return UnderstoodQuery(keywords=query.split())

# 0. Discovering the query space (how do users think about queries? How do they structure the attributes?)
# 1. Mapping the product space to the query space
# 2. Similarity for each query attribute. Tools:
#   * Flat tagging
#   * Taxonomy
#   * Vector search

if __name__ == "__main__":
    log_to_stdout("cheat_at_search.wands_data")
    query_df = queries()
    for query in query_df['query']:
        understood_query = parse_query(query)
        if not understood_query:
            print(f"Failed to parse query '{query}'")
            continue
        if understood_query.material:
            print("QUERY", query, "MATERIAL", understood_query.material)
