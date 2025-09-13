from cheat_at_search.data_dir import key_for_provider
from cheat_at_search.wands_data import products
from cheat_at_search.agent.mcp import serve_tool
from cheat_at_search.tokenizers import snowball_tokenizer
from typing import List, Dict
from openai import OpenAI
import numpy as np
import time


openai_key = key_for_provider("openai")

client = OpenAI(
    api_key=openai_key,
)


def search_products(query: str, top_k: int = 5) -> List[Dict]:
    """
    Search for furniture products

    Args:
        query: The search query string.
        top_k: The number of top results to return.

    Returns:
        A list of dictionaries containing product information.
    """
    print("Searching for:", query, "top_k:", top_k)
    query_tokens = snowball_tokenizer(query)
    scores = np.zeros(len(products))
    for token in query_tokens:
        scores += products['product_name_snowball'].array.score(token) * 10
        scores += products['description_snowball'].array.score(token)

    top_k_indices = np.argsort(scores)[-top_k:][::-1]
    scores = scores[top_k_indices]
    top_products = products.iloc[top_k_indices]
    top_products['score'] = scores

    # Serialize back in JSON
    try:
        results = []
        print("Getting results")
        for id, row in top_products.iterrows():
            results.append({
                'id': id,
                'product_name': row['product_name'],
                'product_description': row['product_description'],
                'score': row['score']
            })

        return results
    except Exception as e:
        print("!!!")
        print("Error serializing results:", e)
        raise e


def test_serving_search_tool():
    thread = serve_tool(fn=search_products, name="search_products",
                        description="Search for products by product name and description",
                        port=8000)
    time.sleep(1)
    thread.join(1)
