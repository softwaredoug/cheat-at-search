# server.py
from typing import List, Dict
from mcp.server.fastmcp import FastMCP, Tool
from cheat_at_search.wands_data import products
from cheat_at_search.data_dir import key_for_provider
from cheat_at_search.logger import log_to_stdout
from searcharray import SearchArray
import Stemmer
import string
import numpy as np
import pyngrok


logger = log_to_stdout("mcp")


stemmer = Stemmer.Stemmer('english', maxCacheSize=0)

fold_to_ascii = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-", u"'''\"\"--")])
punct_trans = str.maketrans({key: ' ' for key in string.punctuation})
all_trans = {**fold_to_ascii, **punct_trans}


def stem_word(word):
    return stemmer.stemWord(word)


def snowball_tokenizer(text):
    if text is float:
        return ''
    if text is None:
        return ''
    text = text.translate(all_trans).replace("'", " ")
    split = text.lower().split()
    return [stem_word(token)
            for token in split]


# Create a backend index
mcp = FastMCP("search-server",
              instructions="This MCP can search for products by product_name and description fields in a product catalog.",
              stateless_http=True)


products['product_name_snowball'] = SearchArray.index(products['product_name'],
                                                      tokenizer=snowball_tokenizer)

products['description_snowball'] = SearchArray.index(products['product_description'],
                                                     tokenizer=snowball_tokenizer)


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


def serve_tool(fn=None, name: str = "search_products",
               description: str = "Search for products by product name and description",
               port=8000):
    if fn is None:
        fn = search_products
    tool = Tool.from_function(fn, name=name, description=description)
    mcp.register_tool(tool)
    ngrok_key = key_for_provider('ngrok')
    pyngrok.ngrok.set_auth_token(ngrok_key)
    public_url = pyngrok.ngrok.connect(port, bind_tls=True).public_url
    print(" * Ngrok public URL:", public_url)

    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
