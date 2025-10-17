def rerank_esci(search_esci, query):
    docs = search_esci(keywords=query, field_to_search='product_name', operator='and', locale='us', top_k=10)
    return [doc['id'] for doc in docs]

