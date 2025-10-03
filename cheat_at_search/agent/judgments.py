from pydantic import BaseModel, Field
from typing import List
from cheat_at_search.wands_data import labeled_query_products
import pandas as pd


class HumanEvaluation(BaseModel):
    """A human judgment of a search result for a given user query."""
    user_query: str = Field(..., description="The original user search query")
    human_label: str = Field(..., description="The human judgment label for the search results")
    product_name: str = Field(..., description="The name of the product judged")
    product_description: str = Field(..., description="The description of the product judged")


def get_human_judgments(user_query: str) -> List[HumanEvaluation]:
    """Get a sample of human judgments for a given user query (the ground truth you're evaluated against).

       It's ok to use this, its not cheating!

       Returns list of human evaluations
    """
    K = 10
    labeled = labeled_query_products.loc[labeled_query_products['query'] == user_query]
    if len(labeled) == 0:
        return []
    relevant = labeled[labeled['label'] == 'Exact']
    irrelevant = labeled[labeled['label'] == 'Irrelevant']
    # Get 3 relevant
    relevant = relevant.sample(min(3, len(relevant)), random_state=42)
    # Get 3 irrelevant
    irrelevant = irrelevant.sample(min(3, len(irrelevant)), random_state=42)
    # Get the rest Partial
    partial = labeled[labeled['label'] == 'Partial']
    partial = partial.sample(min(K - len(relevant) - len(irrelevant), len(partial)), random_state=42)

    labeled = pd.concat([relevant, irrelevant, partial]).sample(frac=1, random_state=42)

    results: List[HumanEvaluation] = []
    for item in labeled.to_dict(orient='records'):
        results.append(HumanEvaluation(user_query=user_query,
                                       human_label=item['label'],
                                       product_name=item['product_name'],
                                       product_description=item['product_description']))

    return results
