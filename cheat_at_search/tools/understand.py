from cheat_at_search.agent.openai_agent import OpenAIAgent
from typing import Literal
from pydantic import BaseModel, Field
from time import perf_counter
from cheat_at_search.data_dir import ensure_data_subdir
from cheat_at_search.logger import log_to_stdout
import pickle


Properties = Literal[
    "same-color",
    "same-size",
    "same-material",
    "same-brand",
    "same-price",
    "same-warranty",
    "same-item-type",
]


cache_dir = ensure_data_subdir("understanders")
logger = log_to_stdout("cheat_at_search.understanders")


class QueryToProductProperties(BaseModel):
    """A collection of query or product properties."""

    properties: list[Properties] = Field(
        ...,
        description="A list of properties extracted from the query or product that they share.",
    )


def make_query_and_product_understanders(model: str = "openai/gpt-4.1-mini"):
    agent = OpenAIAgent(tools=[], model=model, response_model=QueryToProductProperties)

    cache = {}
    try:
        with open(cache_dir / "query_product_understander_cache.pkl", "rb") as f:
            cache = pickle.load(f)
            logger.info(
                f"Loaded query-product understander cache with {len(cache)} entries."
            )
    except FileNotFoundError:
        cache = {}
        logger.info(
            "No existing cache found for query-product understander. Starting fresh."
        )
    except Exception as e:
        cache = {}
        logger.error(f"Error loading query-product understander cache: {e}")

    last_cache_save = perf_counter()

    def query_to_product(
        query: str, title: str, description: str
    ) -> QueryToProductProperties:
        """List the shared properties of the product and query."""
        nonlocal last_cache_save
        cache_key = (model, query, title, description)
        if cache_key in cache:
            return cache[cache_key]
        user_prompt = f"List the properties the product and query share: \n\nProduct Title: {title}\nProduct Description: {description}\nUser Query: {query}\n"
        inputs = [
            {
                "role": "system",
                "content": "You are a query product matching agent. Given a query, product title, and description, list properties product has that the query is asking about.",
            },
            {"role": "user", "content": user_prompt},
        ]
        resp, _, _ = agent.chat(inputs=inputs, return_usage=True)
        response = resp.output_parsed
        cache[cache_key] = response.properties
        if perf_counter() - last_cache_save > 60:
            # Save cache to disk every minute
            try:
                with open(
                    cache_dir / "query_product_understander_cache.pkl", "wb"
                ) as f:
                    pickle.dump(cache, f)
                    logger.info(
                        f"Saved query-product understander cache with {len(cache)} entries."
                    )
                    last_cache_save = perf_counter()
            except Exception as e:
                logger.error(f"Error saving query-product understander cache: {e}")

        # dedup
        property_set = set()
        for prop in response.properties:
            if prop not in property_set:
                property_set.add(prop)

        return property_set

    return query_to_product


if __name__ == "__main__":
    understander = make_query_and_product_understanders()
    result = understander(
        query="Looking for durable size 27 red backpack with multiple compartments",
        title="Durable Red Size 27 Hiking Backpack",
        description="This red backpack is made from high-quality materials and features multiple compartments for organization.",
    )
    print(result)
