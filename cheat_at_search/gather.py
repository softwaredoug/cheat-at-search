from cheat_at_search.wands_data import products, labeled_query_products, queries
from cheat_at_search.model.product import ItemType, BrandedTerms, ProductRoom, Material
from cheat_at_search.model.query import QueryClassification
from cheat_at_search.agent.enrich import AutoEnricher, ProductEnricher
import logging
import os
from cheat_at_search.data_dir import ensure_data_subdir

logger = logging.getLogger(__name__)


item_type_enricher = AutoEnricher(
    model="gpt-4.1-mini",
    system_prompt="You are a helpful furniture, hardware, and home-goods ecommerce shopping assistant that understands furniture products",
    output_cls=ItemType
)

branded_terms_enricher = AutoEnricher(
    model="gpt-4.1-mini",
    system_prompt="You are a helpful furniture, hardware, and home-goods ecommerce shopping assistant that understands furniture products",
    output_cls=BrandedTerms
)


room_enricher = AutoEnricher(
    model="gpt-4.1-mini",
    system_prompt="You are a helpful furniture, hardware, and home-goods ecommerce shopping assistant that understands furniture products",
    output_cls=ProductRoom
)

material_enricher = AutoEnricher(
    model="gpt-4.1-mini",
    system_prompt="You are a helpful furniture, hardware, and home-goods ecommerce shopping assistant that understands furniture products",
    output_cls=Material
)


def get_item_type_prompt(product: dict) -> str:
    prompt = f"""
What is the item type of this product? Remove all branding and give the item type only.

For 'item_type' Use 'no item type matches' if no listed item type matches the item.
For 'item_type_unconstrained' just extract any item type

Here's the product to classify:

Product Name -- {product['product_name']}
        """
    return prompt


def get_branded_terms_prompt(product: dict) -> str:
    prompt = f"""
I am going to give you a product name, you extract any branded terms (product lines, marketing terms, brand names, etc) mentioned.

Product Name: {product['product_name']}
Product Description: {product['product_description']}

Extract any branded terms (collection names, product lines, marketing terms, brand names, etc) mentioned.

Do not include item types, materials, colors, etc.
        """
    return prompt


def get_room_prompt(product: dict) -> str:
    prompt = f"""
I am going to give you a furniture e-commerce product.

You tell me which of the listed room it belongs to. Or if ambiguous, could fit in multiple rooms, or unclear, return 'No Room Fits'

Default to 'No Room Fits' unless given compelling evidence.

Rugs can go in any room - they should get 'No Room Fits'
Hardware can go in any room - they should get 'No Room Fits'
Most decor can go in any room - they should get 'No Room Fits'
If multiple rooms are mentioned - they should get 'No Room Fits'

Product Name: {product['product_name']}
Description: {product['product_description']}
        """
    return prompt


def get_material_prompt(product: dict) -> str:
    prompt = f"""
I am going to give you a furniture e-commerce product.

You tell me which of the listed materials it is made of in a list. If no materials are mentioned, return an empty list

Product Name: {product['product_name']}
Description: {product['product_description']}
    """
    return prompt


classification_enricher = AutoEnricher(
    model="gpt-4.1-nano",
    system_prompt="Your task is to create novel, never seen before, furniture, home goods, or hardware classification that best fit a search query. ",
    output_cls=QueryClassification
)


def get_prompt_fully_qualified(query):
    prompt = f"""

        Some inspiration on what these look like is at the bottom.

        Here is the users request:

        {query}

        Product classifications might look like:

        'Furniture / Living Room Furniture / Coffee Tables & End Tables / Coffee Tables'
        'Décor & Pillows / Decorative Pillows & Blankets / Throw Pillows'
        'Furniture / Bedroom Furniture / Dressers & Chests'
        'Outdoor / Outdoor & Patio Furniture / Patio Furniture Sets / Patio Conversation Sets'
        'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Bathroom Vanities / All Bathroom Vanities'
        'Lighting / Wall Lights / Bathroom Vanity Lighting'
        'Kitchen & Tabletop / Kitchen Organization / Food Storage & Canisters'
        'School Furniture and Supplies / School Furniture / School Chairs & Seating / Stackable Chairs',
        'Baby & Kids / Toddler & Kids Bedroom Furniture / Kids Beds',

        If you feel inspired, return many unique values in a list. Be creative. Cast a wide net with related but diverse categories.

        Return empty list if no clear classification could be inferred, the query is not clearly a furniture query

        """

    return prompt


def fully_classified(query):
    prompt = get_prompt_fully_qualified(query)
    return classification_enricher.enrich(prompt)


def get_top_col_vals(labeled_query_products, column, no_fit_label, cutoff=0.8):
    # Get relevant products per query
    top_products = labeled_query_products[labeled_query_products['grade'] == 2]

    # Aggregate top categories
    categories_per_query_ideal = top_products.groupby('query')[column].value_counts().reset_index()

    # Get as percentage of all categories for this query
    top_cat_proportion = categories_per_query_ideal.groupby(['query', column]).sum() / categories_per_query_ideal.groupby('query').sum()
    top_cat_proportion = top_cat_proportion.drop(columns=column).reset_index()

    # Only look at cases where the category is > 0.8
    top_cat_proportion = top_cat_proportion[top_cat_proportion['count'] > cutoff]
    top_cat_proportion[column].fillna(no_fit_label, inplace=True)
    ground_truth_cat = top_cat_proportion
    # Give No Category Fits to all others without dominant category
    ground_truth_cat = ground_truth_cat.merge(queries, how='right', on='query')[['query', column, 'count']]
    ground_truth_cat[column].fillna(no_fit_label, inplace=True)
    return ground_truth_cat


def enrich_all(products, labeled_query_products):
    item_type = ProductEnricher(
        enricher=item_type_enricher,
        prompt_fn=get_item_type_prompt
    )
    branded_terms = ProductEnricher(
        enricher=branded_terms_enricher,
        prompt_fn=get_item_type_prompt,
    )
    room = ProductEnricher(
        enricher=room_enricher,
        prompt_fn=get_room_prompt
    )
    material = ProductEnricher(
        enricher=material_enricher,
        prompt_fn=get_material_prompt
    )
    item_type.batch_all(products)
    branded_terms.batch_all(products)
    room.batch_all(products)
    material.batch_all(products)

    item_type_enricher.submit_batch()
    branded_terms_enricher.submit_batch()
    room_enricher.submit_batch()
    material_enricher.submit_batch()

    products = item_type.fetch_all(products).rename(columns={'similarity': 'item_type_sim'})
    products = branded_terms.fetch_all(products)
    products = room.fetch_all(products)
    products = material.fetch_all(products)

    labeled_query_products = labeled_query_products.merge(products[['product_id', 'item_type_same', 'item_type',
                                                                    'room', 'branded_terms', 'materials',
                                                                    'item_type_unconstrained', 'item_type_sim']], how='left', on='product_id')
    labeled_query_products['branded_terms'] = labeled_query_products['branded_terms'].apply(set).apply(" sep ".join)
    labeled_query_products['materials'] = labeled_query_products['materials'].apply(set).apply(" sep ".join)

    ground_truth_rooms_q = get_top_col_vals(labeled_query_products, 'room', 'No Room Fits', cutoff=0.8).drop(columns='count')
    ground_truth_item_type_q = get_top_col_vals(labeled_query_products, 'item_type_same', 'no item type matches', cutoff=0.8).drop(columns='count')
    ground_truth_branded_terms_q = get_top_col_vals(labeled_query_products, 'branded_terms',
                                                    'No Branded Terms Fits', cutoff=0.8).drop(columns='count')
    ground_truth_branded_terms_q = get_top_col_vals(labeled_query_products, 'materials',
                                                    'No Materials Fit', cutoff=0.8).drop(columns='count')
    logger.info("Gathering category hierarchy")
    ground_truth_classifications_q = get_top_col_vals(labeled_query_products, 'category hierarchy', 'No Classification Fits', cutoff=0.8).drop(columns='count')
    logger.info("Gathering categories")
    ground_truth_category_q = get_top_col_vals(labeled_query_products, 'category', 'No Category Fits', cutoff=0.8).drop(columns='count')
    logger.info("Gathering sub-categories")
    ground_truth_sub_category_q = get_top_col_vals(labeled_query_products, 'sub_category', 'No SubCategory Fits', cutoff=0.8).drop(columns='count')
    logger.info("Gathering product classes")
    ground_truth_product_class_q = get_top_col_vals(labeled_query_products, 'product_class', 'No Category Fits', cutoff=0.8).drop(columns='count')

    query_attributes = ground_truth_rooms_q.merge(
        ground_truth_item_type_q, how='outer', on='query'
    ).merge(
        ground_truth_branded_terms_q, how='outer', on='query'
    ).merge(
        ground_truth_classifications_q, how='outer', on='query'
    ).merge(
        ground_truth_product_class_q, how='outer', on='query'
    ).merge(
        ground_truth_category_q, how='outer', on='query'
    ).merge(
        ground_truth_sub_category_q, how='outer', on='query'
    )

    # Add classifications
    logger.info("Classifying queries")
    query_attributes['query'] = query_attributes['query'].str.strip()
    for query in query_attributes['query'].unique():
        query_attributes.loc[query_attributes['query'] == query, 'query_classification'] = " sep ".join(fully_classified(query).classifications)

    labeled_query_products = labeled_query_products.merge(
        query_attributes, how='left', on='query',
        suffixes=('_product', '_query')
    )
    return products, query_attributes, labeled_query_products


def main(products, labeled_query_products):
    enrich_output_dir = ensure_data_subdir('enrich_output')
    products, query_attributes, labeled_query_products = enrich_all(products, labeled_query_products)
    products.to_csv(os.path.join(enrich_output_dir, 'enriched_products.csv'), index=False)
    query_attributes.to_csv(os.path.join(enrich_output_dir, 'query_attributes.csv'), index=False)
    labeled_query_products.to_csv(os.path.join(enrich_output_dir, 'labeled_query_products.csv'), index=False)


if __name__ == "__main__":
    main(products, labeled_query_products)
