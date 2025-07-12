from cheat_at_search.wands_data import products, labeled_query_products, queries
from cheat_at_search.model.product import ItemType, BrandedTerms, ProductRoom
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
    item_type.batch_all(products)
    branded_terms.batch_all(products)
    room.batch_all(products)

    item_type_enricher.submit_batch()
    branded_terms_enricher.submit_batch()
    room_enricher.submit_batch()

    products = item_type.fetch_all(products).rename(columns={'similarity': 'item_type_sim'})
    products = branded_terms.fetch_all(products)
    products = room.fetch_all(products)

    labeled_query_products = labeled_query_products.merge(products[['product_id', 'item_type_same', 'item_type',
                                                                    'room', 'branded_terms',
                                                                    'item_type_unconstrained', 'item_type_sim']], how='left', on='product_id')

    labeled_query_products['branded_terms'] = labeled_query_products['branded_terms'].apply(set).apply(" sep ".join)

    ground_truth_rooms_q = get_top_col_vals(labeled_query_products, 'room', 'No Room Fits', cutoff=0.8).drop(columns='count')
    ground_truth_item_type_q = get_top_col_vals(labeled_query_products, 'item_type_same', 'no item type matches', cutoff=0.8).drop(columns='count')
    ground_truth_branded_terms_q = get_top_col_vals(labeled_query_products, 'branded_terms',
                                                    'No Branded Terms Fits', cutoff=0.8).drop(columns='count')

    query_attributes = ground_truth_rooms_q.merge(
        ground_truth_item_type_q, how='outer', on='query'
    ).merge(
        ground_truth_branded_terms_q, how='outer', on='query'
    )
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
