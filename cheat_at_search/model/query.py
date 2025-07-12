from pydantic import BaseModel, Field
from typing import List, Literal
from cheat_at_search.model.category_list import Categories, CategoriesReversed, SubCategories, FullyQualifiedCategories
from sentence_transformers import SentenceTransformer
import numpy as np


class Query(BaseModel):
    """
    Base model for search queries, containing common query attributes.
    """
    keywords: str = Field(
        ...,
        description="The original search query keywords sent in as input"
    )


class SynonymMapping(BaseModel):
    """
    Model for mapping phrases in the query to equivalent phrases or synonyms.
    """
    phrase: str = Field(
        ...,
        description="The original phrase from the query"
    )
    synonyms: List[str] = Field(
        ...,
        description="List of synonyms or equivalent phrases for the original phrase"
    )


class QueryWithSynonyms(Query):
    """
    Extended model for search queries that includes synonyms for keywords.
    Inherits from the base Query model.
    """
    synonyms: List[SynonymMapping] = Field(
        ...,
        description="Mapping of phrases in the query to equivalent phrases or synonyms"
    )


class SpellingCorrectedQuery(Query):
    """
    Model for search queries with spelling corrections applied.
    Inherits keywords from the base Query model.
    """
    corrected_keywords: str = Field(
        ...,
        description="Identical to original query string, but with spelling corrections applied"
    )


class BucketedQuery(Query):
    """
    Extended model for search queries that includes synonyms for keywords.
    Inherits from the base Query model.
    """
    information_need: Literal["navigation", "exploration"] = Field(
        default_factory=str,
        description="Information need of the query, either 'navigation' (go to specific product) or 'exploration' (browse products)"
    )


class QueryCategory(Query):
    """
    Structured representation of a search query for furniture e-commerce.
    Inherits keywords from the base Query model and adds category and sub-category.
    """
    category: Categories = Field(
        description="Category of the product, if identified. Use 'No Category Fits' if ambiguous or no category in list fits"
    )
    sub_category: SubCategories = Field(
        description="Sub-category of the product, if identified. Use 'No SubCategory Fits' if ambiguous or no sub-category in list fits"
    )


class QueryCategoryReversed(Query):
    """
    Structured representation of a search query for furniture e-commerce.
    Inherits keywords from the base Query model and adds category and sub-category.
    """
    category: CategoriesReversed = Field(
        description="Category of the product, if identified. Use 'No Category Fits' if ambiguous or no category in list fits"
    )
    sub_category: SubCategories = Field(
        description="Sub-category of the product, if identified. Use 'No SubCategory Fits' if ambiguous or no sub-category in list fits"
    )


class QueryCategoryFullyQualified(Query):
    """
    Fully qualified search query that includes all structured information.
    """
    full_category: FullyQualifiedCategories = Field(
        description="Fully qualified category of the product, if identified. Use 'No Category Fits' if ambiguous or no category in list fits. Options ordered from most common product categories to least common"
    )

    @property
    def category(self) -> str:
        return self.full_category.split('/')[0].strip()

    @property
    def sub_category(self) -> str:
        parts = self.full_category.split('/')
        return parts[1].strip() if len(parts) > 1 else 'No SubCategory Fits'


classifications_list = np.asarray([
    'Furniture / Bedroom Furniture / Beds & Headboards / Beds',
    'Furniture / Living Room Furniture / Chairs & Seating / Accent Chairs',
    'Rugs / Area Rugs',
    'Furniture / Office Furniture / Desks',
    'Furniture / Living Room Furniture / Coffee Tables & End Tables / Coffee Tables',
    'Furniture / Living Room Furniture / Coffee Tables & End Tables / End & Side Tables',
    'Décor & Pillows / Decorative Pillows & Blankets / Throw Pillows',
    'Furniture / Bedroom Furniture / Dressers & Chests',
    'Outdoor / Outdoor & Patio Furniture / Patio Furniture Sets / Patio Conversation Sets',
    'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Bathroom Vanities / All Bathroom Vanities',
    'Furniture / Living Room Furniture / Console Tables',
    'Décor & Pillows / Art / All Wall Art',
    'Furniture / Kitchen & Dining Furniture / Bar Furniture / Bar Stools & Counter Stools / All Bar Stools & Counter Stools',
    'Furniture / Kitchen & Dining Furniture / Dining Tables & Seating / Kitchen & Dining Chairs',
    'Furniture / Office Furniture / Office Chairs',
    'Décor & Pillows / Mirrors / All Mirrors',
    'Bed & Bath / Bedding / All Bedding',
    'Décor & Pillows / Wall Décor / Wall Accents',
    'Furniture / Living Room Furniture / Chairs & Seating / Recliners',
    'Furniture / Kitchen & Dining Furniture / Dining Tables & Seating / Kitchen and Dining Sets',
    'Décor & Pillows / Window Treatments / Curtains & Drapes',
    'Furniture / Living Room Furniture / Sectionals',
    'Baby & Kids / Toddler & Kids Bedroom Furniture / Kids Beds',
    'Furniture / Living Room Furniture / TV Stands & Media Storage Furniture / TV Stands & Entertainment Centers',
    'Lighting / Ceiling Lights / Chandeliers',
    'Furniture / Bedroom Furniture / Nightstands',
    'Baby & Kids / Toddler & Kids Bedroom Furniture / Kids Desks',
    'Décor & Pillows / Home Accessories / Decorative Objects',
    'Furniture / Bedroom Furniture / Beds & Headboards / Headboards',
    'Furniture / Living Room Furniture / Sofas',
    'Furniture / Living Room Furniture / Cabinets & Chests',
    'Décor & Pillows / Clocks / Wall Clocks',
    'Storage & Organization / Bathroom Storage & Organization / Bathroom Cabinets & Shelving',
    'Lighting / Table & Floor Lamps / Table Lamps',
    'Furniture / Living Room Furniture / Ottomans & Poufs',
    'Furniture / Kitchen & Dining Furniture / Kitchen Islands & Carts',
    'Furniture / Living Room Furniture / Bookcases',
    'Outdoor / Outdoor & Patio Furniture / Outdoor Seating & Patio Chairs / Patio Seating / Patio Sofas & Sectionals',
    'Furniture / Office Furniture / Office Storage Cabinets',
    'Furniture / Kitchen & Dining Furniture / Dining Tables & Seating / Kitchen & Dining Tables',
    'Contractor / Entry & Hallway / Coat Racks & Umbrella Stands',
    'Bed & Bath / Bedding Essentials / Mattress Pads & Toppers',
    'Home Improvement / Hardware / Home Hardware / Switch Plates',
    'Baby & Kids / Toddler & Kids Playroom / Playroom Furniture / Toddler & Kids Chairs & Seating',
    'Storage & Organization / Garage & Outdoor Storage & Organization / Outdoor Covers / Patio Furniture Covers',
    'Rugs / Doormats',
    'Rugs / Kitchen Mats',
    'Furniture / Bedroom Furniture / Beds & Headboards / Beds / Queen Size Beds',
    'Furniture / Bedroom Furniture / Daybeds',
    'Furniture / Living Room Furniture / Living Room Sets',
    'Outdoor / Outdoor & Patio Furniture / Patio Furniture Sets / Patio Dining Sets',
    'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Bathroom Sinks & Faucet Components / Bathroom Sink Faucets / Single Hole Bathroom Sink Faucets',
    'Outdoor / Outdoor Décor / Statues & Sculptures',
    'Décor & Pillows / Art / All Wall Art / Green Wall Art',
    'Furniture / Living Room Furniture / Coffee Tables & End Tables / Coffee Table Sets',
    'Furniture / Living Room Furniture / Chairs & Seating / Chaise Lounge Chairs',
    'Storage & Organization / Wall Shelving & Organization / Wall and Display Shelves',
    'Furniture / Living Room Furniture / Coffee Tables & End Tables / Coffee Tables / Rectangle Coffee Tables',
    'Décor & Pillows / Art / All Wall Art / Brown Wall Art',
    'Furniture / Kitchen & Dining Furniture / Bar Furniture / Bar Stools & Counter Stools / All Bar Stools & Counter Stools / Counter (24-27) Bar Stools & Counter Stools',
    'Furniture / Living Room Furniture / Coffee Tables & End Tables / Plant Stands & Tables',
    'Décor & Pillows / Window Treatments / Curtain Hardware & Accessories',
    'Furniture / Kitchen & Dining Furniture / Dining Tables & Seating / Kitchen & Dining Chairs / Side Kitchen & Dining Chairs',
    'Outdoor / Outdoor & Patio Furniture / Outdoor Seating & Patio Chairs / Patio Seating / Outdoor Club Chairs',
    'Furniture / Living Room Furniture / Chairs & Seating / Benches',
    'Home Improvement / Kitchen Remodel & Kitchen Fixtures / Kitchen Sinks & Faucet Components / Kitchen Sinks / Farmhouse & Apron Kitchen Sinks',
    'Kitchen & Tabletop / Kitchen Organization / Food Pantries',
    'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Towel Storage / Towel & Robe Hooks / Black Towel & Robe Hooks',
    'Storage & Organization / Garage & Outdoor Storage & Organization / Deck Boxes & Patio Storage',
    'Outdoor / Garden / Planters',
    'Lighting / Wall Lights / Bathroom Vanity Lighting',
    'Furniture / Kitchen & Dining Furniture / Sideboards & Buffets',
    'Storage & Organization / Garage & Outdoor Storage & Organization / Storage Racks & Shelving Units',
    'Home Improvement / Hardware / Cabinet Hardware / Cabinet & Drawer Pulls / Bronze Cabinet & Drawer Pulls',
    'Storage & Organization / Storage Containers & Drawers / All Storage Containers',
    'Bed & Bath / Shower Curtains & Accessories / Shower Curtains & Shower Liners',
    'Storage & Organization / Bathroom Storage & Organization / Hampers & Laundry Baskets',
    'Lighting / Light Bulbs & Hardware / Light Bulbs / All Light Bulbs / LED Light Bulbs',
    'Décor & Pillows / Art / All Wall Art / Blue Wall Art',
    'Bed & Bath / Mattresses & Foundations / Innerspring Mattresses',
    'Lighting / Outdoor Lighting / Outdoor Wall Lighting',
    'Storage & Organization / Garage & Outdoor Storage & Organization / Natural Material Storage / Log Storage',
    'Bed & Bath / Bathroom Accessories & Organization / Countertop Bath Accessories',
    'Storage & Organization / Shoe Storage / All Shoe Storage',
    'Home Improvement / Flooring, Walls & Ceiling / Floor Tiles & Wall Tiles / Ceramic Floor Tiles & Wall Tiles',
    'Home Improvement / Hardware / Cabinet Hardware / Cabinet & Drawer Pulls / Black Cabinet & Drawer Pulls',
    'Bed & Bath / Mattresses & Foundations / Adjustable Beds',
    "Rugs / Area Rugs / 2' x 3' Area Rugs",
    'Commercial Business Furniture / Commercial Office Furniture / Office Storage & Filing / Office Carts & Stands / All Carts & Stands',
    'Furniture / Bedroom Furniture / Beds & Headboards / Beds / Twin Beds',
    'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Bathroom Sinks & Faucet Components / Bathroom Sink Faucets / Widespread Bathroom Sink Faucets',
    "Rugs / Area Rugs / 4' x 6' Area Rugs",
    'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Bathroom Sinks & Faucet Components / Bathroom Sink Faucets',
    'Kitchen & Tabletop / Tableware & Drinkware / Table & Kitchen Linens / All Table Linens',
    'Kitchen & Tabletop / Kitchen Organization / Food Storage & Canisters / Food Storage Containers',
    'Décor & Pillows / Flowers & Plants / Faux Flowers',
    'Bed & Bath / Bedding / All Bedding / Twin Bedding',
    'Furniture / Bedroom Furniture / Dressers & Chests / White Dressers & Chests',
    'Home Improvement / Flooring, Walls & Ceiling / Floor Tiles & Wall Tiles / Porcelain Floor Tiles & Wall Tiles',
    'Home Improvement / Flooring, Walls & Ceiling / Flooring Installation & Accessories / Molding & Millwork / Wall Molding & Millwork',
    'Home Improvement / Doors & Door Hardware / Door Hardware & Accessories / Barn Door Hardware',
    'Bed & Bath / Bedding / Sheets & Pillowcases',
    'Furniture / Office Furniture / Chair Mats / Hard Floor Chair Mats',
    'Outdoor / Outdoor Fencing & Flooring / All Fencing',
    'Storage & Organization / Closet Storage & Organization / Clothes Racks & Garment Racks',
    'Kitchen & Tabletop / Kitchen Utensils & Tools / Colanders, Strainers, & Salad Spinners',
    'Outdoor / Hot Tubs & Saunas / Saunas',
    'Décor & Pillows / Decorative Pillows & Blankets / Throw Pillows / Blue Throw Pillows',
    'Bed & Bath / Bedding Essentials / Bed Pillows',
    'Lighting / Wall Lights / Wall Sconces',
    'Outdoor / Front Door Décor & Curb Appeal / Mailboxes',
    'Outdoor / Garden / Greenhouses',
    'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Showers & Bathtubs / Showers & Bathtubs Plumbing / Shower Faucets & Systems',
    'Bed & Bath / Mattresses & Foundations / Queen Mattresses',
    'Furniture / Bedroom Furniture / Jewelry Armoires',
    'Outdoor / Outdoor Shades / Awnings',
    'Baby & Kids / Nursery Bedding / Crib Bedding Sets',
    'Home Improvement / Hardware / Cabinet Hardware / Cabinet & Drawer Knobs / Brass Cabinet & Drawer Knobs',
    'Décor & Pillows / Art / All Wall Art / Red Wall Art',
    'Lighting / Ceiling Lights / All Ceiling Lights',
    'Lighting / Light Bulbs & Hardware / Lighting Components',
    'Furniture / Game Tables & Game Room Furniture / Poker & Card Tables',
    'Appliances / Kitchen Appliances / Range Hoods / All Range Hoods',
    'Home Improvement / Flooring, Walls & Ceiling / Floor Tiles & Wall Tiles / Natural Stone Floor Tiles & Wall Tiles',
    'Furniture / Kitchen & Dining Furniture / Bar Furniture / Bar Stools & Counter Stools / All Bar Stools & Counter Stools / Bar (28-33) Bar Stools & Counter Stools',
    'Outdoor / Outdoor Cooking & Tableware / Outdoor Serving & Tableware / Coolers, Baskets & Tubs / Picnic Baskets & Backpacks',
    'Décor & Pillows / Picture Frames & Albums / All Picture Frames',
    'Bed & Bath / Shower Curtains & Accessories / Shower Curtain Hooks',
    'Outdoor / Outdoor Shades / Outdoor Umbrellas / Patio Umbrella Stands & Bases',
    'Outdoor / Outdoor & Patio Furniture / Patio Bar Furniture / Patio Bar Stools',
    'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Toilets & Bidets / Toilet Paper Holders / Free Standing Toilet Paper Holders',
    'Storage & Organization / Garage & Outdoor Storage & Organization / Bike & Sport Racks',
    'Appliances / Kitchen Appliances / Refrigerators & Freezers / All Refrigerators / French Door Refrigerators',
    'Décor & Pillows / Home Accessories / Decorative Trays',
    'School Furniture and Supplies / School Spaces / Computer Lab Furniture / Podiums & Lecterns',
    'Lighting / Light Bulbs & Hardware / Lighting Shades',
    'Furniture / Kitchen & Dining Furniture / Bar Furniture / Home Bars & Bar Sets',
    'Lighting / Table & Floor Lamps / Floor Lamps',
    'Décor & Pillows / Wall Décor / Wall Accents / Brown Wall Accents',
    'Kitchen & Tabletop / Small Kitchen Appliances / Pressure & Slow Cookers / Slow Cookers / Slow Slow Cookers',
    'Décor & Pillows / Window Treatments / Curtains & Drapes / 90 Inch Curtains & Drapes',
    'Furniture / Bedroom Furniture / Armoires & Wardrobes',
    'Kitchen & Tabletop / Tableware & Drinkware / Flatware & Cutlery / Serving Utensils',
    'Baby & Kids / Baby & Kids Décor & Lighting / All Baby & Kids Wall Art',
    'Furniture / Office Furniture / Desks / Writing Desks',
    'Furniture / Office Furniture / Office Chairs / Task Office Chairs',
    'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Showers & Bathtubs / Shower & Bathtub Doors',
    'Outdoor / Outdoor & Patio Furniture / Outdoor Seating & Patio Chairs / Patio Seating / Patio Rocking Chairs & Gliders',
    'Home Improvement / Flooring, Walls & Ceiling / Walls & Ceilings / Wall Paneling',
    'Outdoor / Garden / Plant Stands & Accessories',
    'Furniture / Kitchen & Dining Furniture / Dining Tables & Seating / Kitchen & Dining Tables / 4 Seat Kitchen & Dining Tables',
    'Décor & Pillows / Home Accessories / Vases, Urns, Jars & Bottles',
    'Lighting / Wall Lights / Under Cabinet Lighting / Strip Under Cabinet Lighting',
    'Furniture / Bedroom Furniture / Bedroom and Makeup Vanities',
    'Pet / Dog / Dog Bowls & Feeding Supplies / Pet Bowls & Feeders',
    'Décor & Pillows / Candles & Holders / Candle Holders',
    'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Showers & Bathtubs / Shower & Bathtub Accessories',
    'Furniture / Office Furniture / Office Chair Accessories / Seat Cushion Office Chair Accessories',
    'Furniture / Office Furniture / Chair Mats',
    'Furniture / Living Room Furniture / Chairs & Seating / Massage Chairs',
    'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Bathroom Vanities / All Bathroom Vanities / Modern & Contemporary Bathroom Vanities',
    'Lighting / Ceiling Fans / All Ceiling Fans',
    'Home Improvement / Kitchen Remodel & Kitchen Fixtures / Kitchen Sinks & Faucet Components / Kitchen Faucets / Black Kitchen Faucets',
    'Lighting / Light Bulbs & Hardware / Light Bulbs / All Light Bulbs / Incandescent Light Bulbs',
    'Home Improvement / Flooring, Walls & Ceiling / Flooring Installation & Accessories / Molding & Millwork',
    'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Showers & Bathtubs / Bathtubs',
    'Décor & Pillows / Art / All Wall Art / Yellow Wall Art',
    'Pet / Dog / Pet Gates, Fences & Doors / Pet Gates',
    'Furniture / Bedroom Furniture / Beds & Headboards / Bed Frames / Twin Bed Frames',
    'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Towel Storage / Towel Bars, Racks, and Stands / Metal Towel Bars, Racks, and Stands',
    'Décor & Pillows / Art / All Wall Art / Pink Wall Art',
    'Home Improvement / Kitchen Remodel & Kitchen Fixtures / Smoke Detectors / Wall & Ceiling Mounted Smoke Detectors',
    'Outdoor / Garden / Planters / Plastic Planters',
    'Décor & Pillows / Mirrors / All Mirrors / Accent Mirrors',
    'Appliances / Kitchen Appliances / Range Hoods / All Range Hoods / Wall Mount Range Hoods',
    'Outdoor / Garden / Garden Décor / Lawn & Garden Accents',
    'Furniture / Living Room Furniture / Coffee Tables & End Tables / Coffee Tables / Round Coffee Tables',
    'Kitchen & Tabletop / Tableware & Drinkware / Dinnerware / Dining Bowls',
    'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Showers & Bathtubs / Showers & Bathtubs Plumbing / Shower Heads / Dual Shower Heads',
    'Home Improvement / Flooring, Walls & Ceiling / Floor Tiles & Wall Tiles / Glass Floor Tiles & Wall Tiles',
    'School Furniture and Supplies / Facilities & Maintenance / Trash & Recycling',
    'Home Improvement / Hardware / Cabinet Hardware / Cabinet & Drawer Pulls / Nickel Cabinet & Drawer Pulls',
    'Storage & Organization / Closet Storage & Organization / Closet Systems',
    'Furniture / Bedroom Furniture / Beds & Headboards / Beds / Full & Double Beds',
    'Commercial Business Furniture / Commercial Office Furniture / Office Storage & Filing / Office Carts & Stands / All Carts & Stands / Printer Carts & Stands',
    'Storage & Organization / Closet Storage & Organization / Closet Accessories',
    'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Bathroom Vanities / All Bathroom Vanities / Traditional Bathroom Vanities',
    'Home Improvement / Plumbing / Core Plumbing / Parts & Components',
    'Holiday Décor / Christmas / Christmas Trees / All Christmas Trees',
    'Décor & Pillows / Decorative Pillows & Blankets / Throw Pillows / Black Throw Pillows',
    'Furniture / Game Tables & Game Room Furniture / Sports Team Fan Shop & Memorabillia / Life Size Cutouts',
    'Lighting / Ceiling Lights / Pendant Lighting',
    'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Towel Storage / Towel & Robe Hooks',
    'Appliances / Washers & Dryers / Dryers / All Dryers / Gas Dryers',
    'Outdoor / Outdoor Recreation / Backyard Play / Kids Cars & Ride-On Toys',
    'Kitchen & Tabletop / Small Kitchen Appliances / Coffee, Espresso, & Tea / Coffee Makers',
    'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Showers & Bathtubs / Showers & Bathtubs Plumbing / Shower Heads',
    'Outdoor / Outdoor & Patio Furniture / Outdoor Seating & Patio Chairs / Patio Seating / Patio Sofas & Sectionals / Sectional Patio Sofas & Sectionals',
    'Lighting / Wall Lights / Under Cabinet Lighting',
    'Foodservice / Foodservice Tables / Table Parts',
    'Lighting / Outdoor Lighting / Landscape Lighting / All Landscape Lighting / Fence Post Cap Landscape Lighting',
    'Lighting / Outdoor Lighting / Landscape Lighting / All Landscape Lighting',
    'Outdoor / Outdoor & Patio Furniture / Outdoor Tables / All Patio Tables',
    'Commercial Business Furniture / Commercial Office Furniture / Office Storage & Filing / Office Carts & Stands / All Carts & Stands / Utility Carts & Stands',
    'Outdoor / Outdoor & Patio Furniture / Outdoor Seating & Patio Chairs / Outdoor Chaise & Lounge Chairs',
    'Furniture / Living Room Furniture / Chairs & Seating / Recliners / Brown Recliners',
    'Pet / Bird / Bird Perches & Play Gyms',
    'Décor & Pillows / Picture Frames & Albums / All Picture Frames / Single Picture Picture Frames',
    'Lighting / Outdoor Lighting / Outdoor Lanterns & Lamps',
    'Home Improvement / Hardware / Cabinet Hardware / Cabinet & Drawer Pulls',
    'Bed Accessories',
    'Clips/Clamps',
    'Décor & Pillows / Wall Décor / Wall Decals',
    'Home Improvement / Flooring, Walls & Ceiling / Floor Tiles & Wall Tiles',
    'Bed & Bath / Bedding / Sheets & Pillowcases / Twin XL Sheets & Pillowcases',
    'Kitchen & Tabletop / Tableware & Drinkware / Serveware / Serving Trays & Boards / Serving Trays & Platters / Serving Serving Trays & Platters',
    'Holiday Décor / Holiday Lighting',
    'Décor & Pillows / Wall Décor / Memo Boards',
    'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Toilets & Bidets / Toilet Paper Holders / Wall Mounted Toilet Paper Holders',
    'Décor & Pillows / Window Treatments / Curtains & Drapes / 63 Inch and Less Curtains & Drapes',
    'Home Improvement / Doors & Door Hardware / Door Hardware & Accessories / Door Knobs / Egg Door Knobs',
    'Décor & Pillows / Clocks / Wall Clocks / Analog Wall Clocks',
    'Home Improvement / Doors & Door Hardware / Interior Doors / Sliding Interior Doors',
    'Outdoor / Outdoor Recreation / Outdoor Games / All Outdoor Games',
    'Home Improvement / Doors & Door Hardware / Door Hardware & Accessories / Door Levers / Round Door Levers',
    'Storage & Organization / Garage & Outdoor Storage & Organization / Sheds / Storage Sheds',
    'Home Improvement / Doors & Door Hardware / Door Hardware & Accessories / Door Levers',
    'School Furniture and Supplies / School Furniture / School Tables / Folding Tables / Wood Folding Tables',
    'Décor & Pillows / Wall Décor / Wall Accents / Green Wall Accents',
    'School Furniture and Supplies / Facilities & Maintenance / Commercial Signage',
    'Storage & Organization / Garage & Outdoor Storage & Organization / Garage Storage Cabinets',
    'Furniture / Bedroom Furniture / Dressers & Chests / Beige Dressers & Chests',
    'Storage & Organization / Wall Shelving & Organization / Wall & Display Shelves',
    'Furniture / Game Tables & Game Room Furniture / Dartboards & Cabinets',
    'Outdoor / Outdoor Décor / Outdoor Pillows & Cushions / Patio Furniture Cushions / Lounge Chair Patio Furniture Cushions',
    'Outdoor / Outdoor & Patio Furniture / Patio Furniture Sets / Patio Dining Sets / Two Person Patio Dining Sets',
    'Décor & Pillows / Decorative Pillows & Blankets / Throw Pillows / Ivory & Cream Throw Pillows',
    'Appliances / Washers & Dryers / Washer & Dryer Sets / Black Washer & Dryer Sets',
    'School Furniture and Supplies / School Furniture / School Chairs & Seating / Stackable Chairs',
    'Home Improvement / Hardware / Cabinet Hardware / Cabinet & Drawer Pulls / Brass Cabinet & Drawer Pulls',
    'School Furniture and Supplies / School Boards & Technology / AV, Mounts & Tech Accessories / Electronic Mounts & Stands / Computer Mounts',
    'Furniture / Living Room Furniture / Chairs & Seating / Accent Chairs / Papasan Accent Chairs',
    'Storage & Organization / Shoe Storage / All Shoe Storage / Rack Shoe Storage',
    'Storage & Organization / Shoe Storage / All Shoe Storage / Cabinet Shoe Storage',
    'Storage & Organization / Storage Containers & Drawers / Storage Drawers',
    'Appliances / Kitchen Appliances / Wine & Beverage Coolers / Water Coolers',
    'Furniture / Living Room Furniture / Chairs & Seating / Rocking Chairs',
    'Kitchen & Tabletop / Tableware & Drinkware / Serveware / Serving Bowls & Baskets / Serving Bowls / NA Serving Bowls',
    'Furniture / Living Room Furniture / TV Stands & Media Storage Furniture / Projection Screens / Inflatable Projection Screens',
    'Appliances / Kitchen Appliances / Large Appliance Parts & Accessories',
    'Storage & Organization / Bathroom Storage & Organization / Hampers & Laundry Baskets / Laundry Hampers & Laundry Baskets',
    'Furniture / Office Furniture / Office Stools',
    'Outdoor / Outdoor & Patio Furniture / Outdoor Seating & Patio Chairs / Patio Seating / Outdoor Club Chairs / Metal Outdoor Club Chairs',
    'School Furniture and Supplies / School Furniture / School Tables / Folding Tables',
    'Lighting / Wall Lights / Bathroom Vanity Lighting / Traditional Bathroom Vanity Lighting',
    'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Bathroom Sinks & Faucet Components / Bathroom Sink Faucets / Centerset Bathroom Sink Faucets',
    'Décor & Pillows / Flowers & Plants / Faux Flowers / Orchid Faux Flowers',
    'Home Improvement / Flooring, Walls & Ceiling / Floor Tiles & Wall Tiles / Metal Floor Tiles & Wall Tiles',
    'Home Improvement / Kitchen Remodel & Kitchen Fixtures / Kitchen Sinks & Faucet Components / Kitchen Sinks',
    'Storage & Organization / Garage & Outdoor Storage & Organization / Outdoor Covers / Grill Covers / Charcoal Grill Grill Covers',
    'Outdoor / Outdoor Décor / Outdoor Wall Décor',
    'Storage & Organization / Cleaning & Laundry Organization / Laundry Room Organizers',
    'Reception Area / Reception Seating / Reception Sofas & Loveseats',
    'Kitchen & Tabletop / Cookware & Bakeware / Baking Sheets & Pans / Bread & Loaf Pans / Steel Bread & Loaf Pans',
    'Furniture / Living Room Furniture / Chairs & Seating / Accent Chairs / Wingback Accent Chairs',
    'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Showers & Bathtubs / Showers & Bathtubs Plumbing / Shower Heads / Fixed Shower Heads',
    'Kitchen & Tabletop / Kitchen Utensils & Tools / Kitchen Gadgets / Pasta Makers & Accessories',
    'School Furniture and Supplies / School Furniture / School Chairs & Seating / Classroom Chairs / High School & College Classroom Chairs',
    'Furniture / Living Room Furniture / Sectionals / Stationary Sectionals',
    'Furniture / Kitchen & Dining Furniture / Sideboards & Buffets / Drawer Equipped Sideboards & Buffets',
    'Kitchen & Tabletop / Cookware & Bakeware / Baking Sheets & Pans / Bread & Loaf Pans',
    'Kitchen & Tabletop / Kitchen Utensils & Tools / Cooking Utensils / All Cooking Utensils / Kitchen Cooking Utensils',
    'Décor & Pillows / Flowers & Plants / Live Plants',
    'Furniture / Living Room Furniture / TV Stands & Media Storage Furniture / Projection Screens / Folding Frame Projection Screens',
    'Kitchen & Tabletop / Kitchen Organization / Food Storage & Canisters / Kitchen Canisters & Jars / Metal Kitchen Canisters & Jars',
    'Outdoor / Outdoor Décor / Outdoor Fountains',
    'Outdoor / Outdoor Shades / Pergolas / Wood Pergolas',
    'Décor & Pillows / Candles & Holders / Candle Holders / Sconce Candle Holders',
    'Kitchen & Tabletop / Tableware & Drinkware / Serveware / Cake & Tiered Stands',
    'Home Improvement / Kitchen Remodel & Kitchen Fixtures / Kitchen Sinks & Faucet Components / Kitchen Faucets / Chrome Kitchen Faucets',
    'Décor & Pillows / Decorative Pillows & Blankets / Throw Pillows / White Throw Pillows',
    'Outdoor / Outdoor Fencing & Flooring / Turf',
    'Décor & Pillows / Window Treatments / Valances & Kitchen Curtains',
    'Home Improvement / Hardware / Cabinet Hardware / Cabinet & Drawer Knobs / Black Cabinet & Drawer Knobs',
    'Home Improvement / Kitchen Remodel & Kitchen Fixtures / Kitchen Sinks & Faucet Components / Kitchen Faucets / Bronze Kitchen Faucets',
    'Appliances / Washers & Dryers / Washer & Dryer Sets',
    'Décor & Pillows / Clocks / Mantel & Tabletop Clocks',
    'Home Improvement / Doors & Door Hardware / Interior Doors',
    'Storage & Organization / Wall Shelving & Organization / Wall & Display Shelves / Floating Wall & Display Shelves',
    'Outdoor / Outdoor Recreation / Backyard Play / Climbing Toys & Slides',
    'Home Improvement / Building Equipment / Dollies / Hand Truck Dollies',
    'Baby & Kids / Toddler & Kids Bedroom Furniture / Baby & Kids Dressers',
    'Décor & Pillows / Mirrors / All Mirrors / Leaning & Floor Mirrors',
    'Kitchen & Tabletop / Tableware & Drinkware / Drinkware / Mugs & Teacups',
    'Décor & Pillows / Flowers & Plants / Wreaths',
    'Outdoor / Outdoor Shades / Pergolas / Metal Pergolas',
    'Bed & Bath / Bedding / Sheets & Pillowcases / Twin Sheets & Pillowcases',
    'Outdoor / Outdoor Shades / Pergolas',
    'Reception Area / Reception Seating / Office Sofas & Loveseats',
    'Décor & Pillows / Home Accessories / Indoor Fountains',
    'Kitchen & Tabletop / Kitchen Organization / Food Storage & Canisters / Kitchen Canisters & Jars / Ceramic Kitchen Canisters & Jars',
    'Décor & Pillows / Window Treatments / Curtain Hardware & Accessories / Bracket Curtain Hardware & Accessories',
    'Home Improvement / Flooring, Walls & Ceiling / Walls & Ceilings / Accent Tiles / Ceramic Accent Tiles',
    'Home Improvement / Flooring, Walls & Ceiling / Walls & Ceilings / Accent Tiles',
    'Furniture / Living Room Furniture / Chairs & Seating / Accent Chairs / Arm Accent Chairs',
    'Furniture / Living Room Furniture / Coffee Tables & End Tables / Coffee Tables / Free Form Coffee Tables',
    'Décor & Pillows / Flowers & Plants / Faux Flowers / Rose Faux Flowers',
    'Bed & Bath / Mattresses & Foundations / Innerspring Mattresses / Twin Innerspring Mattresses',
    'Outdoor / Outdoor Décor / Outdoor Pillows & Cushions / Patio Furniture Cushions / Dining Chair Patio Furniture Cushions',
    'Furniture / Living Room Furniture / TV Stands & Media Storage Furniture / TV Stands & Entertainment Centers / Traditional TV Stands & Entertainment Centers',
    'Furniture / Living Room Furniture / Coffee Tables & End Tables / Plant Stands & Tables / Square Plant Stands & Tables',
    'Storage & Organization / Wall Shelving & Organization / Wall & Display Shelves / Corner Wall & Display Shelves',
    "Rugs / Area Rugs / 3' x 5' Area Rugs",
    'Kitchen & Tabletop / Tableware & Drinkware / Drinkware / Mugs & Teacups / Coffee Mugs & Teacups',
    'Contractor / Entry & Hallway / Coat Racks & Umbrella Stands / Wall Mounted Coat Racks & Umbrella Stands',
    "Baby & Kids / Toddler & Kids Playroom / Indoor Play / Kids' Playhouses",
    'Furniture / Living Room Furniture / Coffee Tables & End Tables / Coffee Tables / Square Coffee Tables',
    'Baby & Kids / Toddler & Kids Playroom / Indoor Play / Dollhouses & Accessories',
    'Bed & Bath / Bedding / All Bedding / Queen Bedding',
])

model = SentenceTransformer('all-MiniLM-L6-v2')
real_classifications = model.encode(classifications_list)
top_level_category_list = np.asarray([c.split(" / ")[0].strip() for c in classifications_list])
top_level_category_list


known_categories = set([c.split(" / ")[0].strip() for c in classifications_list])
known_sub_categories = set([c.split(" / ")[1].strip() for c in classifications_list if len(c.split(" / ")) > 1])


class QueryClassification(Query):
    """
    Represents a classification of a product.

    In this case, hallucinated, something the model is making up that looks like one
    of our classifications
    """
    hallucinated_classification: List[str] = Field(
        description="The classification you created for the query."
    )

    @property
    def classifications(self):
        """Search with model to resolve to real classification."""
        actuals = []
        for halluc_class in self.hallucinated_classification:
            query_embedding = model.encode(halluc_class)
            dot_prods = np.dot(real_classifications, query_embedding)
            actual = classifications_list[np.argmax(dot_prods)]
            while actual in actuals:
                dot_prods[np.argmax(dot_prods)] = -1
                actual = classifications_list[np.argmax(dot_prods)]

            actuals.append(actual)
        return actuals

    @property
    def categories(self):
        resolved_classification = self.classifications
        if len(resolved_classification) == 0 or resolved_classification == ["No Classification Fits"]:
            return []
        cats = []
        for c in resolved_classification:
            cats.append(c.split(" / ")[0].strip())
        return set(cats)

    @property
    def sub_categories(self):
        resolved_classification = self.classifications
        if len(resolved_classification) == 0 or resolved_classification == ["No Classification Fits"]:
            return []

        cats = []
        for c in resolved_classification:
            c_split = c.split(" / ")
            if len(c_split) < 2:
                continue
            cats.append(c.split(" / ")[1].strip())
        return set(cats)


class StructuredQuery(BaseModel):
    """
    Structured representation of a search query for furniture e-commerce.
    Inherits keywords from the base Query model.
    """
    search_terms: str = Field(
        default="",
        description="A rebuilt / better search query to use to search the product catalog"
    )
    material: str = Field(
        default="",
        description="Material extracted from the query, or empty string if none found"
    )
    color: str = Field(
        default="",
        description="Color mentioned in the query, or empty string if none found"
    )
    furniture_type: str = Field(
        default="",
        description="Type of furniture mentioned in the query"
    )
    room: str = Field(
        default="",
        description="Room where the furniture would be placed, if mentioned"
    )
    dimensions: List[str] = Field(
        default_factory=list,
        description="Any dimensions mentioned in the query"
    )
