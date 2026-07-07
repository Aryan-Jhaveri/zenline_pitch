"""Category taxonomy adapted from my work on Vastra (vastra.cc).

Vastra is an AI wardrobe app I built that does photo-based garment
categorization. Its categorizer is anchored to a small, stable vocabulary
of article types — the same intuition I'm applying here at retail-catalog
scope. Keeping a fixed vocabulary of article types and reusing existing
spellings before inventing new ones is the "product ontology" idea: a
retailer-side substitute agent reasons about relationships between SKUs
whose attributes come from a fixed vocabulary, not free-form LLM output.

This module exposes that taxonomy as a Python dict plus helpers that
align the paramaggarwal dataset's `articleType` values (e.g. "Tshirts",
"Casual Shoes") onto Vastra's subcategory names where a match exists.
Where there is no match, the articleType passes through unchanged — I
don't force-fit.

Pure data + lookup; no LLM, no network.
"""

from __future__ import annotations

# The Vastra taxonomy, transcribed from closet-saas/src/constants/categories.ts.
# Keys are master categories; values are subcategory vocabularies.
VASTRA_TAXONOMY: dict[str, list[str]] = {
    "tops": [
        "t-shirt",
        "shirt",
        "blouse",
        "sweater",
        "hoodie",
        "tank top",
        "polo",
        "cardigan",
        "turtleneck",
    ],
    "bottoms": [
        "jeans",
        "trousers",
        "shorts",
        "skirt",
        "leggings",
        "sweatpants",
        "chinos",
    ],
    "dresses": ["dress", "jumpsuit", "romper"],
    "outerwear": ["jacket", "coat", "blazer", "vest", "parka", "raincoat"],
    "shoes": ["sneakers", "boots", "loafers", "heels", "sandals", "flats", "oxfords"],
    "accessories": [
        "hat",
        "cap",
        "scarf",
        "belt",
        "watch",
        "sunglasses",
        "bag",
        "jewelry",
    ],
    "activewear": [
        "sports top",
        "sports bra",
        "gym shorts",
        "leggings",
        "tracksuit",
    ],
    "underwear": ["socks", "underwear", "bra"],
}

# Flatten the subcategory vocabulary for the "reuse existing spelling" lookup.
ALL_SUBCATEGORIES: list[str] = [s for subs in VASTRA_TAXONOMY.values() for s in subs]

# Mapping from paramaggarwal `articleType` values to Vastra subcategory
# names. Only entries where the alignment is unambiguous are mapped; the
# rest pass through unchanged in `align_article_type`. Built by hand from
# the dataset's observed articleType values (top ~140 types).
_ARTICLE_TYPE_ALIASES: dict[str, str] = {
    "Tshirts": "t-shirt",
    "Shirts": "shirt",
    "Casual Shirts": "shirt",
    "Formal Shirts": "shirt",
    "Tops": "blouse",
    "Sweaters": "sweater",
    "Sweatshirts": "hoodie",
    "Hoodies": "hoodie",
    "Tunics": "tank top",
    "Tank Tops": "tank top",
    "Polo Shirts": "polo",
    "Cardigans": "cardigan",
    "Jeans": "jeans",
    "Trousers": "trousers",
    "Formal Trousers": "trousers",
    "Casual Trousers": "chinos",
    "Chinos": "chinos",
    "Track Pants": "sweatpants",
    "Track Shorts": "shorts",
    "Shorts": "shorts",
    "Skirts": "skirt",
    "Skorts": "skirt",
    "Leggings": "leggings",
    "Jeggings": "leggings",
    "Jumpsuits": "jumpsuit",
    "Dresses": "dress",
    "Kurtas": "dress",
    "Kurtis": "dress",
    "Jackets": "jacket",
    "Blazers": "blazer",
    "Coats": "coat",
    "Waistcoat": "vest",
    "Vests": "vest",
    "Sneakers": "sneakers",
    "Sports Shoes": "sneakers",
    "Casual Shoes": "loafers",
    "Loafers": "loafers",
    "Boots": "boots",
    "Heels": "heels",
    "Sandals": "sandals",
    "Flip Flops": "sandals",
    "Flats": "flats",
    "Formal Shoes": "oxfords",
    "Socks": "socks",
    "Briefs": "underwear",
    "Boxers": "underwear",
    "Trunk": "underwear",
    "Innerwear Vests": "vest",
    "Watches": "watch",
    "Sunglasses": "sunglasses",
    "Belts": "belt",
    "Bags": "bag",
    "Backpacks": "bag",
    "Handbags": "bag",
    "Caps": "cap",
    "Hats": "hat",
    "Scarves": "scarf",
    "Stoles": "scarf",
    "Jewellery": "jewelry",
}


def align_article_type(article_type: str) -> str:
    """Align a paramaggarwal `articleType` onto the Vastra subcategory vocabulary.

    Returns the Vastra subcategory name if an alias is defined, else the
    original articleType unchanged (we do not force-fit ambiguous cases).
    Empty / None input returns an empty string.
    """
    if not article_type:
        return ""
    return _ARTICLE_TYPE_ALIASES.get(article_type, article_type)


def is_known_subcategory(subcategory: str) -> bool:
    """True if `subcategory` is in the Vastra vocabulary (after alignment)."""
    return subcategory in ALL_SUBCATEGORIES


def category_for_subcategory(subcategory: str) -> str | None:
    """Return the Vastra master category for a subcategory, or None."""
    for cat, subs in VASTRA_TAXONOMY.items():
        if subcategory in subs:
            return cat
    return None
