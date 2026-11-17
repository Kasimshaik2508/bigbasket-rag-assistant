"""
product_art.py — picks an illustration for a product card.

WHY this exists: the Kaggle BigBasket dataset has no image column, so we
can't show real photos. Instead each card gets an illustration that matches
what the product IS (cereal, dal, shampoo...), which is always correct and
needs no internet or extra API.

HOW it works:
  1. Look at the product's type, sub-category and name (most specific first).
  2. Check the type first, then sub-category, then name; within each, walk
     the RULES list top to bottom and the first keyword found wins.
     Specific rules (e.g. "granola") come before general ones (e.g. "snack"),
     so "Multi Millet Granola" gets a cereal bowl, not a generic snack.
  3. If nothing matches, fall back to the product's category icon.
  4. The tile colour comes from the category, so cards from the same aisle
     share a colour family.
"""

import re

# (keywords, emoji) — order matters: more specific first.
RULES = [
    # breakfast & bakery
    (("granola", "muesli", "cornflake", "corn flake", "oats", "cereal", "flakes", "chocos"), "🥣"),
    (("bread", "bun", "pav", "rusk", "khari", "toast"), "🍞"),
    (("cake", "pastry", "pastries", "muffin", "brownie"), "🍰"),
    (("cookie", "biscuit", "cracker", "wafer"), "🍪"),
    (("chocolate", "cocoa", "candy", "candies", "toffee"), "🍫"),
    (("ice cream", "dessert", "kulfi"), "🍨"),
    (("mithai", "sweet", "ladoo", "barfi", "halwa"), "🍮"),
    # staples
    (("rice", "poha", "basmati", "quinoa", "grain", "millet"), "🍚"),
    (("dal", "pulse", "lentil", "rajma", "chana", "moong", "toor", "urad", "masoor", "beans"), "🫘"),
    (("atta", "flour", "sooji", "maida", "besan", "rava"), "🌾"),
    (("hair oil", "hair serum", "massage oil", "body oil", "essential oil", "baby oil"), "🧴"),
    (("ghee", "butter"), "🧈"),
    (("oil", "vinegar"), "🫒"),
    (("salt", "sugar", "jaggery"), "🧂"),
    (("ketchup", "sauce", "dip", "mayonnaise", "mayo", "tinned", "canned", "processed"), "🥫"),
    (("masala", "spice", "chilli", "chili", "pepper", "turmeric", "haldi", "jeera", "cumin",
      "herb", "seasoning", "marinade"), "🌶️"),
    (("honey",), "🍯"),
    (("jam", "spread", "peanut butter", "nutella"), "🍓"),
    (("pickle", "chutney"), "🥒"),
        (("noodle", "pasta", "vermicelli", "soup", "macaroni"), "🍜"),
    (("ready to cook", "ready to eat", "frozen"), "🍱"),
    (("dry fruit", "almond", "cashew", "raisin", "walnut", "nut", "pista", "dates"), "🥜"),
    (("namkeen", "chips", "snack", "bhujia", "popcorn", "makhana"), "🍿"),
    # drinks & dairy
    (("tea", "chai", "green tea"), "🍵"),
    (("coffee",), "☕"),
    (("juice", "fruit drink", "squash", "syrup"), "🧃"),
    (("fridge bottle", "water bottle"), "🫙"),
    (("water", "aerated", "sparkling", "mineral"), "💧"),
    (("supplement", "multivitamin", "vitamin", "capsule"), "💊"),
    (("energy", "soft drink", "soda", "cola", "health drink", "protein", "cocktail mix", "non-alcoholic"), "🥤"),
    (("cheese", "paneer"), "🧀"),
    (("milk", "curd", "yogurt", "yoghurt", "dairy", "lassi", "buttermilk", "cream", "non dairy"), "🥛"),
    (("egg",), "🥚"),
    # fresh & meat
    (("fish", "seafood", "prawn"), "🐟"),
    (("chicken", "mutton", "lamb", "meat", "pork", "sausage", "bacon", "salami"), "🍗"),
    (("fruit", "apple", "banana", "mango", "orange"), "🍎"),
    (("vegetable", "veggie", "sprout", "cuts", "onion", "potato", "tomato"), "🥦"),
    (("flower", "bouquet"), "💐"),
    # cleaning cloths before baby wipes
    (("dust cloth", "duster", "microfiber", "mop", "wiper"), "🧽"),
    # baby
    (("diaper",), "🧷"),
    (("baby food", "formula", "feeding", "nursing", "baby"), "🍼"),
    # personal care
    (("toothpaste", "toothbrush", "oral", "mouthwash"), "🪥"),
    (("shampoo", "conditioner", "hair"), "🧴"),
    (("soap", "body wash", "hand wash", "bath", "shower"), "🧼"),
    (("lipstick", "makeup", "kajal", "eyeliner", "mascara", "nail", "foundation"), "💄"),
    (("perfume", "fragrance", "deo", "deodorant", "talc", "powder", "attar"), "🌸"),
    (("razor", "shaving", "beard", "grooming", "trimmer"), "🪒"),
    (("sanitary", "feminine", "pad", "tampon"), "🌷"),
    (("medicine", "health", "ayurveda", "ayurvedic", "tablet", "capsule", "balm", "first aid"), "💊"),
    (("cream", "lotion", "face", "skin", "sunscreen", "serum", "moisturiser", "moisturizer", "body care", "face care"), "🧴"),
    # household
    (("detergent", "dishwash", "laundry", "fabric"), "🫧"),
    (("mop", "brush", "scrub", "cleaner", "cleaning", "wiper"), "🧽"),
    (("freshener", "repellent", "mosquito", "insect"), "🌬️"),
    (("garbage", "disposable", "foil", "tissue", "napkin"), "🗑️"),
    (("pooja", "agarbatti", "incense", "diya"), "🪔"),
    (("pet", "dog", "cat"), "🐾"),
    (("garden", "plant", "seed", "pot", "soil"), "🪴"),
    (("knife", "knive", "peeler", "chopper", "grater", "strainer", "ladle", "spatula", "kitchen tool", "baking accessori"), "🔪"),
    (("cookware", "non stick", "pan", "kadai", "tawa", "pressure cooker", "bakeware"), "🍳"),
    (("steel", "utensil", "cutlery", "crockery", "plate", "bowl", "spoon", "glass", "mug", "cup"), "🍽️"),
    (("storage", "container", "jar", "box", "bottle", "flask", "casserole"), "🫙"),
    (("appliance", "electrical", "bulb", "battery", "led", "charger"), "🔌"),
    (("stationery", "pen", "notebook", "pencil"), "✏️"),
    (("car", "shoe"), "👟"),
    (("party", "festive", "balloon", "candle", "lighter"), "🎉"),
    (("cotton", "ear bud"), "🧴"),
]

# Category fallback icon + tile colour (soft tint so the emoji stands out).
CATEGORY_ART = {
    "Beauty & Hygiene":          ("🧴", "#F6E9F1"),
    "Gourmet & World Food":      ("🫒", "#EEF1E2"),
    "Snacks & Branded Foods":    ("🍪", "#FBEEDB"),
    "Foodgrains, Oil & Masala":  ("🌾", "#F7F0DC"),
    "Cleaning & Household":      ("🧽", "#E4F0F5"),
    "Kitchen, Garden & Pets":    ("🍳", "#EAF2E6"),
    "Bakery, Cakes & Dairy":     ("🥛", "#FBF3E6"),
    "Beverages":                 ("☕", "#F3E9E1"),
    "Baby Care":                 ("🍼", "#E9EEF8"),
    "Fruits & Vegetables":       ("🥕", "#E6F3E3"),
    "Eggs, Meat & Fish":         ("🥚", "#F8ECE6"),
    "Policy":                    ("📄", "#EEF0F2"),
}
DEFAULT_ART = ("🛒", "#EEF4E8")


def _has(text, word):
    # whole-word match (plural s/es allowed) so "tea" doesn't fire inside
    # "steam" and "pad" doesn't fire inside "padded"
    return re.search(r"(?<![a-z])" + re.escape(word) + r"(?:s|es)?(?![a-z])", text) is not None


def pick_art(product="", category="", sub_category="", ptype=""):
    """Return (emoji, background_colour) for one product."""
    cat_icon, tint = CATEGORY_ART.get(category, DEFAULT_ART)
    # Most specific field first: the product TYPE ("Body Care") is a better
    # signal than words in the NAME ("Active Fruits ... Lotion").
    for field in (ptype, sub_category, product):
        text = (field or "").lower()
        if not text:
            continue
        for words, emoji in RULES:
            if any(_has(text, w) for w in words):
                return emoji, tint
    return cat_icon, tint


def parse_header(chunk_text):
    """Pull sub-category and type out of a chunk's header lines
    ("Category: A > B" and "Type: C") — they are in the chunk text even
    though they aren't stored as separate metadata fields."""
    sub = re.search(r"Category:[^\n>]*>\s*([^\n]*)", chunk_text or "")
    typ = re.search(r"Type:\s*([^\n]*)", chunk_text or "")
    return (sub.group(1).strip() if sub else ""), (typ.group(1).strip() if typ else "")
