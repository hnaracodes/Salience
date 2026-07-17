"""Horikawa emotion decoding constants."""

HORIKAWA_SCHEMA_VERSION = 1

# Figshare features.zip (KamitaniLab EmotionVideoNeuralRepresentationPython).
FIGSHARE_FEATURES_URL = "https://ndownloader.figshare.com/files/48630544"
FIGSHARE_FEATURES_MD5 = "63f9721cf0a42173fec9c54e931cbf49"

# 34 emotion categories (Cowen & Keltner 2017; index order in figshare category mats).
HORIKAWA_34_CATEGORIES = (
    "admiration",
    "adoration",
    "aesthetic_appreciation",
    "amusement",
    "anger",
    "anxiety",
    "awe",
    "awkwardness",
    "boredom",
    "calmness",
    "confusion",
    "contempt",
    "craving",
    "disappointment",
    "disgust",
    "empathic_pain",
    "entrancement",
    "envy",
    "excitement",
    "fear",
    "guilt",
    "horror",
    "interest",
    "joy",
    "nostalgia",
    "pride",
    "relief",
    "romance",
    "sadness",
    "satisfaction",
    "sexual_desire",
    "surprise",
    "sympathy",
    "triumph",
)

# 14 affective dimensions (Cowen & Keltner 2017 Table S2; index order in figshare dimension mats).
HORIKAWA_14_DIMENSIONS = (
    "approach",
    "arousal",
    "attention",
    "certainty",
    "commitment",
    "control",
    "dominance",
    "effort",
    "fairness",
    "identity",
    "obstruction",
    "safety",
    "upswing",
    "valence",
)

# Product-facing subset for horikawa_ridge_v1 (8 mapped categories).
DEFAULT_PRODUCT_CATEGORIES = (
    "amusement",
    "awe",
    "contentment",
    "excitement",
    "fear",
    "anger",
    "sadness",
    "confusion",
)

# Map each of 34 Horikawa categories to a product label (or None if unaggregated).
CATEGORY_34_TO_PRODUCT_8: dict[str, str | None] = {
    "admiration": None,
    "adoration": None,
    "aesthetic_appreciation": None,
    "amusement": "amusement",
    "anger": "anger",
    "anxiety": "fear",
    "awe": "awe",
    "awkwardness": "confusion",
    "boredom": "confusion",
    "calmness": "contentment",
    "confusion": "confusion",
    "contempt": "anger",
    "craving": "excitement",
    "disappointment": "sadness",
    "disgust": None,
    "empathic_pain": "sadness",
    "entrancement": "excitement",
    "envy": "anger",
    "excitement": "excitement",
    "fear": "fear",
    "guilt": "anger",
    "horror": "fear",
    "interest": "excitement",
    "joy": "excitement",
    "nostalgia": "contentment",
    "pride": None,
    "relief": "contentment",
    "romance": None,
    "sadness": "sadness",
    "satisfaction": "contentment",
    "sexual_desire": None,
    "surprise": None,
    "sympathy": None,
    "triumph": None,
}

# Synthetic demo ratings for bootstrap training when figshare cache absent.
DEMO_CATEGORY_RATINGS = {
    "amusement": 0.12,
    "awe": 0.08,
    "contentment": 0.15,
    "excitement": 0.10,
    "fear": 0.05,
    "anger": 0.04,
    "sadness": 0.06,
    "confusion": 0.07,
}

# Dimension ratings are on 1–9 Likert in figshare; normalize with (x - 1) / 8.
DIMENSION_LIKERT_MIN = 1.0
DIMENSION_LIKERT_MAX = 9.0
