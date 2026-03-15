"""Synthetic product description generator.

No external datasets required — all data is generated from random word combos.
Produces realistic-looking product documents for benchmarking.
"""
from __future__ import annotations

import random
from typing import Any

_ADJECTIVES = [
    "wireless", "portable", "premium", "compact", "lightweight", "waterproof",
    "ergonomic", "foldable", "rechargeable", "noise-cancelling", "high-fidelity",
    "ultra-thin", "heavy-duty", "smart", "solar-powered", "voice-activated",
    "multi-functional", "durable", "flexible", "magnetic",
]

_NOUNS = [
    "headphones", "speaker", "keyboard", "mouse", "monitor", "webcam", "laptop",
    "tablet", "charger", "cable", "stand", "hub", "earbuds", "microphone",
    "camera", "drone", "watch", "bracelet", "backpack", "case",
]

_BRANDS = [
    "TechPro", "NovaBeat", "ZenCore", "AlphaWave", "PixelForge", "SonicBridge",
    "AetherTech", "ClearVox", "DeltaSound", "OmegaLink",
]

_CATEGORIES = [
    "electronics", "audio", "computing", "accessories", "wearables",
    "photography", "gaming", "office", "home", "outdoor",
]

_FEATURES = [
    "with fast charging", "up to 40hr battery", "Bluetooth 5.3",
    "USB-C compatible", "IP68 waterproof rating", "AI noise reduction",
    "120Hz refresh rate", "dual microphone array", "multi-device pairing",
    "foldable design", "under $50", "ideal for travel", "includes carry case",
    "1-year warranty", "available in 5 colors",
]


def generate_product(idx: int, rng: random.Random) -> dict[str, Any]:
    """Generate a single fake product document."""
    adj = rng.choice(_ADJECTIVES)
    noun = rng.choice(_NOUNS)
    brand = rng.choice(_BRANDS)
    category = rng.choice(_CATEGORIES)
    price = round(rng.uniform(9.99, 299.99), 2)
    features = rng.sample(_FEATURES, k=rng.randint(2, 4))

    description = (
        f"{brand} {adj} {noun}. "
        f"Category: {category}. "
        f"Features: {', '.join(features)}. "
        f"Price: ${price:.2f}."
    )

    return {
        "id": str(idx),
        "title": f"{brand} {adj.title()} {noun.title()}",
        "description": description,
        "brand": brand,
        "category": category,
        "price": price,
        "tags": [adj, noun, category],
    }


def generate_dataset(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """Generate n product documents."""
    rng = random.Random(seed)
    return [generate_product(i, rng) for i in range(n)]


def generate_queries(n: int, seed: int = 99) -> list[str]:
    """Generate n realistic product search queries."""
    rng = random.Random(seed)
    queries = []
    for _ in range(n):
        adj = rng.choice(_ADJECTIVES)
        noun = rng.choice(_NOUNS)
        feature = rng.choice(_FEATURES)
        style = rng.randint(0, 2)
        if style == 0:
            queries.append(f"{adj} {noun}")
        elif style == 1:
            queries.append(f"{noun} {feature}")
        else:
            queries.append(f"best {adj} {noun} {feature}")
    return queries
