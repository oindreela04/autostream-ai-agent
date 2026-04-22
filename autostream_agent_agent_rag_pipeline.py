"""
rag_pipeline.py
---------------
Simple RAG (Retrieval-Augmented Generation) pipeline that loads
AutoStream's knowledge base and returns relevant context snippets
for a given user query.
"""

import json
import os
from pathlib import Path


KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.json"


def load_knowledge_base() -> dict:
    """Load the JSON knowledge base from disk."""
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_knowledge_text(kb: dict) -> str:
    """Flatten the knowledge base into a single readable text block."""
    lines = []

    lines.append(f"Company: {kb['company']}")
    lines.append(f"About: {kb['tagline']}\n")

    lines.append("=== PRICING PLANS ===")
    for plan in kb["plans"]:
        lines.append(f"\n{plan['name']} — {plan['price']}")
        for feature in plan["features"]:
            lines.append(f"  • {feature}")

    lines.append("\n=== COMPANY POLICIES ===")
    for policy in kb["policies"]:
        lines.append(f"  • {policy}")

    lines.append("\n=== FAQs ===")
    for faq in kb["faqs"]:
        lines.append(f"Q: {faq['question']}")
        lines.append(f"A: {faq['answer']}\n")

    return "\n".join(lines)


def retrieve_context(query: str) -> str:
    """
    Simple keyword-based retrieval.
    Returns the full knowledge base text (small enough for full inclusion).
    In a production system this would use embeddings + vector search.
    """
    kb = load_knowledge_base()
    return build_knowledge_text(kb)


# Pre-built context string (cached at import time)
FULL_CONTEXT = retrieve_context("")
