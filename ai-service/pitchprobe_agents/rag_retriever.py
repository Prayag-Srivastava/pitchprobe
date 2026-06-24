"""
RAG Retriever for PitchProbe
=============================
Loads the persistent Chroma vector store (built by Part 3 Advanced)
and provides utilities for document-aware retrieval.

This module is the SINGLE SOURCE OF TRUTH for vector store access.
Any agent that needs to search pitch decks imports from here.
"""

import os
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ============================================================
# CONFIG
# ============================================================
# Points to the same Chroma DB built by Part 3 Advanced.
# If this folder doesn't exist, run part3_advanced first.

PERSIST_DIR = "./chroma_db_advanced"
EMBEDDING_MODEL = "models/gemini-embedding-001"
SCORE_THRESHOLD = 0.85   # distance threshold — chunks above this are too weak

# ============================================================
# SINGLETON — load vector store ONCE, reuse everywhere
# ============================================================
# We use a module-level cache so the embeddings model and Chroma
# connection aren't recreated on every call. Imports happen once.

_embeddings = None
_vector_store = None


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return a cached embeddings instance."""
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings


def get_vector_store() -> Chroma:
    """Return a cached Chroma vector store, loaded from disk."""
    global _vector_store
    if _vector_store is None:
        if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
            raise FileNotFoundError(
                f"❌ No vector store found at {PERSIST_DIR}.\n"
                f"   Run part3_advanced_pitch_deck_analyzer.py FIRST to build it."
            )
        _vector_store = Chroma(
            embedding_function=get_embeddings(),
            persist_directory=PERSIST_DIR,
        )
    return _vector_store


# ============================================================
# DISCOVERY — what decks are available?
# ============================================================

def get_available_decks() -> list[str]:
    """
    Return the list of distinct document_name values in the vector store.
    Used by the discovery tool so agents know which decks exist.
    """
    store = get_vector_store()
    # Chroma's get() returns all documents with metadata
    all_data = store.get(include=["metadatas"])
    
    names = set()
    for metadata in all_data["metadatas"]:
        name = metadata.get("document_name")
        if name:
            names.add(name)
    
    return sorted(names)


# ============================================================
# DOCUMENT-AWARE SEARCH
# ============================================================

def search_deck(query: str, deck_name: str, k: int = 5) -> list[tuple[Document, float]]:
    """
    Search a specific pitch deck for the given query.
    Returns (Document, distance_score) tuples, filtered by confidence.
    """
    store = get_vector_store()
    
    # Use Chroma's metadata filter to scope retrieval to this deck only.
    # similarity_search_with_score returns (doc, distance) tuples
    # where LOWER distance = MORE relevant.
    results = store.similarity_search_with_score(
        query=query,
        k=k,
        filter={"document_name": deck_name},
    )
    
    # Apply confidence threshold — reject weak matches
    filtered = [(doc, score) for doc, score in results if score <= SCORE_THRESHOLD]
    return filtered