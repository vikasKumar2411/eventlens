# v2_langchain/qdrant_store.py
from __future__ import annotations

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

from v2_langchain.config import (
    COLLECTION_NAME,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
    QDRANT_URL,
)


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def get_vector_store() -> QdrantVectorStore:
    embeddings = get_embeddings()

    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
        content_payload_key="text",
        metadata_payload_key="",  # we’ll handle metadata directly below if needed
    )