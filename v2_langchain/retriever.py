# v2_langchain/retriever.py
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from qdrant_client import QdrantClient

from v2_langchain.config import (
    COLLECTION_NAME,
    OLLAMA_BASE_URL,
    EMBED_MODEL,
    QDRANT_URL,
    TOP_K,
)
from v2_langchain.qdrant_store import get_embeddings


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def retrieve_documents(query: str, k: int = TOP_K) -> List[Document]:
    embeddings = get_embeddings()
    qdrant = get_qdrant_client()

    query_vector = embeddings.embed_query(query)

    res = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
        with_payload=True,
        with_vectors=False,
    )

    docs: List[Document] = []
    for hit in res.points:
        payload = hit.payload or {}

        page_content = str(payload.get("text") or "")
        metadata = {
            "symbol": payload.get("symbol"),
            "company_name": payload.get("company_name"),
            "sec_accession_number": payload.get("sec_accession_number"),
            "release_dt_utc": payload.get("release_dt_utc"),
            "title": payload.get("title"),
            "chunk_index": payload.get("chunk_index"),
            "score": hit.score,
        }

        docs.append(Document(page_content=page_content, metadata=metadata))

    return docs