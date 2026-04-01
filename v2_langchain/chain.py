# v2_langchain/chain.py
from __future__ import annotations

from typing import List

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from v2_langchain.config import LLM_MODEL, OLLAMA_BASE_URL
from v2_langchain.prompts import QA_PROMPT
from v2_langchain.retriever import retrieve_documents

def format_docs(docs: List[Document], text_chars: int = 1200) -> str:
    blocks: List[str] = []

    for i, doc in enumerate(docs, 1):
        md = doc.metadata or {}
        snippet = (doc.page_content or "").replace("\n", " ").strip()[:text_chars]

        block = (
            f"[Source {i}]\n"
            f"symbol: {md.get('symbol')}\n"
            f"company_name: {md.get('company_name')}\n"
            f"accession: {md.get('sec_accession_number')}\n"
            f"release_dt_utc: {md.get('release_dt_utc')}\n"
            f"title: {md.get('title')}\n"
            f"chunk_index: {md.get('chunk_index')}\n"
            f"text: {snippet}\n"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def run_langchain_rag(question: str, k: int = 5) -> dict:
    docs = retrieve_documents(question, k=k)

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    context = format_docs(docs)
    s = QA_PROMPT | llm | StrOutputParser()
    answer = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    return {
        "question": question,
        "docs": docs,
        "context": context,
        "answer": answer,
    }