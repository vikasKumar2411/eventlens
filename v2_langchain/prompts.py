# v2_langchain/prompts.py
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

QA_PROMPT = ChatPromptTemplate.from_template(
    """
You are a financial filings assistant working only from retrieved SEC 8-K evidence.

Use ONLY the provided context.
Do not invent facts.
If the evidence is insufficient or ambiguous, say so clearly.

User question:
{question}

Context:
{context}

Return:
1. A concise answer
2. Bullet citations using the source metadata when available
""".strip()
)