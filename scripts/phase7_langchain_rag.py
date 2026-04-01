#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ensure repo root is importable when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v2_langchain.chain import run_langchain_rag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--show_docs", action="store_true")
    args = ap.parse_args()

    result = run_langchain_rag(
        question=args.question,
        k=args.top_k,
    )

    if args.show_docs:
        print("\n=== RETRIEVED DOCS ===\n")
        for i, doc in enumerate(result["docs"], 1):
            md = doc.metadata or {}
            snippet = (doc.page_content or "").replace("\n", " ")[:220]
            print(
                f"[Source {i}] "
                f"symbol={md.get('symbol')} | "
                f"accession={md.get('sec_accession_number')} | "
                f"chunk={md.get('chunk_index')}"
            )
            print(f"title: {md.get('title')}")
            print(f"text: {snippet}")
            print("-" * 100)

    print("\n=== ANSWER ===\n")
    print(result["answer"])


if __name__ == "__main__":
    main()