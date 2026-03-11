import argparse
import json

from eventlens_tools import (
    answer_from_evidence,
    candidates_to_json,
    evidence_to_json,
    extract_event_candidates,
    search_sec_filings,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)

    parser.add_argument("--collection", default="sec_8k_chunks")
    parser.add_argument("--qdrant_url", default="http://localhost:6333")
    parser.add_argument("--ollama_host", default="http://127.0.0.1:11434")
    parser.add_argument("--embed_model", default="nomic-embed-text:latest")
    parser.add_argument("--llm_model", default="qwen2.5:7b-instruct")

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--candidate_k", type=int, default=30)
    parser.add_argument("--max_chunks_per_accession", type=int, default=2)

    parser.add_argument("--symbol")
    parser.add_argument("--exchange")
    parser.add_argument("--accession")
    parser.add_argument("--date_from")
    parser.add_argument("--date_to")

    parser.add_argument("--evidence_chars", type=int, default=1200)
    parser.add_argument("--show_evidence", action="store_true")
    parser.add_argument("--show_candidates", action="store_true")
    args = parser.parse_args()

    evidence = search_sec_filings(
        query=args.question,
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        ollama_host=args.ollama_host,
        embed_model=args.embed_model,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        max_chunks_per_accession=args.max_chunks_per_accession,
        symbol=args.symbol,
        exchange=args.exchange,
        accession=args.accession,
        date_from=args.date_from,
        date_to=args.date_to,
    )

    if args.show_evidence:
        print("\n=== TOOL: search_sec_filings ===\n")
        print(json.dumps(evidence_to_json(evidence), indent=2))

    candidates = extract_event_candidates(evidence)

    if args.show_candidates:
        print("\n=== TOOL: extract_event_candidates ===\n")
        print(json.dumps(candidates_to_json(candidates), indent=2))

    answer = answer_from_evidence(
        question=args.question,
        evidence_chunks=evidence,
        llm_model=args.llm_model,
        ollama_host=args.ollama_host,
        evidence_chars=args.evidence_chars,
    )

    print("\n=== TOOL: answer_from_evidence ===\n")
    print(answer)


if __name__ == "__main__":
    main()