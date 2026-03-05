import argparse
import hashlib
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

def ollama_embed(texts: List[str], model: str, host: str) -> List[List[float]]:
    out: List[List[float]] = []
    for t in texts:
        r = requests.post(
            f"{host}/api/embeddings",
            json={"model": model, "prompt": t},
            timeout=180,
        )
        r.raise_for_status()
        out.append(r.json()["embedding"])
    return out

def chunk_text(text: str, target_chars: int = 3500, overlap_chars: int = 500) -> List[Tuple[int, int, str]]:
    if not text:
        return []
    text = text.strip()
    n = len(text)
    chunks: List[Tuple[int, int, str]] = []
    start = 0
    while start < n:
        end = min(n, start + target_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end == n:
            break
        start = max(0, end - overlap_chars)
    return chunks

def stable_chunk_ids(accession: str, chunk_index: int, start: int, end: int) -> tuple[str, str]:
    raw = f"{accession}:{chunk_index}:{start}:{end}"
    sha = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, sha))  # deterministic UUID
    return sha, point_id

def ensure_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_dir", required=True, help="data/processed/filings_parquet")
    ap.add_argument("--collection", default="sec_8k_chunks")
    ap.add_argument("--qdrant_url", default="http://localhost:6333")
    ap.add_argument("--ollama_host", default="http://127.0.0.1:11434")
    ap.add_argument("--embed_model", default="nomic-embed-text:latest")
    ap.add_argument("--max_filings", type=int, default=200)
    ap.add_argument("--batch_chunks", type=int, default=32)
    ap.add_argument("--target_chars", type=int, default=3500)
    ap.add_argument("--overlap_chars", type=int, default=500)
    args = ap.parse_args()

    parquet_dir = Path(args.parquet_dir)
    files = sorted(parquet_dir.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found under {parquet_dir}")

    qdrant = QdrantClient(url=args.qdrant_url)

    probe = ollama_embed(["probe"], model=args.embed_model, host=args.ollama_host)[0]
    vector_size = len(probe)
    print("Embedding vector size:", vector_size)

    ensure_collection(qdrant, args.collection, vector_size)

    filings_seen = 0
    points_buf: List[qm.PointStruct] = []
    t0 = time.time()

    def flush():
        nonlocal points_buf
        if not points_buf:
            return
        texts = [p.payload["text"] for p in points_buf]  # type: ignore[index]
        vecs = ollama_embed(texts, model=args.embed_model, host=args.ollama_host)
        for p, v in zip(points_buf, vecs):
            p.vector = v
        qdrant.upsert(collection_name=args.collection, points=points_buf)
        points_buf = []

    for f in files:
        df = pd.read_parquet(f)

        for row in df.itertuples(index=False):
            if filings_seen >= args.max_filings:
                break

            accession = str(getattr(row, "sec_accession_number"))
            symbol = getattr(row, "symbol", None)
            company_name = getattr(row, "company_name", None)
            exchange = getattr(row, "exchange", None)
            release_dt_utc = getattr(row, "release_dt_utc", None)
            title = getattr(row, "title", None)

            text = getattr(row, "raw_text_clean", "") or ""
            chunks = chunk_text(text, target_chars=args.target_chars, overlap_chars=args.overlap_chars)

            for i, (start, end, chunk) in enumerate(chunks):
                chunk_id, point_id = stable_chunk_ids(accession, i, start, end)
                payload: Dict[str, Any] = {
                    "chunk_id": chunk_id,
                    "sec_accession_number": accession,
                    "symbol": str(symbol) if symbol is not None else None,
                    "company_name": str(company_name) if company_name is not None else None,
                    "exchange": str(exchange) if exchange is not None else None,
                    "release_dt_utc": str(release_dt_utc) if release_dt_utc is not None else None,
                    "title": str(title) if title is not None else None,
                    "chunk_index": i,
                    "char_start": start,
                    "char_end": end,
                    "text": chunk,
                }
                points_buf.append(qm.PointStruct(id=point_id, vector=[0.0] * vector_size, payload=payload))

                if len(points_buf) >= args.batch_chunks:
                    flush()

            filings_seen += 1
            if filings_seen % 25 == 0:
                print(f"Indexed filings: {filings_seen} | elapsed: {time.time() - t0:.1f}s")

        if filings_seen >= args.max_filings:
            break

    flush()
    print(f"Done. Indexed filings: {filings_seen} | elapsed: {time.time() - t0:.1f}s")
    print(f"Collection: {args.collection}")

if __name__ == "__main__":
    main()
