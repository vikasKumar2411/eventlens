import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Light cleanup: keep paragraphs, remove obvious separators
SEP_PATTERNS = [
    r"^===.*?===\s*$",  # === MAIN 8-K FILING ===
]

def clean_raw_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for line in s.split("\n"):
        if any(re.match(pat, line.strip()) for pat in SEP_PATTERNS):
            continue
        lines.append(line)
    s = "\n".join(lines)

    # Collapse excessive blank lines
    s = re.sub(r"\n{4,}", "\n\n\n", s).strip()
    return s


def preprocess_chunk(df: pd.DataFrame) -> pd.DataFrame:
    expected = [
        "sec_accession_number",
        "release_datetime",
        "title",
        "sec_filing_type",
        "keywords",
        "exchange",
        "symbol",
        "company_name",
        "excerpt",
        "raw_text",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = None

    df["sec_accession_number"] = df["sec_accession_number"].astype(str).str.strip()

    # Parse tz-aware datetime and normalize to UTC
    dt = pd.to_datetime(df["release_datetime"], errors="coerce", utc=True)
    df["release_dt_utc"] = dt
    df["release_date"] = df["release_dt_utc"].dt.date.astype("string")
    df["year"] = df["release_dt_utc"].dt.year.astype("Int64")
    df["month"] = df["release_dt_utc"].dt.month.astype("Int64")

    df["raw_text_clean"] = df["raw_text"].apply(clean_raw_text)

    for c in ["title", "excerpt", "keywords", "exchange", "symbol", "company_name"]:
        df[c] = df[c].astype("string")

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to raw CSV")
    ap.add_argument("--out_dir", required=True, help="Output directory for parquet partitions")
    ap.add_argument("--chunksize", type=int, default=50_000, help="Rows per chunk")
    ap.add_argument("--partition", action="store_true", help="Partition parquet by year/month")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    total_in = 0
    total_out = 0

    reader = pd.read_csv(csv_path, chunksize=args.chunksize, low_memory=False)

    for i, df in enumerate(tqdm(reader, desc="Preprocessing CSV chunks")):
        total_in += len(df)
        df = preprocess_chunk(df)

        df = df[df["sec_accession_number"].notna() & (df["sec_accession_number"] != "nan")]

        # De-dup across chunks
        keep_mask = []
        for acc in df["sec_accession_number"].tolist():
            if acc in seen:
                keep_mask.append(False)
            else:
                seen.add(acc)
                keep_mask.append(True)
        df = df[pd.Series(keep_mask, index=df.index)]
        total_out += len(df)

        if args.partition:
            for (y, m), g in df.groupby(["year", "month"], dropna=False):
                py = int(y) if pd.notna(y) else -1
                pm = int(m) if pd.notna(m) else -1
                pdir = out_dir / f"year={py}" / f"month={pm}"
                pdir.mkdir(parents=True, exist_ok=True)
                out_path = pdir / f"part-{i:05d}.parquet"
                g.to_parquet(out_path, index=False)
        else:
            out_path = out_dir / f"part-{i:05d}.parquet"
            df.to_parquet(out_path, index=False)

    manifest = out_dir / "_manifest.txt"
    manifest.write_text(
        f"input_csv={csv_path}\nrows_in={total_in}\nrows_out={total_out}\nunique_accessions={len(seen)}\n",
        encoding="utf-8",
    )

    print(f"\nDone.\nRows in: {total_in}\nRows out (deduped): {total_out}\nUnique accessions: {len(seen)}")
    print(f"Output: {out_dir}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
