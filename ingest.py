"""
Ingestion pipeline: reads everything under ./data, converts it to plain text,
chunks it, embeds each chunk with Gemini, and stores it in the local Chroma
vector store (./chroma_db).

Run this once after dropping the dataset into ./data, and re-run any time
the source files change. Chroma's .add() upserts by id, so re-running is safe.

    python ingest.py
"""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from rag import embed, collection

DATA_DIR = Path("data")
CHUNK_SIZE = 500      # words per chunk
CHUNK_OVERLAP = 50    # words of overlap between consecutive chunks


def clean_text(text: str) -> str:
    """
    Normalize whitespace and fix common PDF-extraction artifacts before chunking.
    """
    if not text:
        return ""
    # Rejoin words broken across a line by a hyphen, e.g. "invest-\nment" -> "investment"
    text = re.sub(r"-\n", "", text)
    # Collapse all whitespace (newlines, tabs, repeated spaces) to single spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple word-based sliding-window chunker."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return clean_text("\n".join(pages))


def is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def flatten_json_record(record, prefix: str = "") -> str:
    """
    Turn a JSON value into a readable text block, skipping null/empty fields
    and expanding nested lists of records instead of dumping raw JSON, e.g.
    'client_id: CL002; risk_profile: Conservative; holdings: [fund: X, allocation_pct: 15 | fund: Y, allocation_pct: 20]'
    """
    if isinstance(record, dict):
        parts = []
        for key, value in record.items():
            if is_empty(value):
                continue
            label = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict):
                parts.append(flatten_json_record(value, prefix=f"{label}."))
            elif isinstance(value, list):
                items = [flatten_json_record(item) for item in value if not is_empty(item)]
                parts.append(f"{label}: [{' | '.join(items)}]")
            else:
                parts.append(f"{label}: {value}")
        return "; ".join(parts)
    return str(record)


def ingest_pdfs():
    pdf_dir = DATA_DIR  # PDFs sit directly in data/, not data/pdfs/
    if not pdf_dir.exists():
        print(f"Skipping {pdf_dir} — not found")
        return
    for pdf_path in pdf_dir.glob("*.pdf"):
        print(f"Ingesting PDF: {pdf_path.name}")
        text = extract_pdf_text(pdf_path)
        for i, chunk in enumerate(chunk_text(text)):
            add_chunk(
                text=chunk,
                source=pdf_path.name,
                doc_type="pdf",
                extra_metadata={"chunk_index": i},
            )


def ingest_json():
    json_dir = DATA_DIR / "letters"  # client_correspondence.json, clients_portfolio.json
    if not json_dir.exists():
        print(f"Skipping {json_dir} — not found")
        return
    for json_path in json_dir.glob("*.json"):
        print(f"Ingesting JSON: {json_path.name}")
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        records = data if isinstance(data, list) else [data]
        for i, record in enumerate(records):
            text = flatten_json_record(record)
            client_id = record.get("client_id") if isinstance(record, dict) else None
            for j, chunk in enumerate(chunk_text(text)):
                extra = {"record_index": i, "chunk_index": j}
                if client_id:
                    extra["client_id"] = client_id
                add_chunk(
                    text=chunk,
                    source=json_path.name,
                    doc_type="json",
                    extra_metadata=extra,
                )


def ingest_csv():
    csv_dir = DATA_DIR / "letters"  # clients_portfolio.csv, transactions.csv
    if not csv_dir.exists():
        print(f"Skipping {csv_dir} — not found")
        return
    for csv_path in csv_dir.glob("*.csv"):
        print(f"Ingesting CSV: {csv_path.name}")
        df = pd.read_csv(csv_path)
        for i, row in df.iterrows():
            text = "; ".join(
                f"{col}: {row[col]}" for col in df.columns if not is_empty(row[col])
            )
            extra = {"row_index": int(i)}
            if "client_id" in df.columns:
                extra["client_id"] = str(row["client_id"])
            # Rows are usually short enough to be one chunk each
            add_chunk(
                text=text,
                source=csv_path.name,
                doc_type="csv",
                extra_metadata=extra,
            )


def add_chunk(text: str, source: str, doc_type: str, extra_metadata: dict):
    text = text.strip()
    if not text:
        return
    metadata = {"source": source, "doc_type": doc_type, **extra_metadata}
    # Hash of source + text so re-running ingest on unchanged content upserts
    # the same id instead of piling up duplicates; edited text gets a new id.
    chunk_id = hashlib.sha256(f"{source}:{text}".encode("utf-8")).hexdigest()
    collection.add(
        ids=[chunk_id],
        embeddings=[embed(text, task_type="RETRIEVAL_DOCUMENT")],
        documents=[text],
        metadatas=[metadata],
    )


if __name__ == "__main__":
    ingest_pdfs()
    ingest_json()
    ingest_csv()
    print(f"Done. Collection now has {collection.count()} chunks.")