"""
Core RAG functions: embed, retrieve, generate, log_result.
Used by both ingest.py (embedding at index time) and app.py (the UI).
"""

import os
import re
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set — check your .env file")

genai.configure(api_key=GOOGLE_API_KEY)

# Persistent local vector store — created on first run under ./chroma_db
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("wealth_docs")

# Embeddings still go through Gemini — Hugging Face has no equivalent to
# Gemini's task-typed embedding model, and swapping this wasn't part of the ask.
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Generation now runs locally via Hugging Face transformers instead of an API call.
HF_MODEL_NAME = "Qwen/Qwen2-7B-Instruct"

if torch.backends.mps.is_available():
    DEVICE = "mps"       # Apple Silicon GPU
    TORCH_DTYPE = torch.float16
elif torch.cuda.is_available():
    DEVICE = "cuda"
    TORCH_DTYPE = torch.float16
else:
    DEVICE = "cpu"        # will work but is noticeably slow for a 7B model
    TORCH_DTYPE = torch.float32

print(f"Loading {HF_MODEL_NAME} on {DEVICE} — first run downloads ~15GB of weights "
      f"and can take a while; needs ~14GB+ RAM/VRAM to hold the model.")

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

generation_model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_NAME,
    torch_dtype=TORCH_DTYPE,
).to(DEVICE)
generation_model.eval()

# Matches client IDs like CL002, cl15, CL015 anywhere in a query.
CLIENT_ID_PATTERN = re.compile(r"\bCL0*(\d{1,4})\b", re.IGNORECASE)


def embed(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """
    Embed a piece of text with Gemini.
    task_type: "RETRIEVAL_DOCUMENT" when indexing chunks,
               "RETRIEVAL_QUERY" when embedding a user question.
    """
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type=task_type,
    )
    return result["embedding"]


def extract_client_id(query: str) -> str | None:
    """Pull a client ID like 'CL015' out of a natural-language question, if present."""
    match = CLIENT_ID_PATTERN.search(query)
    if not match:
        return None
    return f"CL{match.group(1).zfill(3)}"


def _query_chroma(query_embedding, k: int, where: dict | None = None) -> dict:
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )


def retrieve(query: str, k: int = 5, where: dict | None = None) -> dict:
    """
    Embed the query and pull the top-k most similar chunks from Chroma.

    If `where` isn't given and the query names a client (e.g. "CL015"), this
    runs a second, client-scoped retrieval and merges it in ahead of the
    general results. Without this, a client's own JSON/correspondence
    records can lose out on pure text similarity to a more verbose PDF
    chunk that happens to share more vocabulary with the question, even
    when the client-specific record is exactly what's needed.
    """
    query_embedding = embed(query, task_type="RETRIEVAL_QUERY")

    if where is not None:
        return _query_chroma(query_embedding, k, where)

    general = _query_chroma(query_embedding, k)

    client_id = extract_client_id(query)
    if not client_id:
        return general

    client_specific = _query_chroma(query_embedding, k, where={"client_id": client_id})

    # Merge client-specific results first (guaranteed inclusion), then fill
    # with general results, deduplicating by chunk id, capped so context
    # doesn't grow unbounded.
    seen_ids = set()
    merged_ids, merged_docs, merged_metas, merged_dists = [], [], [], []

    def add(result: dict):
        for _id, doc, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            if _id in seen_ids:
                continue
            seen_ids.add(_id)
            merged_ids.append(_id)
            merged_docs.append(doc)
            merged_metas.append(meta)
            merged_dists.append(dist)

    add(client_specific)
    add(general)

    limit = k + 3
    return {
        "ids": [merged_ids[:limit]],
        "documents": [merged_docs[:limit]],
        "metadatas": [merged_metas[:limit]],
        "distances": [merged_dists[:limit]],
    }


def format_context(chunks: list[str], metadatas: list[dict]) -> str:
    """Number each chunk and label it with its source, so the model can cite it by name."""
    labeled = []
    for i, (doc, meta) in enumerate(zip(chunks, metadatas), start=1):
        source = meta.get("source", "unknown")
        labeled.append(f"[{i}] Source: {source}\n{doc}")
    return "\n\n".join(labeled)


def generate(query: str, chunks: list[str], metadatas: list[dict]) -> str:
    """
    Answer the query using only the retrieved chunks as context, via a
    locally-run Qwen2-7B-Instruct, citing the source filename after every
    claim so the UI can show only the files actually used rather than
    everything that was retrieved.
    """
    context = format_context(chunks, metadatas)
    system_prompt = (
        "You are a wealth advisor assistant. Answer the question using ONLY "
        "the numbered sources you are given.\n\n"
        "After every factual claim, cite the source it came from in the "
        "exact format [Source: filename]. If a sentence draws on multiple "
        "sources, cite each one, e.g. [Source: a.pdf, Source: b.pdf]. Only "
        "cite a source if you actually used it — do not cite a source you "
        "didn't draw on.\n\n"
        "If the sources do not contain enough information to answer "
        "confidently, say so explicitly rather than guessing."
    )
    user_prompt = f"Sources:\n{context}\n\nQuestion: {query}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = generation_model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Only decode the newly generated tokens, not the echoed prompt
    response_ids = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def extract_cited_sources(answer_text: str) -> set[str]:
    """Pull every '[Source: filename]' tag out of the generated answer."""
    return {s.strip() for s in re.findall(r"\[Source:\s*([^\],\]]+)\]", answer_text)}


def filter_relevant(chunks: list[str], metadatas: list[dict], answer_text: str):
    """
    Keep only the retrieved chunks whose source was actually cited in the
    answer. If the model didn't cite anything (formatting slip, or a short
    answer with no claims), fall back to returning everything retrieved
    rather than silently showing nothing.
    """
    cited = extract_cited_sources(answer_text)
    if not cited:
        return chunks, metadatas

    filtered_chunks, filtered_metadatas = [], []
    for chunk, meta in zip(chunks, metadatas):
        if meta.get("source", "") in cited:
            filtered_chunks.append(chunk)
            filtered_metadatas.append(meta)

    if not filtered_chunks:  # citations didn't match any retrieved source name
        return chunks, metadatas
    return filtered_chunks, filtered_metadatas


def log_result(
    query: str,
    answer: str,
    chunks: list[str],
    metadatas: list[dict],
    path: str = "eval_log.csv",
) -> None:
    """
    Append one exchange in the same column shape as the golden dataset,
    so it can be merged back in for scoring.
    """
    row = {
        "Sample query": query,
        "Response generated by the chatbot": answer,
        "Source(s) referenced by the chatbot": "; ".join(
            m.get("source", "") for m in metadatas
        ),
        "Paragraph(s) referenced by the chatbot": " | ".join(chunks),
    }
    pd.DataFrame([row]).to_csv(
        path, mode="a", header=not os.path.exists(path), index=False
    )