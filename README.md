# Wealth Advisor RAG Assistant

A retrieval-augmented generation (RAG) chatbot that answers questions about fund fact sheets, client portfolios, and client correspondence, using only the documents it's given as context. Retrieval runs against a local Chroma vector store; embeddings come from Gemini; answer generation runs locally via a Hugging Face model (Qwen2-7B-Instruct).

## How it works

1. **Ingest** (`ingest.py`) — reads every file under `data/`, converts it to plain text, splits it into overlapping chunks, embeds each chunk with Gemini, and stores it in a local Chroma collection (`./chroma_db`).
2. **Retrieve** (`rag.py: retrieve`) — embeds the user's question and pulls the top-k most similar chunks. If the question names a client ID (e.g. `CL015`), a second client-scoped query runs alongside the general one and its results are merged in first, so a client's own records aren't crowded out by a more verbose but less specific PDF chunk.
3. **Generate** (`rag.py: generate`) — feeds the retrieved chunks to the local Qwen2-7B model with a system prompt that requires every claim to be cited in the form `[Source: filename]`, and requires the model to say so explicitly if the sources don't have enough information.
4. **Serve** (`app.py`) — a Streamlit chat UI that shows the answer plus an expandable "Sources & relevant paragraphs" panel, with a per-exchange button to log the exchange to `eval_log.csv`.
5. **Evaluate** (`eval.py`) — runs the pipeline against a golden dataset and scores retrieval and citation quality automatically, plus a local-LLM-judged factual-correctness pass flagged for manual spot-checking.

## Project structure

```
.
├── app.py              # Streamlit UI
├── rag.py              # embed / retrieve / generate / logging — shared by app.py and ingest.py
├── ingest.py           # one-time / re-run-able ingestion pipeline
├── eval.py             # automated evaluation against the golden dataset
├── requirements.txt
├── .env                # GOOGLE_API_KEY (not committed — see env.example)
├── data/               # source documents (see "Data layout" below)
├── chroma_db/          # persistent Chroma vector store (created on first ingest)
├── golden_dataset_for_RAG_evaluation.xlsx   # hand-authored test set for eval.py
├── eval_log.csv        # exchanges logged from the UI for later scoring
└── eval_report.csv     # output of eval.py
```

### Data layout

`ingest.py` expects:

```
data/
├── *.pdf                     # fund fact sheets etc. — read directly from data/
└── letters/
    ├── *.json                # e.g. client_correspondence.json, clients_portfolio.json
    └── *.csv                 # e.g. clients_portfolio.csv, transactions.csv
```

- PDFs are read page-by-page, cleaned (hyphenation fixes, whitespace collapse), and chunked at 500 words with 50 words of overlap.
- JSON records are flattened into readable `key: value` text (nested objects and lists included), one record per chunk group.
- CSV rows are flattened into `column: value` text, one row per chunk.
- Every chunk gets a `source`, `doc_type`, and position metadata (`chunk_index` / `record_index` / `row_index`), plus `client_id` when one is present — this is what powers both the client-scoped retrieval boost and the citation labels in the UI.
- Chunk IDs are a hash of `source + text`, so re-running `ingest.py` after adding or editing files is safe: unchanged chunks upsert in place, edited text gets a new ID.

## Setup

1. **Install dependencies**

   ```
   pip install -r requirements.txt
   ```

2. **Configure your API key** — copy `env.example` to `.env` and set:

   ```
   GOOGLE_API_KEY=your_actual_key_here
   ```

   This key is used only for Gemini embeddings (`models/gemini-embedding-001`); generation is fully local and needs no API key.

3. **Hardware for generation** — `rag.py` loads `Qwen/Qwen2-7B-Instruct` via `transformers` at import time (first run downloads ~15 GB of weights). It auto-selects `mps` (Apple Silicon), `cuda`, or falls back to CPU, and needs roughly 14 GB+ of RAM/VRAM to hold the model comfortably.

4. **Add your documents** to `data/` following the layout above.

## Usage

**Build the vector store:**
```
python ingest.py
```
Re-run any time source files change.

**Run the chatbot:**
```
streamlit run app.py
```
Ask a question in the chat box; each answer comes with an expandable list of the retrieved chunks it's grounded in. Use the "Log this exchange for evaluation" button under any answer to append it to `eval_log.csv`.

**Run automated evaluation:**
```
python eval.py
```
Reads `golden_dataset_for_RAG_evaluation.xlsx` (sheet `Golden Dataset`, skipping the `EX` example row), runs each sample query through the same retrieve/generate pipeline, and writes `eval_report.csv` with:

| Metric | What it measures |
|---|---|
| Context Precision | Share of retrieved sources that were actually expected |
| Context Recall | Share of expected sources that were actually retrieved |
| Citation Grounding | Share of `[Source: ...]` tags in the answer that match a retrieved source (catches citing something never given to the model) |
| LLM-judged correctness | Local Qwen2-7B model's Yes/No/Partial verdict against the reference answer — **a triage signal only**; spot-check "No"/"Partial" rows by hand, since a same-family model grading its own output isn't a substitute for human review |

## Notes and caveats

- Embeddings and generation are on different providers by design: Hugging Face has no equivalent to Gemini's task-typed (`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`) embedding model, so embeddings stayed on Gemini while generation moved local.
- The UI's "Sources & relevant paragraphs" panel shows whatever was retrieved for that turn, not filtered down to only cited sources — `rag.py` has a `filter_relevant()` helper for that narrower view if you want to wire it in.
- `eval_log.csv` is written in the same column shape as the golden dataset so logged production exchanges can be merged back in for scoring later.
