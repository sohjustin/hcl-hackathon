"""
Streamlit chatbot UI for the wealth advisor RAG assistant.

Run with:
    streamlit run app.py
"""

import streamlit as st
from rag import retrieve, generate, log_result

st.set_page_config(page_title="Wealth Advisor Assistant")
st.title("Wealth Advisor Assistant")

if "history" not in st.session_state:
    st.session_state.history = []


def chunk_label(meta: dict) -> str:
    """
    Build a distinguishing label for one retrieved chunk, e.g.
    'fund_factsheet_safe.pdf — chunk 2', 'clients_portfolio.csv — row 14',
    or 'client_correspondence.json — record 4'.
    """
    source = meta.get("source", "unknown")
    doc_type = meta.get("doc_type")
    if doc_type == "pdf" and "chunk_index" in meta:
        return f"{source} — chunk {meta['chunk_index'] + 1}"
    if doc_type == "json" and "record_index" in meta:
        label = f"{source} — record {meta['record_index'] + 1}"
        if meta.get("chunk_index", 0) > 0:
            label += f", chunk {meta['chunk_index'] + 1}"
        return label
    if doc_type == "csv" and "row_index" in meta:
        return f"{source} — row {meta['row_index'] + 1}"
    return source


def render_sources(chunks, metadatas):
    if not chunks:
        return
    with st.expander(f"Sources & relevant paragraphs ({len(chunks)})"):
        for i, (doc, meta) in enumerate(zip(chunks, metadatas)):
            st.markdown(f"**{chunk_label(meta)}**")
            st.caption(doc)
            if i < len(chunks) - 1:
                st.divider()


# Replay previous turns
for i, turn in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        render_sources(turn["chunks"], turn["metadatas"])
        if st.button("Log this exchange for evaluation", key=f"log_{i}"):
            log_result(turn["query"], turn["answer"], turn["chunks"], turn["metadatas"])
            st.success("Logged to eval_log.csv")

# Handle new input
query = st.chat_input("Ask a question...")

if query:
    with st.chat_message("user"):
        st.write(query)

    results = retrieve(query, k=5)
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    answer = generate(query, chunks, metadatas)

    with st.chat_message("assistant"):
        st.write(answer)
        render_sources(chunks, metadatas)

    st.session_state.history.append({
        "query": query,
        "answer": answer,
        "chunks": chunks,
        "metadatas": metadatas,
    })

    st.rerun()