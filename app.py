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


def render_sources(chunks, metadatas):
    with st.expander("Sources & retrieved paragraphs"):
        for doc, meta in zip(chunks, metadatas):
            st.markdown(f"**{meta.get('source', 'unknown')}**")
            st.caption(doc[:400] + ("..." if len(doc) > 400 else ""))


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
    answer = generate(query, chunks)

    with st.chat_message("assistant"):
        st.write(answer)
        render_sources(chunks, metadatas)

    st.session_state.history.append({
        "query": query,
        "answer": answer,
        "chunks": chunks,
        "metadatas": metadatas,
    })

    # Rerun so the new turn's log button gets picked up by the replay loop above,
    # with a stable key based on its position in history.
    st.rerun()