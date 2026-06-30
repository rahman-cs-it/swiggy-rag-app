"""
app.py
------
Streamlit UI for the Swiggy Annual Report RAG system.

Run with:
    streamlit run src/app.py
"""

import streamlit as st
from rag import SwiggyRAG

st.set_page_config(page_title="Swiggy Annual Report Q&A", page_icon="🛵")

st.title("🛵 Swiggy Annual Report — RAG Q&A")
st.caption("Ask questions about Swiggy's FY 2023-24 Annual Report. "
           "Answers are generated strictly from the document content.")


@st.cache_resource
def load_rag():
    return SwiggyRAG()


try:
    rag = load_rag()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

query = st.text_input("Your question", placeholder="e.g. What was Swiggy's total revenue in FY24?")
top_k = st.slider("Number of context chunks to retrieve", min_value=2, max_value=10, value=5)

if st.button("Ask") and query.strip():
    with st.spinner("Retrieving context and generating answer ..."):
        result = rag.answer(query, top_k=top_k)

    st.subheader("Answer")
    st.write(result["answer"])

    with st.expander("Supporting context"):
        for c in result["contexts"]:
            st.markdown(f"**Page {c['page_number']}** (similarity score: {c['score']:.3f})")
            st.write(c["text"])
            st.divider()
