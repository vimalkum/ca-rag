import os
import json
import tempfile
import streamlit as st
import pandas as pd

from modules.doc_loader import DocumentLoader
from modules.evaluator import (
    DSA_QUESTIONS,
    build_chunks_from_langchain_docs,
    evaluate_document
)
from modules.retrieval import Chunk

st.set_page_config(page_title="Policy / DSA Evaluation", layout="wide")
st.title("Policy / Data Sharing Agreement Evaluator")

with st.sidebar:
    st.header("Inputs")
    mode = st.radio("Input type", ["Upload PDF", "Paste text"], index=0)
    #use_cuad = st.checkbox("Include CUAD questions", value=True)
    #use_dsa = st.checkbox("Include DSA checklist questions", value=True)
    top_k = st.slider("Retriever top-k", 3, 12, 6)

    st.divider()
    st.caption("LLM config")
    st.text_input("OPENAI_MODEL (optional)", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), key="openai_model")
    st.caption("Set OPENAI_API_KEY in your environment for LLM extraction.")

def _load_from_pdf(upload) -> list[Chunk]:
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, upload.name)
        with open(pdf_path, "wb") as f:
            f.write(upload.getbuffer())

        loader = DocumentLoader(dataset_path=tmpdir)
        docs = loader.load_documents()
        return build_chunks_from_langchain_docs(docs)

def _load_from_text(text: str) -> list[Chunk]:
    # Simple chunking aligned with your chunk sizes
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    parts = splitter.split_text(text)
    return [Chunk(text=p, source="pasted_text", page=None) for p in parts]

chunks = None
if mode == "Upload PDF":
    upload = st.file_uploader("Upload a generated policy/contract PDF", type=["pdf"])
    if upload is not None:
        chunks = _load_from_pdf(upload)
else:
    text = st.text_area("Paste policy/DSA text", height=250)
    if text and text.strip():
        chunks = _load_from_text(text)

if chunks:
    st.success(f"Loaded {len(chunks)} chunks.")
    questions = []
    questions.extend(DSA_QUESTIONS)
    # CUAD is big; sample or filter for speed
    st.write(f"Running evaluation on **{len(questions)}** questions…")
    report = evaluate_document(chunks, questions, k=top_k)

    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Coverage", f"{report['coverage']*100:.1f}%")
    c2.metric("Answered", report["answered"])
    c3.metric("Total Questions", report["total"])

    st.subheader("Results")
    df = pd.DataFrame(report["results"])
    # Make table friendlier
    view = df[["label", "answered", "question", "answer", "risk_notes"]].copy()
    st.dataframe(view, width='stretch')

    st.subheader("Evidence Viewer")
    idx = st.number_input("Pick a row index", min_value=0, max_value=len(df)-1, value=0, step=1)
    row = report["results"][int(idx)]
    st.markdown(f"**Label:** {row['label']}")
    st.markdown(f"**Question:** {row['question']}")
    st.markdown(f"**Answered:** {row['answered']}")
    st.markdown(f"**Answer:** {row.get('answer','')}")
    st.markdown(f"**Risk notes:** {row.get('risk_notes','')}")
    st.markdown("**Evidence:**")
    ev = row.get("evidence", []) or []
    if not ev:
        st.info("No evidence extracted.")
    else:
        for i, e in enumerate(ev, 1):
            st.code(e)

    st.subheader("Download report")
    st.download_button(
        "Download JSON",
        data=json.dumps(report, indent=2).encode("utf-8"),
        file_name="policy_eval_report.json",
        mime="application/json",
    )
else:
    st.info("Upload a PDF or paste text to begin.")
