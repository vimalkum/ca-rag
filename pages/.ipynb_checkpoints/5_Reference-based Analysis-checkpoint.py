import os
import json
import tempfile
from typing import Dict

import numpy as np
import streamlit as st
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

from modules.doc_loader import DocumentLoader
from modules.evaluator import (
    DSA_QUESTIONS,
    build_chunks_from_langchain_docs,
    evaluate_document,
)

try:
    from modules.retrieval import Chunk
except Exception:
    Chunk = None


# ---------------------------
# Helpers: load PDFs -> chunks
# ---------------------------
def pdf_to_chunks(pdf_path: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        dst = os.path.join(tmpdir, os.path.basename(pdf_path))
        with open(pdf_path, "rb") as fin, open(dst, "wb") as fout:
            fout.write(fin.read())

        loader = DocumentLoader(dataset_path=tmpdir)
        docs = loader.load_documents()
        chunks = build_chunks_from_langchain_docs(docs)
        return chunks


def chunks_to_text(chunks) -> str:
    parts = []
    for c in chunks:
        t = getattr(c, "text", None)
        if t is None and isinstance(c, dict):
            t = c.get("text")
        if t:
            parts.append(str(t))
    return "\n".join(parts).strip()


# ---------------------------
# Metric 1: Cosine similarity
# ---------------------------
@st.cache_resource(show_spinner=False)
def _load_embedder(model_name: str):
    return SentenceTransformer(model_name)


def cosine_similarity_chunks(gen_chunks, gt_chunks, model_name: str = "all-MiniLM-L6-v2") -> float:
    model = _load_embedder(model_name)

    gen_texts = [getattr(c, "text", "") for c in gen_chunks]
    gt_texts = [getattr(c, "text", "") for c in gt_chunks]

    gen_texts = [t for t in gen_texts if t and t.strip()]
    gt_texts = [t for t in gt_texts if t and t.strip()]

    if not gen_texts or not gt_texts:
        return 0.0

    gen_emb = model.encode(gen_texts, normalize_embeddings=True)
    gt_emb = model.encode(gt_texts, normalize_embeddings=True)

    gen_doc = np.mean(gen_emb, axis=0)
    gt_doc = np.mean(gt_emb, axis=0)

    return float(np.dot(gen_doc, gt_doc) / (np.linalg.norm(gen_doc) * np.linalg.norm(gt_doc) + 1e-12))


# ---------------------------
# Metric 2: ROUGE on chunk-text
# ---------------------------
def rouge_on_chunk_text(gen_chunks, gt_chunks) -> Dict[str, float]:
    pred = chunks_to_text(gen_chunks)
    ref = chunks_to_text(gt_chunks)

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=True
    )
    s = scorer.score(ref, pred)

    return {
        "ROUGE-1": s["rouge1"].fmeasure,
        "ROUGE-2": s["rouge2"].fmeasure,
        "ROUGE-L": s["rougeL"].fmeasure,
        "ROUGE-LSUM": s["rougeLsum"].fmeasure,
    }


# ---------------------------
# Optional: Your QA/Coverage eval
# ---------------------------
def coverage_eval(chunks, top_k: int = 6) -> Dict:
    questions = list(DSA_QUESTIONS)
    report = evaluate_document(chunks, questions, k=top_k)
    return report


# ---------------------------
# Full aligned evaluation
# ---------------------------
def evaluate_pair_aligned(gen_pdf: str, gt_pdf: str, top_k: int = 6, run_coverage: bool = True) -> Dict:
    gen_chunks = pdf_to_chunks(gen_pdf)
    gt_chunks = pdf_to_chunks(gt_pdf)

    out = {}
    out["Cosine"] = cosine_similarity_chunks(gen_chunks, gt_chunks)
    out.update(rouge_on_chunk_text(gen_chunks, gt_chunks))

    if run_coverage:
        out["Generated_Coverage_Report"] = coverage_eval(gen_chunks, top_k=top_k)
        out["GroundTruth_Coverage_Report"] = coverage_eval(gt_chunks, top_k=top_k)

    return out


# ---------------------------
# Streamlit UI
# ---------------------------
def _save_uploaded_file(uploaded_file) -> str:
    """
    Save an st.file_uploader object to a temporary file and return its path.
    """
    suffix = os.path.splitext(uploaded_file.name)[-1].lower() or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name


def main():
    st.set_page_config(page_title="Reference-based Evaluator", layout="wide")
    st.title("Reference-based Similarity Measure")

    with st.sidebar:
        st.header("Inputs")
        gen_file = st.file_uploader("Generated DSA", type=["pdf"], key="gen_pdf")
        gt_file = st.file_uploader("Ground Truth DSA", type=["pdf"], key="gt_pdf")

        st.divider()
        st.header("Settings")
        top_k = st.slider("Coverage top_k", min_value=1, max_value=20, value=6, step=1)
        run_coverage = st.checkbox("Run coverage eval (can be slow)", value=False)
        embed_model = st.text_input("Embedding model", value="all-MiniLM-L6-v2")

        run_btn = st.button("Run evaluation", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Upload both PDFs, set options, then click **Run evaluation**.")
        return

    if gen_file is None or gt_file is None:
        st.error("Please upload **both** the Generated PDF and the Ground Truth PDF.")
        return

    gen_path = gt_path = None
    try:
        with st.status("Saving uploads…", expanded=False):
            gen_path = _save_uploaded_file(gen_file)
            gt_path = _save_uploaded_file(gt_file)
            
        with st.spinner("Running chunking + metrics…"):
            # Chunk once, compute with selected model
            gen_chunks = pdf_to_chunks(gen_path)
            gt_chunks = pdf_to_chunks(gt_path)

            metrics = {
                "Cosine": cosine_similarity_chunks(gen_chunks, gt_chunks, model_name=embed_model),
                **rouge_on_chunk_text(gen_chunks, gt_chunks),
            }

            if run_coverage:
                metrics["Generated_Coverage_Report"] = coverage_eval(gen_chunks, top_k=top_k)
                metrics["GroundTruth_Coverage_Report"] = coverage_eval(gt_chunks, top_k=top_k)

        # Display
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Cosine", f"{metrics['Cosine']:.4f}")
        c2.metric("ROUGE-1", f"{metrics['ROUGE-1']:.4f}")
        c3.metric("ROUGE-2", f"{metrics['ROUGE-2']:.4f}")
        c4.metric("ROUGE-L", f"{metrics['ROUGE-L']:.4f}")
        c5.metric("ROUGE-LSUM", f"{metrics['ROUGE-LSUM']:.4f}")

        st.divider()

        # JSON + downloads
        base_report = {
            "Cosine": metrics["Cosine"],
            "ROUGE-1": metrics["ROUGE-1"],
            "ROUGE-2": metrics["ROUGE-2"],
            "ROUGE-L": metrics["ROUGE-L"],
            "ROUGE-LSUM": metrics["ROUGE-LSUM"],
        }

        st.subheader("Metrics JSON")
        st.json(base_report)

        st.download_button(
            "Download metrics.json",
            data=json.dumps(base_report, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name="metrics.json",
            mime="application/json",
            use_container_width=False,
        )

        if run_coverage:
            st.subheader("Coverage reports")
            cov = {
                "generated": metrics.get("Generated_Coverage_Report"),
                "ground_truth": metrics.get("GroundTruth_Coverage_Report"),
            }
            with st.expander("View coverage JSON", expanded=False):
                st.json(cov)

            st.download_button(
                "Download coverage_reports.json",
                data=json.dumps(cov, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name="coverage_reports.json",
                mime="application/json",
            )

    finally:
        # Cleanup temp files we created
        for p in (gen_path, gt_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


if __name__ == "__main__":
    main()
