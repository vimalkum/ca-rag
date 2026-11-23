import os
import re
import gc
import streamlit as st
import csv,json
from datetime import datetime
from typing import List, Tuple, Dict
from modules.doc_loader import DocumentLoader
from modules.prompt_manager import PromptManager
from modules.embeddings import EmbeddingDB
from modules.llm_manager import LanguageModel
from modules.generator import DSAGenerator

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm

try:
    import torch
    from rapidfuzz import fuzz
    def similarity(a: str, b: str) -> float:
        # token_set_ratio is robust to word order/punctuation changes
        return fuzz.token_set_ratio(a, b) / 100.0
    _SIM_BACKEND = "rapidfuzz.token_set_ratio"
except Exception:
    torch = None
    from difflib import SequenceMatcher
    def similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()
    _SIM_BACKEND = "difflib.SequenceMatcher"

models = {'Llama-3.1-8B': "meta-llama/Llama-3.1-8B-Instruct",
          'Llama-3.2.3B': "meta-llama/Llama-3.2-3B-Instruct",
          'Mistral-7B':"mistralai/Mistral-7B-Instruct-v0.3"}
#model_path = "meta-llama/Llama-3.1-8B-Instruct"
#model_path = "mistralai/Mistral-7B-Instruct-v0.3"
        
class CA_RAG_App:
    def __init__(self):
        self.embedding_path = "sentence-transformers/all-MiniLM-l6-v2"
        self.output_dir = "./Output/"
        os.makedirs(self.output_dir, exist_ok=True)

        # Your wrapper uses this; keep it but be aware it may be provider-specific
        self.device = 1

    @staticmethod
    def _safe_filename(name: str, default: str = "DSA_v2") -> str:
        name = (name or "").strip()
        if not name:
            return default
        # Remove unsafe chars
        name = re.sub(r"[^a-zA-Z0-9_\- ]+", "", name).strip()
        name = name.replace(" ", "_")
        return name or default

    @staticmethod
    def _dict_to_pdf_flowables(out_dict: dict):
        """
        Convert {title: text} into ReportLab Flowables.
        """
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Heading2"],
            spaceAfter=8,
        )
        body_style = ParagraphStyle(
            "BodyStyle",
            parent=styles["Normal"],
            leading=16,
        )

        flow = []
        for k, v in out_dict.items():
            flow.append(Paragraph(str(k), title_style))

            # Preserve line breaks for reportlab Paragraph
            body = (v or "").replace("\n", "<br/>")
            flow.append(Paragraph(body, body_style))

            flow.append(Spacer(1, 12))

        return flow
    def _save_metrics_json(self, metrics: dict, base_name: str) -> str:
        """
        Save metrics dict as JSON under output_dir.
        Returns the full path.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = self._safe_filename(base_name, default="DSA_v2")
        path = os.path.join(self.output_dir, f"{safe_name}_metrics_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        return path

    def _save_metrics_csv(self, metrics: dict, base_name: str) -> str:
        """
        Save metrics['calls'] as CSV under output_dir.
        Returns the full path.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = self._safe_filename(base_name, default="DSA_v2")
        path = os.path.join(self.output_dir, f"{safe_name}_metrics_calls_{ts}.csv")

        rows = metrics.get("calls", []) if isinstance(metrics, dict) else []
        if not rows:
            # still create an empty file with headers
            headers = ["stage", "title", "prompt_tokens", "generated_tokens", "total_tokens", "latency_s", "latency_ms", "tflops_est"]
        else:
            # union of keys across rows (stable header order preferred)
            preferred = ["stage", "title", "prompt_tokens", "generated_tokens", "total_tokens", "latency_s", "latency_ms", "tflops_est"]
            all_keys = set().union(*(r.keys() for r in rows))
            headers = preferred + [k for k in sorted(all_keys) if k not in preferred]

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        return path
    
    def normalize_title(self, s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        # optional: remove trailing punctuation
        s = re.sub(r"[^\w\s\-]", "", s)
        return s

    
    def prf(self, matched: int, predicted: int, gold: int) -> Dict[str, float]:
        precision = matched / predicted if predicted else 0.0
        recall = matched / gold if gold else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}


    def greedy_1to1_match(self,
        generated: List[str],
        reference: List[str],
        threshold: float,
    ) -> Tuple[List[Dict], List[str], List[str]]:
        """
        Greedy 1-to-1 matching: sorts all (gen, ref) pairs by similarity desc and
        picks the best non-conflicting pairs above threshold.
        Returns:
          matches: list of dicts {gen, ref, score}
          unmatched_gen
          unmatched_ref
        """
        gen = [g.strip() for g in generated if g and g.strip()]
        ref = [r.strip() for r in reference if r and r.strip()]
    
        pairs = []
        for i, g in enumerate(gen):
            for j, r in enumerate(ref):
                pairs.append((i, j, similarity(g, r)))
    
        pairs.sort(key=lambda x: x[2], reverse=True)
    
        used_g, used_r = set(), set()
        matches = []
        for i, j, score in pairs:
            if score < threshold:
                break
            if i in used_g or j in used_r:
                continue
            used_g.add(i)
            used_r.add(j)
            matches.append({"gen": gen[i], "ref": ref[j], "score": score})
    
        unmatched_gen = [g for idx, g in enumerate(gen) if idx not in used_g]
        unmatched_ref = [r for idx, r in enumerate(ref) if idx not in used_r]
        return matches, unmatched_gen, unmatched_ref
    def mean_best_similarity(self, generated: List[str], reference: List[str]) -> float:
        gen = [g.strip() for g in generated if g and g.strip()]
        ref = [r.strip() for r in reference if r and r.strip()]
        if not gen or not ref:
            return 0.0
        best_scores = [max(similarity(g, r) for r in ref) for g in gen]
        return sum(best_scores) / len(best_scores)
    def semantic_topic_coverage(
        self,
        generated: List[str],
        reference: List[str],
        threshold: float,
    ) -> Tuple[float, List[Dict]]:
        """
        Topic coverage (semantic):
        For each ground-truth title, find the best matching generated title.
        A topic is 'covered' if best_similarity >= threshold.

        Returns:
            coverage: float in [0,1]
            details: list of dicts:
                {
                  "ref": ground_truth_title,
                  "best_gen": best_matching_generated_title or None,
                  "score": best_similarity,
                  "covered": bool
                }
        """
        gen = [g.strip() for g in generated if g and g.strip()]
        ref = [r.strip() for r in reference if r and r.strip()]

        if not ref:
            return 0.0, []

        details = []
        covered_count = 0

        for r in ref:
            if not gen:
                best_score = 0.0
                best_gen = None
            else:
                # best semantic match among generated
                scores = [(g, similarity(g, r)) for g in gen]
                best_gen, best_score = max(scores, key=lambda x: x[1])

            covered = best_score >= threshold
            if covered:
                covered_count += 1

            details.append({
                "ref": r,
                "best_gen": best_gen,
                "score": best_score,
                "covered": covered,
            })

        coverage = covered_count / len(ref)
        return coverage, details

    def show_title_evaluation(self, generated_titles: List[str],
    ground_truth_titles: List[str],
    default_threshold: float = 0.80):
        
        st.markdown("## Title Evaluation")
    
        st.caption(f"Similarity backend: `{_SIM_BACKEND}`")
    
        # --- Exact match metrics (normalized set) ---
        gen_norm = [self.normalize_title(t) for t in generated_titles if t and t.strip()]
        gt_norm = [self.normalize_title(t) for t in ground_truth_titles if t and t.strip()]
    
        gen_set = set(gen_norm)
        gt_set = set(gt_norm)
    
        exact_matched = len(gen_set & gt_set)
        exact = self.prf(exact_matched, len(gen_set), len(gt_set))
    
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Generated (unique)", len(gen_set))
        c2.metric("Ground Truth (unique)", len(gt_set))
        c3.metric("Exact matches", exact_matched)
        c4.metric("Exact F1", f"{exact['f1']:.2f}")
    
        st.write(
            f"**Exact Precision:** {exact['precision']:.2f}  |  "
            f"**Exact Recall:** {exact['recall']:.2f}  |  "
            f"**Exact F1:** {exact['f1']:.2f}"
        )
    
        st.divider()
    
        # --- Semantic (fuzzy) match metrics ---
        st.markdown("### Semantic Match (1-to-1)")
        threshold = 0.75 
        '''st.slider(
            "Similarity threshold",
            min_value=0.50,
            max_value=0.95,
            value=float(default_threshold),
            step=0.01,
            help="Higher = stricter. 0.80 is a good default for section titles.",
        )'''
    
        matches, un_gen, un_gt = self.greedy_1to1_match(
            generated_titles,
            ground_truth_titles,
            threshold,
        )
        fuzzy = self.prf(
            len(matches),
            len([t for t in generated_titles if t and t.strip()]),
            len([t for t in ground_truth_titles if t and t.strip()]),
        )

        # NEW: semantic topic coverage (from GT perspective)
        coverage, cov_details = self.semantic_topic_coverage(
            generated_titles,
            ground_truth_titles,
            threshold,
        )

        mean_best = self.mean_best_similarity(generated_titles, ground_truth_titles)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Matched pairs (1-to-1)", len(matches))
        c2.metric("Semantic Precision", f"{fuzzy['precision']:.2f}")
        c3.metric("Semantic Recall", f"{fuzzy['recall']:.2f}")
        c4.metric("Semantic F1", f"{fuzzy['f1']:.2f}")

        st.write(f"**Mean best-match similarity (generated→GT):** {mean_best:.2f}")
        st.write(f"**Semantic topic coverage (GT topics covered):** {coverage:.2f}")
    
        # --- Detailed tables / lists ---
        with st.expander("Matched pairs (generated ↔ ground truth)", expanded=True):
            if matches:
                # Make a simple table without pandas
                for m in sorted(matches, key=lambda x: x["score"], reverse=True):
                    st.write(f"- **{m['gen']}**  ↔  {m['ref']}  (score={m['score']:.2f})")
            else:
                st.info("No matches above the threshold.")
    
        with st.expander("Unmatched generated titles", expanded=False):
            if un_gen:
                for t in un_gen:
                    st.write(f"- {t}")
            else:
                st.write("None ✅")
    
        with st.expander("Unmatched ground truth titles", expanded=False):
            if un_gt:
                for t in un_gt:
                    st.write(f"- {t}")
            else:
                st.write("None ✅")
        with st.expander("Topic coverage details (by ground-truth title)", expanded=False):
            if cov_details:
                for d in sorted(cov_details, key=lambda x: x["score"], reverse=True):
                    status = "✅ covered" if d["covered"] else "❌ not covered"
                    if d["best_gen"]:
                        st.write(
                            f"- **GT:** {d['ref']}  \n"
                            f"  **Best gen:** {d['best_gen']}  \n"
                            f"  **Score:** {d['score']:.2f}  ({status})"
                        )
                    else:
                        st.write(
                            f"- **GT:** {d['ref']}  \n"
                            f"  **Best gen:** _none_  \n"
                            f"  **Score:** {d['score']:.2f}  ({status})"
                        )
            else:
                st.write("No ground-truth titles provided.")

    def run(self):
        st.markdown(
            """
            <style>
            .header-container {
                background: linear-gradient(135deg, #e0ecf8, #f5f9ff);
                padding: 30px 10px 20px 10px;
                border-radius: 12px;
                margin-bottom: 30px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            }
            .title {
                font-size: 38px;
                font-weight: 700;
                text-align: center;
                color: #153E75;
                font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                margin: 0;
            }
            </style>

            <div class="header-container">
                <div class="title">CA-RAG Pipeline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----- Sidebar settings -----
        with st.sidebar:
            st.header("⚙️ Settings")
            dlist = {
                "Health": "./modules/Dataset/HSE",
                "Education": "./modules/Dataset/UL",
                "Finance": "./modules/Dataset/FDIC",
            }

            model_select = st.selectbox("Model", list(models.keys()),index=0)
            model_path = models[model_select]
            if "model" not in st.session_state:
                st.session_state.model = model_path

            domain = st.selectbox("Domain", ["Health", "Education", "Finance"], index=0)
            ddlist = dlist[domain]
            st.write("Domain " + domain + " selected")

            st.subheader("Select variants")
            oflag = st.checkbox("Ontology")
            kgflag = st.checkbox("Knowledge Graph")

            load_model = st.button("Load Model")

        # ----- Model/chain init (triggered by button) -----
        if load_model:
            try:
                loader = DocumentLoader(ddlist)
                prompt_manager = PromptManager()
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", prompt_manager.system_prompt),
                        ("user", "{input}"),
                    ]
                )

                titles = loader.read_txt("./modules/Dataset/titles.txt")
                desc = loader.extract_desc("./modules/Dataset/descriptions.txt")
                docs = loader.load_documents()

                embedder = EmbeddingDB(self.embedding_path, device="cuda")
                embeddings, db = embedder.load(docs)
                retriever = db.as_retriever()
                if 'rl_model' in st.session_state:
                    st.session_state.model = st.session_state.rl_model
                print(st.session_state.model)
                llm_model = LanguageModel(st.session_state.model, device=self.device)
                llm = llm_model.load_model()

                question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
                rag_chain = create_retrieval_chain(db.as_retriever(), question_answer_chain)

                # Store everything needed later
                st.session_state.rag = {
                    "embedder": embedder,
                    "db": db,
                    "retriever": retriever,
                    "llm_model": llm_model,
                    "llm": llm,
                    "rag_chain": rag_chain,
                }
                st.session_state.prompt_manager = prompt_manager
                st.session_state.titles = titles
                st.session_state.desc = desc
                st.session_state.oflag = oflag
                st.session_state.kgflag = kgflag

                st.success("Model and RAG pipeline loaded successfully.")

            except Exception as e:
                st.error(f"Failed to load model/pipeline: {e}")

        # ----- Main input -----
        user_input = st.text_area(
            "Scenario:",
            value="Describing the BUs involved in data sharing along with all the constraints such as third party use, purpose of sharing etc.",
        )

        # ----- Generate agreement -----
        if st.button("Generate Agreement"):
            if "rag" not in st.session_state:
                st.error("Please click **Load Model** first (sidebar).")
            else:
                try:
                    dsa_gen = DSAGenerator(
                        st.session_state.titles,
                        st.session_state.desc,
                        st.session_state.rag['rag_chain'],
                        st.session_state.prompt_manager,
                        st.session_state.rag['llm'],
                        st.session_state.rag['retriever'],
                        st.session_state.oflag,
                        st.session_state.kgflag,
                    )
                    out, ext_out, prmpt_dict, docs, metrics = dsa_gen.generate(
                        {"user_input": user_input}, create=True
                    )
                    
                    st.session_state.generated_output = out
                    st.session_state.prmpt = prmpt_dict
                    st.session_state.docs = docs
                    st.session_state.ext_out = ext_out
                    st.session_state.metrics = metrics
                    if 'generated_titles' not in st.session_state and isinstance(out, dict):
                        st.session_state.generated_titles = list(out.keys())
                        print(list(out.keys()), st.session_state.titles)
                    if "generated_titles" in st.session_state and "titles" in st.session_state:
                        self.show_title_evaluation(generated_titles=st.session_state.generated_titles, ground_truth_titles=st.session_state.titles,
        default_threshold=0.80)

                    # ✅ Save metrics to disk (JSON + optional CSV)
                    base_name = "DSA_v2"  # or derive from scenario/domain if you want
                    metrics_json_path = self._save_metrics_json(metrics, base_name=base_name)
                    metrics_csv_path = self._save_metrics_csv(metrics, base_name=base_name)
                    
                    st.session_state.metrics_json_path = metrics_json_path
                    st.session_state.metrics_csv_path = metrics_csv_path
                    
                    st.success("Agreement generated (metrics saved).")

                except Exception as e:
                    st.error(f"Generation failed: {e}")

        # ----- Display output if available -----
        if "generated_output" in st.session_state and st.session_state.generated_output:
            st.markdown("### Generated Agreement")
            for key, value in st.session_state.generated_output.items():
                st.write(f"**{key}**")
                st.write(value)

            # ----- Save PDF section (not nested under generate button) -----
            st.markdown("### Export")
            file_out = st.text_input("Enter File Name:", value="DSA_v2")

            if st.button("Save as PDF"):
                try:
                    safe_name = self._safe_filename(file_out, default="DSA_v2")
                    pdf_path = os.path.join(self.output_dir, f"{safe_name}.pdf")

                    doc = SimpleDocTemplate(
                        pdf_path,
                        pagesize=A4,
                        rightMargin=2 * cm,
                        leftMargin=2 * cm,
                        topMargin=2 * cm,
                        bottomMargin=2 * cm,
                    )

                    flowables = self._dict_to_pdf_flowables(st.session_state.generated_output)
                    doc.build(flowables)

                    st.success(f"Saved successfully as {safe_name}.pdf")
                    # Optional: add download button
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="Download PDF",
                            data=f,
                            file_name=f"{safe_name}.pdf",
                            mime="application/pdf",
                        )
                except Exception as e:
                    st.error(f"PDF export failed: {e}")

    def flush(self):
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    # st.set_page_config(layout="wide")
    app = CA_RAG_App()
    app.run()
