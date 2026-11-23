import streamlit as st
import re
import json
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

# Optional PDF extraction
try:
    import PyPDF2
    _HAS_PDF = True
except Exception:
    _HAS_PDF = False

# NLTK
import nltk
from nltk.corpus import stopwords, wordnet
from nltk import word_tokenize, pos_tag
from nltk.chunk import RegexpParser
from nltk.stem import WordNetLemmatizer

# ML / NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    pipeline,
)
import torch


# ----------------------------
# NLTK bootstrap
# ----------------------------
@st.cache_data(ttl=24 * 3600)
def ensure_nltk():
    """
    Downloads NLTK resources once (cached).
    """
    resources = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    for pkg, res_path in resources.items():
        try:
            nltk.data.find(res_path)
        except LookupError:
            nltk.download(pkg)
    return True


# ----------------------------
# App config
# ----------------------------
st.set_page_config(
    page_title="DSA Policy Evaluation Dashboard",
    page_icon="📄",
    layout="wide"
)

st.title("📄 DSA Policy Evaluation Dashboard")
st.markdown(
    "Dataset-driven evaluation using **CUAD**, **LEDGAR**, **LexGLUE (unfair_tos)**, "
    "**domain PDFs**, plus **TF-IDF** and **BERT-based semantic checks**"
)
st.divider()

DATASET_SOURCES = {
    "CUAD_LOCAL": "./CUADv1.json",
}

SAMPLE_POLICY = """DATA SHARING AGREEMENT
...
(keep your sample policy here)
"""


# ----------------------------
# Text normalization + NLP helpers
# ----------------------------
_LEMM = WordNetLemmatizer()

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_for_match(s: str) -> str:
    """
    Normalization for matching:
    - lowercase
    - collapse whitespace
    - keep letters/numbers/hyphen/space
    """
    s = normalize_text(s).lower()
    s = re.sub(r"[^a-z0-9\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@st.cache_data(ttl=24 * 3600)
def get_stopwords():
    ensure_nltk()
    sw = set(stopwords.words("english"))
    # add frequent legal fillers
    sw |= {
        "shall", "will", "must", "may", "might", "could",
        "this", "that", "these", "those", "thereof", "hereby",
        "including", "include", "without", "within", "pursuant",
        "agreement", "party", "parties", "section", "clause",
        "data", "information"
    }
    return sw


@st.cache_data(ttl=3600)
def build_tfidf_keywords(
    label_to_texts,
    max_features_per_label: int = 10,
    max_docs_per_label: int = 200,
    min_df: int = 2
):
    """
    Build TF-IDF-based keywords for each label using a bag-of-words representation.

    label_to_texts: dict[label -> list[str]]
    Returns: dict[label -> list[str]] of top TF-IDF terms per label.
    """
    docs = []
    labels = []

    for label, texts in label_to_texts.items():
        for t in texts[:max_docs_per_label]:
            t = normalize_text(t)
            if t:
                docs.append(t)
                labels.append(label)

    if not docs:
        return {}

    sw = list(get_stopwords())
    vectorizer = TfidfVectorizer(
        preprocessor=normalize_for_match,
        token_pattern=r"[a-z][a-z\-]{2,}",
        stop_words=sw,
        min_df=min_df
    )

    X = vectorizer.fit_transform(docs)
    feature_names = np.array(vectorizer.get_feature_names_out())

    tfidf_keywords = {}
    labels_unique = sorted(set(labels), key=lambda x: str(x).lower())

    for label in labels_unique:
        idxs = [i for i, lab in enumerate(labels) if lab == label]
        if not idxs:
            continue

        label_matrix = X[idxs]
        mean_scores = np.asarray(label_matrix.mean(axis=0)).ravel()
        if not np.any(mean_scores > 0):
            continue

        top_idx = mean_scores.argsort()[::-1][:max_features_per_label]
        top_terms = feature_names[top_idx]
        tfidf_keywords[label] = [t for t in top_terms if t]

    return tfidf_keywords


def _wn_pos(treebank_tag: str):
    """
    Map Penn POS tag to WordNet POS.
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


@st.cache_data(ttl=3600)
def lemmatize_text(text: str):
    """
    Returns:
      - lemmas: list of lemmatized tokens
      - lemma_set: set of lemmas (fast membership)
      - lemma_counts: Counter of lemmas
      - normalized: normalized string for phrase matching
    Cached because it's used repeatedly.
    """
    ensure_nltk()
    sw = get_stopwords()

    normalized = normalize_for_match(text)
    toks = word_tokenize(normalized)

    toks = [t for t in toks if re.fullmatch(r"[a-z][a-z\-]{2,}", t)]
    if not toks:
        return [], set(), Counter(), normalized

    tagged = pos_tag(toks)
    lemmas = []
    for w, tag in tagged:
        if w in sw:
            continue
        lem = _LEMM.lemmatize(w, pos=_wn_pos(tag))
        if lem and len(lem) >= 3 and lem not in sw:
            lemmas.append(lem)

    counts = Counter(lemmas)
    return lemmas, set(lemmas), counts, normalized


def phrase_present(normalized_text: str, phrase: str) -> bool:
    """
    Word-boundary phrase matching on already-normalized text.
    """
    ph = normalize_for_match(phrase)
    if not ph:
        return False
    pattern = r"(?<!\w)" + re.escape(ph) + r"(?!\w)"
    return re.search(pattern, normalized_text) is not None


def extract_text_from_pdf(path: Path) -> str:
    if not _HAS_PDF:
        return ""
    try:
        reader = PyPDF2.PdfReader(str(path))
        chunks = []
        for p in reader.pages[:20]:
            chunks.append(p.extract_text() or "")
        return normalize_text(" ".join(chunks))
    except Exception:
        return ""


def extract_text_from_uploaded_pdf(uploaded_file) -> str:
    """
    Extract text from an uploaded PDF (Streamlit UploadedFile).
    """
    if not _HAS_PDF or uploaded_file is None:
        return ""

    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        chunks = []
        for p in reader.pages[:20]:
            chunks.append(p.extract_text() or "")
        return normalize_text(" ".join(chunks))
    except Exception:
        return ""


# ----------------------------
# Domain-specific BERT encoders
# ----------------------------
DOMAIN_BERT_MODELS = {
    "Health": "emilyalsentzer/Bio_ClinicalBERT",
    "Education": "vasugoel/K-12BERT",#"nlpaueb/legal-bert-base-uncased",
    "Finance": "ProsusAI/finbert",
}

@st.cache_resource
def load_domain_bert(domain: str):
    """
    Load a BERT-like model & tokenizer based on the selected domain.
    """
    model_name = DOMAIN_BERT_MODELS.get(domain, "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _bert_mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * input_mask_expanded, 1)
    summed_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return summed / summed_mask


def encode_text_bert(text: str, tokenizer, model, max_length: int = 512) -> torch.Tensor:
    """
    Encode a single text into a BERT embedding (1D tensor).
    """
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )
        outputs = model(**inputs)
        embedding = _bert_mean_pooling(outputs, inputs["attention_mask"])
        return embedding[0]


# ----------------------------
# CUAD: load local JSON (TF-IDF keywords)
# ----------------------------
@st.cache_data(ttl=3600)
def load_and_process_cuad(local_json_path: str = DATASET_SOURCES["CUAD_LOCAL"]):
    """
    Load CUAD-like dataset and extract clause keywords using TF-IDF.
    """

    def _extract_from_record_list(records):
        clause_examples = defaultdict(list)
        clause_questions = {}

        for item in records:
            for paragraph in item.get("paragraphs", []):
                for qa in paragraph.get("qas", []):
                    question = normalize_text(qa.get("question", ""))
                    answers = qa.get("answers") or []

                    clause_type = (
                        question.replace(
                            'Highlight the parts (if any) of this contract related to "', ""
                        )
                        .replace('" that should be reviewed by a lawyer. Details: ', "")
                        .strip('."')
                    ).strip()

                    if clause_type:
                        clause_questions[clause_type] = question
                        for ans in answers:
                            t = normalize_text(ans.get("text", ""))
                            if t and len(t) > 10:
                                clause_examples[clause_type].append(t)

        return clause_examples, clause_questions

    try:
        path = Path(local_json_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict) and "data" in data:
                records = data["data"]
            elif isinstance(data, list):
                records = data
            else:
                raise ValueError(
                    f"Unsupported CUAD JSON structure at {path.resolve()}"
                )

            clause_examples, clause_questions = _extract_from_record_list(records)

        else:
            st.info(
                f"Local CUAD file not found at {path.resolve()}, "
                "trying HuggingFace 'cuad' dataset..."
            )
            ds = load_dataset("cuad", "plain_text", split="train")
            clause_examples = defaultdict(list)
            clause_questions = {}

            for ex in ds:
                question = normalize_text(ex.get("question", ""))
                answers = ex.get("answers", {}).get("text", []) or []

                clause_type = (
                    question.replace(
                        'Highlight the parts (if any) of this contract related to "', ""
                    )
                    .replace('" that should be reviewed by a lawyer. Details: ', "")
                    .strip('."')
                ).strip()

                if clause_type:
                    clause_questions[clause_type] = question
                    for t in answers:
                        t = normalize_text(t)
                        if t and len(t) > 10:
                            clause_examples[clause_type].append(t)

        # TF-IDF-based clause keywords
        clause_keywords = build_tfidf_keywords(
            clause_examples,
            max_features_per_label=8,
            max_docs_per_label=200,
            min_df=2
        )

        return clause_keywords, clause_questions, len(clause_examples)

    except Exception as e:
        st.warning(f"Could not load CUAD dataset (local or HF): {e}")
        return None, None, 0


# ----------------------------
# LEDGAR via HuggingFace datasets (TF-IDF)
# ----------------------------
@st.cache_data(ttl=3600)
def load_and_process_ledgar(split: str = "train", max_samples: int = 2000):
    """
    Learns provision keywords using a TF-IDF bag-of-words approach.
    """
    try:
        ds = load_dataset("lex_glue", "ledgar", split=split)

        label_feature = ds.features.get("label", None)
        label_names = (
            label_feature.names
            if (label_feature is not None and hasattr(label_feature, "names"))
            else None
        )

        provision_texts = defaultdict(list)
        provision_labels = set()

        n = min(len(ds), max_samples)
        for item in ds.select(range(n)):
            text = normalize_text(item.get("text", ""))
            label_id = item.get("label", None)

            if text and label_id is not None:
                if label_names and isinstance(label_id, int) and label_id < len(label_names):
                    label = label_names[label_id]
                else:
                    label = str(label_id)

                provision_labels.add(label)
                provision_texts[label].append(text)

        provision_keywords = build_tfidf_keywords(
            provision_texts,
            max_features_per_label=10,
            max_docs_per_label=200,
            min_df=2
        )

        return provision_keywords, sorted(list(provision_labels), key=lambda x: x.lower()), len(provision_texts)

    except Exception as e:
        st.warning(f"Could not load LEDGAR dataset: {e}")
        return None, None, 0


# ----------------------------
# LexGLUE unfair_tos (TF-IDF keywords)
# ----------------------------
@st.cache_data(ttl=3600)
def load_and_process_lexglue_unfair_tos(
    split: str = "train",
    max_samples: int = 6000,
    min_per_label: int = 25
):
    """
    Learns unfair_tos keywords using TF-IDF.
    """
    try:
        ds = load_dataset("lex_glue", "unfair_tos", split=split)

        labels_feature = ds.features.get("labels", None)
        label_names = None
        try:
            if labels_feature is not None and hasattr(labels_feature, "feature") and hasattr(labels_feature.feature, "names"):
                label_names = labels_feature.feature.names
        except Exception:
            label_names = None

        per_label_texts = defaultdict(list)

        n = min(len(ds), max_samples)
        for item in ds.select(range(n)):
            text = normalize_text(item.get("text", ""))
            labs = item.get("labels", None)
            if not text or labs is None:
                continue

            label_ids = []
            if isinstance(labs, list) and len(labs) > 0:
                if all(isinstance(x, int) for x in labs) and (max(labs) > 1 or len(set(labs)) > 2):
                    majority = Counter(labs).most_common(1)[0][0]
                    label_ids = [majority]
                elif all(x in [0, 1] for x in labs) and len(labs) > 2:
                    label_ids = [i for i, v in enumerate(labs) if v == 1]
                else:
                    label_ids = [labs[0]]

            for lid in label_ids:
                if label_names and isinstance(lid, int) and lid < len(label_names):
                    lname = label_names[lid]
                else:
                    lname = f"label_{lid}"
                per_label_texts[lname].append(text)

        usable = {k: v for k, v in per_label_texts.items() if len(v) >= min_per_label}
        if not usable:
            return None, None, 0, "Not enough samples to learn patterns from LexGLUE/unfair_tos (0 usable)."

        lex_keywords = build_tfidf_keywords(
            usable,
            max_features_per_label=12,
            max_docs_per_label=400,
            min_df=3
        )

        return lex_keywords, sorted(list(usable.keys()), key=lambda x: x.lower()), len(usable), None

    except Exception as e:
        return None, None, 0, f"Could not load LexGLUE unfair_tos: {e}"


# ----------------------------
# Domain requirements from PDFs (TF-IDF + noun phrases)
# ----------------------------
def extract_noun_phrases(text: str, max_phrases: int = 40):
    """
    NLTK noun-phrase extraction for domain requirement phrases.
    """
    ensure_nltk()
    sw = get_stopwords()

    normalized = normalize_for_match(text)
    toks = word_tokenize(normalized)
    toks = [t for t in toks if re.fullmatch(r"[a-z][a-z\-]{2,}", t)]
    toks = [t for t in toks if t not in sw]

    if not toks:
        return []

    tagged = pos_tag(toks)

    grammar = r"""
      NP: {<JJ.*>*<NN.*>+}
    """
    chunker = RegexpParser(grammar)
    tree = chunker.parse(tagged)

    phrases = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
        phrase = " ".join([w for w, _ in subtree.leaves()])
        if 5 <= len(phrase) <= 80:
            phrases.append(phrase)

    counts = Counter(phrases)
    return [p for p, _ in counts.most_common(max_phrases)]


@st.cache_data(ttl=3600)
def load_and_process_domain_pdfs(domain_root: str, max_pdfs: int = 30):
    """
    Learns domain keywords from PDFs using TF-IDF + noun phrases.
    """
    root = Path(domain_root)
    if not root.exists():
        return None, None, 0, f"Folder not found: {root.resolve()}"

    pdfs = sorted(list(root.rglob("*.pdf")))[:max_pdfs]
    if not pdfs:
        return None, None, 0, f"No PDFs found in: {root.resolve()}"

    texts = []
    for p in pdfs:
        t = extract_text_from_pdf(p)
        if t and len(t) > 200:
            texts.append(t)

    if not texts:
        return None, None, 0, "Could not extract text from PDFs (install PyPDF2 and ensure PDFs are text-based)."

    all_text = " ".join(texts)

    # Domain keywords via TF-IDF (all docs as one label)
    label_to_texts = {"domain": texts}
    tfidf_kw = build_tfidf_keywords(
        label_to_texts,
        max_features_per_label=35,
        max_docs_per_label=300,
        min_df=3
    )
    keywords = tfidf_kw.get("domain", [])

    _, _, _, normalized = lemmatize_text(all_text)
    phrases = extract_noun_phrases(normalized, max_phrases=50)

    return keywords, phrases, len(texts), None


@st.cache_data(ttl=3600)
def get_domain_corpus_text(domain_root: str, max_pdfs: int = 30) -> str:
    """
    Build a single concatenated text corpus for the given domain PDFs.
    Used for BERT semantic similarity.
    """
    root = Path(domain_root)
    if not root.exists():
        return ""

    pdfs = sorted(list(root.rglob("*.pdf")))[:max_pdfs]
    if not pdfs:
        return ""

    chunks = []
    for p in pdfs:
        t = extract_text_from_pdf(p)
        if t and len(t) > 200:
            chunks.append(t)

    return " ".join(chunks)


def compute_domain_bert_score(policy_text: str, domain: str, domain_root: str) -> int:
    """
    Use a domain-specific BERT to compute a semantic Domain Alignment score.
    0–100 based on cosine similarity between policy and domain corpus.
    """
    corpus_text = get_domain_corpus_text(domain_root)
    if not corpus_text or not policy_text.strip():
        return 0

    tokenizer, model = load_domain_bert(domain)

    policy_emb = encode_text_bert(policy_text, tokenizer, model)
    corpus_emb = encode_text_bert(corpus_text, tokenizer, model)

    policy_norm = policy_emb / policy_emb.norm(p=2)
    corpus_norm = corpus_emb / corpus_emb.norm(p=2)
    cos_sim = torch.dot(policy_norm, corpus_norm).item()  # -1..1

    score = int(max(0.0, min(1.0, (cos_sim + 1.0) / 2.0)) * 100)
    return score


# ----------------------------
# LegalBERT unfair_tos semantic model
# ----------------------------
@st.cache_resource
def load_unfair_tos_semantic_model(model_name: str = "Agreemind/legalbert-unfair-tos"):
    """
    Load a LegalBERT-based classifier fine-tuned on UNFAIR-ToS.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=512
    )

    id2label = model.config.id2label
    unfair_labels = {
        lbl for lbl in id2label.values()
        if lbl.lower() not in {"none", "other"}
    }
    return clf, unfair_labels


def evaluate_semantic_unfair_terms(
    text: str,
    clf,
    unfair_labels,
    min_score: float = 0.6,
    max_sentences: int = 400,
):
    """
    Use a LegalBERT-based classifier to detect unfair ToS clauses at sentence level.
    Returns:
      - hits: list of {sentence, label, score}
      - semantic_score: 0–100, where more unfair hits => lower score
    """
    ensure_nltk()
    from nltk.tokenize import sent_tokenize

    sentences = [s.strip() for s in sent_tokenize(text) if len(s.split()) >= 5]
    sentences = sentences[:max_sentences]

    if not sentences:
        return [], 100

    preds = clf(sentences, batch_size=8, truncation=True, max_length=512)

    hits = []
    for sent, pred in zip(sentences, preds):
        label = pred["label"]
        score = float(pred["score"])
        if label in unfair_labels and score >= min_score:
            hits.append(
                {
                    "sentence": sent,
                    "label": label,
                    "score": score,
                }
            )

    penalty = min(60, 5 * len(hits))
    semantic_score = max(0, 100 - penalty)
    return hits, semantic_score


# ----------------------------
# Evaluators
# ----------------------------
def evaluate_with_cuad(text, clause_keywords):
    if not clause_keywords:
        return [], 0

    _, lemma_set, _, _ = lemmatize_text(text)

    results = []
    for clause_type, keywords in clause_keywords.items():
        found = [k for k in keywords if k in lemma_set]
        results.append({
            "name": clause_type,
            "present": len(found) > 0,
            "matches": found,
            "keywords": keywords
        })

    present_count = sum(1 for r in results if r["present"])
    score = (present_count / len(results)) * 100 if results else 0
    return results, score


def evaluate_with_ledgar(text, provision_keywords):
    if not provision_keywords:
        return [], 0

    _, lemma_set, _, _ = lemmatize_text(text)

    results = []
    for label, keywords in provision_keywords.items():
        matches = [k for k in keywords if k in lemma_set]
        results.append({
            "name": label,
            "present": len(matches) > 0,
            "match_count": len(matches),
            "matches": matches[:8],
            "keywords": keywords
        })

    present_count = sum(1 for r in results if r["present"])
    score = (present_count / len(results)) * 100 if results else 0
    return results, score


def evaluate_lexglue_consistency(text, lex_keywords):
    """
    Keyword-based unfairness via TF-IDF lemma patterns.
    """
    if not lex_keywords:
        return [], 0, "LexGLUE patterns unavailable (dataset not loaded or insufficient usable samples)."

    _, lemma_set, _, _ = lemmatize_text(text)

    hits = []
    unique_hit_terms = set()

    for lname, kws in lex_keywords.items():
        found = [k for k in kws if k in lemma_set]
        if found:
            unique_hit_terms |= set(found)
            hits.append({
                "name": lname,
                "hits": len(found),
                "examples": found[:8],
            })

    total_unique = len(unique_hit_terms)
    score = max(0, 100 - (total_unique * 6))
    return hits, score, None


def evaluate_domain_requirements(text, domain_keywords, domain_phrases):
    """
    Domain coverage using TF-IDF keywords & noun phrases.
    """
    if not domain_keywords and not domain_phrases:
        return [], 0, "Domain requirements unavailable (no usable PDFs / text extraction failed)."

    _, lemma_set, _, normalized = lemmatize_text(text)

    checks = []

    found_kw = [k for k in (domain_keywords or []) if k in lemma_set]
    kw_score = min(100, int(100 * (len(found_kw) / max(1, len(domain_keywords or [])))))

    found_ph = [p for p in (domain_phrases or []) if phrase_present(normalized, p)]
    ph_score = min(100, int(100 * (len(found_ph) / max(1, len(domain_phrases or [])))))

    score = int((kw_score * 0.4) + (ph_score * 0.6))

    checks.append({
        "name": "Domain keyword coverage",
        "found": len(found_kw),
        "total": len(domain_keywords or []),
        "examples": found_kw[:10],
        "missing_examples": [k for k in (domain_keywords or []) if k not in lemma_set][:10]
    })

    checks.append({
        "name": "Domain requirement phrases",
        "found": len(found_ph),
        "total": len(domain_phrases or []),
        "examples": found_ph[:10],
        "missing_examples": [p for p in (domain_phrases or []) if not phrase_present(normalized, p)][:8]
    })

    return checks, score, None


def get_score_color(score):
    if score >= 80:
        return "green"
    elif score >= 60:
        return "orange"
    return "red"


# ----------------------------
# Sidebar: domain + dataset status
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    domain = st.selectbox("Domain", ["Health", "Education", "Finance"], index=0)

    st.subheader("📁 Domain PDF folders")
    st.caption("Point these to local folders containing domain documents (PDFs).")
    health_dir = st.text_input("Health folder", value="./modules/Dataset/HSE")
    edu_dir = st.text_input("Education folder", value="./modules/Dataset/UL")
    fin_dir = st.text_input("Finance folder", value="./modules/Dataset/FDIC")

    st.divider()
    st.subheader("🔄 Dataset Status")

    with st.spinner("Loading datasets..."):
        cuad_keywords, cuad_questions, cuad_count = load_and_process_cuad()
        ledgar_keywords, ledgar_labels, ledgar_count = load_and_process_ledgar()
        lex_keywords, lex_labels, lex_count, lex_err = load_and_process_lexglue_unfair_tos()

        domain_root = {"Health": health_dir, "Education": edu_dir, "Finance": fin_dir}[domain]
        domain_keywords, domain_phrases, domain_docs_n, domain_err = load_and_process_domain_pdfs(domain_root)

    st.markdown(f"- {'✅' if cuad_keywords else '⚠️'} CUAD (local/ HF): {cuad_count} clause types")
    st.markdown(f"- {'✅' if ledgar_keywords else '⚠️'} LEDGAR: {ledgar_count} provisions")
    st.markdown(f"- {'✅' if lex_keywords else '⚠️'} LexGLUE unfair_tos: {lex_count} label groups")
    st.markdown(f"- {'✅' if domain_keywords or domain_phrases else '⚠️'} Domain PDFs ({domain}): {domain_docs_n} docs")

    if lex_err:
        st.warning(lex_err)
    if domain_err:
        st.warning(domain_err)

    st.divider()
    if st.button("📝 Load Sample Policy", use_container_width=True):
        st.session_state["policy_text"] = SAMPLE_POLICY


# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["📝 Input Policy", "📊 Evaluation Results", "🔬 Learned Patterns"])

with tab1:
    st.subheader("Enter Policy Text")

    # Initialize keys
    st.session_state.setdefault("policy_input", "")
    st.session_state.setdefault("policy_text", "")
    st.session_state.setdefault("pdf_text_pending", "")

    policy_text = st.text_area(
        "Paste your Data Sharing Agreement / policy text here:",
        height=420,
        key="policy_input"
    )

    # Keep canonical policy_text in sync with the editor
    st.session_state["policy_text"] = policy_text

    st.markdown("Or upload a **PDF** to extract policy text:")

    uploaded_pdf = st.file_uploader(
        "Upload policy PDF",
        type=["pdf"],
        key="policy_pdf_uploader"
    )

    if uploaded_pdf is not None:
        pdf_text = extract_text_from_uploaded_pdf(uploaded_pdf)
        if not pdf_text.strip():
            st.error(
                "Could not extract text from the uploaded PDF. "
                "Make sure it contains selectable text (not just scanned images)."
            )
        else:
            # Store extracted text somewhere safe (NOT the widget key)
            st.session_state["pdf_text_pending"] = pdf_text
            st.success("PDF text extracted. Click 'Load into editor' to replace the text above.")

            def load_pdf_into_editor():
                # This is safe because it runs as part of the button click handling
                st.session_state["policy_input"] = st.session_state.get("pdf_text_pending", "")
                st.session_state["policy_text"] = st.session_state["policy_input"]

            st.button("⬇️ Load into editor", on_click=load_pdf_into_editor)

            def append_pdf_to_editor():
                existing = st.session_state.get("policy_input", "")
                incoming = st.session_state.get("pdf_text_pending", "")
                sep = "\n\n" if existing.strip() and incoming.strip() else ""
                st.session_state["policy_input"] = f"{existing}{sep}{incoming}"
                st.session_state["policy_text"] = st.session_state["policy_input"]

            st.button("➕ Append to editor", on_click=append_pdf_to_editor)

    word_count = len(st.session_state["policy_input"].split())
    st.caption(
        f"📊 Word count: {word_count} | Characters: {len(st.session_state['policy_input'])}"
    )

    colA, colB = st.columns([3, 1])
    with colA:
        evaluate_button = st.button("🔍 Evaluate Policy", type="primary", use_container_width=True)
    with colB:
        def clear_policy():
            st.session_state["policy_input"] = ""
            st.session_state["policy_text"] = ""
            st.session_state["pdf_text_pending"] = ""

        st.button("🗑️ Clear", on_click=clear_policy, use_container_width=True)

    if evaluate_button:
        if not policy_text.strip():
            st.error("⚠️ Please enter policy text to evaluate")
        else:
            with st.spinner("Running evaluation..."):
                # CUAD, LEDGAR coverage
                cuad_results, cuad_score = evaluate_with_cuad(policy_text, cuad_keywords)
                ledgar_results, ledgar_score = evaluate_with_ledgar(policy_text, ledgar_keywords)

                # Keyword-based unfairness (LexGLUE TF-IDF)
                lex_hits, lex_keyword_score, lex_msg = evaluate_lexglue_consistency(policy_text, lex_keywords)

                # Semantic unfairness (LegalBERT)
                clf_unfair, unfair_labels = load_unfair_tos_semantic_model()
                semantic_hits, semantic_score = evaluate_semantic_unfair_terms(
                    policy_text,
                    clf_unfair,
                    unfair_labels
                )

                # Combined LexGLUE fairness score
                lex_score = int(0.5 * lex_keyword_score + 0.5 * semantic_score)

                # Domain coverage via TF-IDF + phrases
                dom_checks, dom_tfidf_score, dom_msg = evaluate_domain_requirements(
                    policy_text,
                    domain_keywords,
                    domain_phrases
                )

                # Domain semantic score via BERT
                domain_root = {"Health": health_dir, "Education": edu_dir, "Finance": fin_dir}[domain]
                dom_bert_score = compute_domain_bert_score(policy_text, domain, domain_root)

                combined_domain_score = int(0.5 * dom_tfidf_score + 0.5 * dom_bert_score)

                overall_score = int(
                    (cuad_score * 0.30) +
                    (ledgar_score * 0.25) +
                    (lex_score * 0.20) +
                    (combined_domain_score * 0.25)
                )

                st.session_state["evaluation_results"] = {
                    "scores": {
                        "overall": overall_score,
                        "cuad": cuad_score,
                        "ledgar": ledgar_score,
                        "lexglue": lex_score,
                        "domain": combined_domain_score
                    },
                    "domain": domain,
                    "cuad_results": cuad_results,
                    "ledgar_results": ledgar_results,
                    "lexglue_hits": lex_hits,
                    "lexglue_message": lex_msg,
                    "lexglue_keyword_score": lex_keyword_score,
                    "lexglue_semantic_score": semantic_score,
                    "lexglue_semantic_hits": semantic_hits,
                    "domain_checks": dom_checks,
                    "domain_message": dom_msg,
                    "domain_tfidf_score": dom_tfidf_score,
                    "domain_bert_score": dom_bert_score,
                    "word_count": word_count,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                st.success("✅ Evaluation complete! Open the **Evaluation Results** tab.")


with tab2:
    if "evaluation_results" not in st.session_state:
        st.info("👈 Evaluate a policy first in **Input Policy**.")
    else:
        res = st.session_state["evaluation_results"]
        scores = res["scores"]

        st.subheader("📈 Overall Score")
        color = get_score_color(scores["overall"])
        st.markdown(
            f"<h1 style='text-align:center;color:{color};'>{scores['overall']}/100</h1>",
            unsafe_allow_html=True
        )
        st.caption(f"Domain: **{res['domain']}** | Evaluated: {res['timestamp']}")

        st.divider()
        st.subheader("📊 Score Breakdown")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clause Compliance (CUAD)", f"{scores['cuad']:.0f}%")
        c2.metric("Provision Coverage (LEDGAR)", f"{scores['ledgar']:.0f}%")
        c3.metric("Fairness (LexGLUE + LegalBERT)", f"{scores['lexglue']:.0f}%")
        c4.metric("Domain Alignment", f"{scores['domain']:.0f}%")

        st.divider()
        st.subheader("🚨 Violations (Only)")

        # CUAD violations
        if res["cuad_results"]:
            missing_cuad = [c for c in res["cuad_results"] if not c["present"]]
            st.markdown("### 📚 CUAD missing clause signals")
            if not missing_cuad:
                st.success("No CUAD-based missing clause signals detected.")
            else:
                for c in missing_cuad[:40]:
                    with st.expander(f"❌ {c['name']}"):
                        st.write(f"Learned TF-IDF keywords (lemmas): {', '.join(c['keywords'])}")
                        st.write("Recommendation: add/clarify language addressing this clause type.")

        # LEDGAR violations
        if res["ledgar_results"]:
            missing_ledgar = [p for p in res["ledgar_results"] if not p["present"]]
            st.markdown("### ⚖️ LEDGAR missing provision signals")
            if not missing_ledgar:
                st.success("No LEDGAR-based missing provision signals detected.")
            else:
                for p in missing_ledgar[:60]:
                    with st.expander(f"❌ {p['name']}"):
                        st.write(f"Learned TF-IDF keywords (lemmas): {', '.join(p['keywords'])}")
                        st.write("Recommendation: consider adding a provision that covers this legal concept.")

        st.divider()

        st.subheader("⚠️ Potential Unfair / One-sided Language (LexGLUE + LegalBERT)")
        if res.get("lexglue_message"):
            st.warning(res["lexglue_message"])

        st.caption(
            f"LexGLUE keyword unfairness score: {res.get('lexglue_keyword_score', 0):.0f}/100 | "
            f"LegalBERT semantic unfairness score: {res.get('lexglue_semantic_score', 0):.0f}/100"
        )

        # Keyword-based hits
        if res.get("lexglue_hits"):
            st.markdown("#### 🔎 Keyword-based unfairness patterns (TF-IDF)")
            for h in sorted(res["lexglue_hits"], key=lambda x: x["hits"], reverse=True):
                with st.expander(f"🚩 {h['name']} ({h['hits']} lemma hits)"):
                    st.write("Examples matched (lemmas):")
                    for ex in h["examples"]:
                        st.write(f"• {ex}")
                    st.write("Recommendation: rewrite to reduce unilateral rights / add notice, limits, and reciprocity.")
        else:
            st.success("No LexGLUE keyword-based unfair_tos hits detected.")

        # Semantic hits
        if res.get("lexglue_semantic_hits"):
            st.markdown("#### 🤖 Semantic unfairness flags (LegalBERT)")
            for i, h in enumerate(res["lexglue_semantic_hits"][:40], start=1):
                label = h["label"]
                score = h["score"]
                sent = h["sentence"]
                with st.expander(f"🚩 [{i}] {label} (confidence: {score:.2f})"):
                    st.write(sent)
                    st.write(
                        "Recommendation: review this sentence for potential one-sided or unfair effects; "
                        "add notice, limits, and reciprocal obligations as appropriate."
                    )
        else:
            st.success("No high-confidence semantic unfair_tos flags detected by LegalBERT.")

        st.divider()

        st.subheader(f"🏷️ Domain Requirements Coverage ({res['domain']})")
        if res.get("domain_message"):
            st.warning(res["domain_message"])

        st.caption(
            f"Domain TF-IDF coverage score: {res.get('domain_tfidf_score', 0):.0f}/100 | "
            f"Domain BERT semantic score: {res.get('domain_bert_score', 0):.0f}/100"
        )

        if res.get("domain_checks"):
            for chk in res["domain_checks"]:
                with st.expander(f"{chk['name']} — {chk['found']}/{chk['total']} found"):
                    if chk["examples"]:
                        st.write("Found examples:")
                        for x in chk["examples"]:
                            st.write(f"• {x}")
                    if chk["missing_examples"]:
                        st.write("Missing examples (top):")
                        for x in chk["missing_examples"]:
                            st.write(f"• {x}")

        st.divider()
        st.subheader("💡 Recommendations")

        recs = []

        if res["cuad_results"]:
            missing_cuad = [c for c in res["cuad_results"] if not c["present"]]
            for c in missing_cuad[:8]:
                kws = ", ".join(c["keywords"][:5])
                recs.append(f"Add/clarify **{c['name']}** (signals: {kws})")

        if res["ledgar_results"]:
            missing_ledgar = [p for p in res["ledgar_results"] if not p["present"]]
            for p in missing_ledgar[:8]:
                kws = ", ".join(p["keywords"][:5])
                recs.append(f"Add a provision aligned with **{p['name']}** (signals: {kws})")

        if res.get("lexglue_hits"):
            top_hit = sorted(res["lexglue_hits"], key=lambda x: x["hits"], reverse=True)[:3]
            for h in top_hit:
                recs.append(
                    f"Reduce one-sided patterns flagged as **{h['name']}** "
                    f"(rewrite language; add notice, limits, and reciprocity)."
                )

        if res.get("lexglue_semantic_hits"):
            for h in res["lexglue_semantic_hits"][:3]:
                recs.append(
                    f"Review sentence flagged as **{h['label']}** "
                    f"(semantic unfairness, confidence {h['score']:.2f})."
                )

        if res.get("domain_checks"):
            phrase_chk = next((x for x in res["domain_checks"] if "phrases" in x["name"].lower()), None)
            if phrase_chk and phrase_chk.get("missing_examples"):
                recs.append(
                    "Add domain-specific obligations referencing: "
                    f"**{', '.join(phrase_chk['missing_examples'][:4])}**"
                )

        if not recs:
            st.success("No major gaps detected by the current learned-pattern checks.")
        else:
            for r in recs[:18]:
                st.write(f"• {r}")

        st.divider()
        st.subheader("⬇️ Download Report")

        report_json = json.dumps(res, indent=2)

        st.download_button(
            label="Download JSON report",
            data=report_json,
            file_name=f"dsa_eval_{res['domain'].lower()}_{res['timestamp'].replace(' ', '_').replace(':','-')}.json",
            mime="application/json",
            use_container_width=True
        )

        rows = [
            {"metric": "overall", "value": scores["overall"]},
            {"metric": "cuad", "value": round(scores["cuad"], 2)},
            {"metric": "ledgar", "value": round(scores["ledgar"], 2)},
            {"metric": "lexglue", "value": round(scores["lexglue"], 2)},
            {"metric": "domain", "value": round(scores["domain"], 2)},
        ]
        df = pd.DataFrame(rows)
        st.download_button(
            label="Download CSV summary",
            data=df.to_csv(index=False),
            file_name=f"dsa_eval_summary_{res['domain'].lower()}_{res['timestamp'].replace(' ', '_').replace(':','-')}.csv",
            mime="text/csv",
            use_container_width=True
        )


with tab3:
    st.subheader("🔬 Learned Patterns")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 📚 CUAD clause keywords (TF-IDF, lemmatized)")
        if cuad_keywords:
            rows = []
            for k, v in sorted(cuad_keywords.items(), key=lambda x: x[0].lower()):
                rows.append({"Clause Type": k, "Keywords": ", ".join(v)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=350)
        else:
            st.warning("CUAD keywords not available.")

        st.markdown("### ⚖️ LEDGAR provision keywords (TF-IDF, lemmatized)")
        if ledgar_keywords:
            rows = []
            for k, v in sorted(ledgar_keywords.items(), key=lambda x: str(x[0]).lower()):
                rows.append({"Provision", ", ".join(v)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=350)
        else:
            st.warning("LEDGAR keywords not available.")

    with c2:
        st.markdown("### ⚠️ LexGLUE unfair_tos keywords (TF-IDF, lemmatized)")
        if lex_keywords:
            rows = []
            for k, v in sorted(lex_keywords.items(), key=lambda x: x[0].lower()):
                rows.append({"UnfairTOS Category": k, "Keywords": ", ".join(v[:12])})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=350)
        else:
            st.warning("LexGLUE unfair_tos patterns not available.")
            if lex_err:
                st.info(lex_err)

        st.markdown(f"### 🏷️ Domain requirements learned from PDFs ({domain})")
        if domain_keywords or domain_phrases:
            st.write("**Top learned domain keywords (TF-IDF lemmas):**")
            st.write(", ".join((domain_keywords or [])[:35]))
            st.write("")
            st.write("**Top learned requirement phrases (noun phrases):**")
            for p in (domain_phrases or [])[:25]:
                st.write(f"• {p}")
        else:
            st.warning("Domain requirements not available. Check the folder path and PDF text extraction.")
            if domain_err:
                st.info(domain_err)
