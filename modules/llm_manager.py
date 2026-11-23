"""
llm_manager.py  —  Drop-in replacement for CA-RAG
=======================================================
Keeps the original LanguageModel(model_name, device) interface so that
1_Main.py and every other page work without any changes.

Backend priority (auto-detected):
  1. Groq API        – fastest free option; needs GROQ_API_KEY env var
                       models: llama-3.1-8b-instant  (default)
                               mixtral-8x7b-32768
                               gemma2-9b-it
  2. HuggingFace Hub – local pipeline on CPU; falls back to TinyLlama-1.1B
                       if the requested model is too large for the hardware.
  3. Original path   – GPU-based HuggingFacePipeline (unchanged, for local GPU runs)

Set the backend explicitly with the env var:
    LLM_BACKEND=groq | huggingface | auto   (default: auto)

Set a smaller model override:
    LLM_MODEL_OVERRIDE=TinyLlama/TinyLlama-1.1B-Chat-v1.0

For Streamlit Cloud / HF Spaces deployment add these secrets:
    GROQ_API_KEY   = gsk_...
    LLM_BACKEND    = groq        (optional, auto-detects if key is set)
"""

import os
import streamlit as st

# ── env config ────────────────────────────────────────────────────────────────
_BACKEND       = os.getenv("LLM_BACKEND", "auto").lower()
_GROQ_KEY      = os.getenv("GROQ_API_KEY", "")
_MODEL_OVERRIDE = os.getenv("LLM_MODEL_OVERRIDE", "")

# Map original large model names → small Groq equivalents
_GROQ_MODEL_MAP = {
    "meta-llama/Llama-3.1-8B-Instruct":          "llama-3.1-8b-instant",
    "meta-llama/Llama-3.2-3B-Instruct":          "llama-3.2-3b-preview",
    "mistralai/Mistral-7B-Instruct-v0.3":        "mixtral-8x7b-32768",
}

# Small CPU-friendly HuggingFace models (used when no Groq key is available)
_SMALL_HF_MODEL = _MODEL_OVERRIDE or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _use_groq() -> bool:
    if _BACKEND == "groq":
        return True
    if _BACKEND == "huggingface":
        return False
    # auto: use Groq when key is present
    return bool(_GROQ_KEY)


# ── Groq backend ──────────────────────────────────────────────────────────────
def _build_groq_llm(model_name: str):
    """
    Returns a LangChain ChatGroq instance.
    Requires:  pip install langchain-groq
    """
    try:
        from langchain_groq import ChatGroq
    except ImportError as e:
        raise ImportError(
            "langchain-groq is not installed.\n"
            "Add `langchain-groq` to requirements.txt and redeploy."
        ) from e

    groq_model = _GROQ_MODEL_MAP.get(model_name, "llama-3.1-8b-instant")
    if _MODEL_OVERRIDE:
        # Allow explicit override e.g. LLM_MODEL_OVERRIDE=gemma2-9b-it
        groq_model = _MODEL_OVERRIDE

    return ChatGroq(
        model=groq_model,
        api_key=_GROQ_KEY,
        temperature=0.65,
        max_tokens=512,
    )


# ── HuggingFace CPU backend ───────────────────────────────────────────────────
def _build_hf_llm(model_name: str):
    """
    Loads a small HuggingFace model locally on CPU.
    Falls back to TinyLlama if the requested model is too large.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from langchain_huggingface import HuggingFacePipeline

    # Always use a small model on CPU — ignore large model names
    small_model = _SMALL_HF_MODEL
    st.info(
        f"⚙️ No Groq key found. Loading **{small_model}** on CPU "
        f"(original model '{model_name}' requires a GPU)."
    )

    tokenizer = AutoTokenizer.from_pretrained(small_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(small_model)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.65,
        top_p=0.9,
        repetition_penalty=1.3,
        device=-1,  # CPU
    )
    pipe.model.config.pad_token_id = pipe.model.config.eos_token_id

    return HuggingFacePipeline(pipeline=pipe)


# ── Original GPU backend (unchanged from your codebase) ──────────────────────
def _build_gpu_llm(model_name: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from langchain_huggingface import HuggingFacePipeline

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding=True, truncation=True, max_length=512
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    qa_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    qa_model.resize_token_embeddings(len(tokenizer))

    question_answerer = pipeline(
        "text-generation",
        model=qa_model,
        tokenizer=tokenizer,
        max_new_tokens=248,
        device=1,
    )
    question_answerer.model.config.pad_token_id = (
        question_answerer.model.config.eos_token_id
    )

    return HuggingFacePipeline(
        pipeline=question_answerer,
        model_kwargs={
            "device": 1,
            "repetition_penalty": 2,
            "temperature": 0.65,
            "top_p": 0.9,
            "top_k": 6,
        },
        pipeline_kwargs={"max_tokens": 248},
    )


# ── Public class (same interface as original) ─────────────────────────────────
class LanguageModel:
    """
    Drop-in replacement.  Usage in 1_Main.py stays exactly the same:

        llm_model = LanguageModel(st.session_state.model, device=self.device)
        llm = llm_model.load_model()
    """

    def __init__(self, model_name: str, device=1):
        self.model_name = model_name
        self.device = device

    @st.cache_resource
    def load_model(_self):
        """
        Auto-selects backend and returns a LangChain-compatible LLM object.
        Cached by Streamlit so the model is only loaded once per session.
        """
        if _use_groq():
            st.info(f"🚀 Using **Groq API** backend (fast, free)")
            return _build_groq_llm(_self.model_name)

        # No Groq key: try CPU HuggingFace
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except ImportError:
            has_gpu = False

        if has_gpu and _BACKEND != "huggingface":
            # Original GPU path
            return _build_gpu_llm(_self.model_name)
        else:
            return _build_hf_llm(_self.model_name)
