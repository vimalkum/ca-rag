import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingDB:
    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device
        self.embeddings = None
        self.db = None

    @st.cache_resource
    def load(_self, _docs):
        _self.embeddings = HuggingFaceEmbeddings(
            model_name=_self.model_path,
            model_kwargs={'device': _self.device},
            encode_kwargs={'normalize_embeddings': False}
        )
        _self.db = FAISS.from_documents(_docs, _self.embeddings)
        return _self.embeddings, _self.db


