import streamlit as st
def cleanup_model():
    """Clean up model from memory"""
    if 'llm' in st.session_state:
        # Move model to CPU first (helps with cleanup)
        if hasattr(st.session_state.llm, 'to'):
            st.session_state.llm.to('cpu')
        
        del st.session_state.llm
        
    if 'rag' in st.session_state:
        # Move model to CPU first (helps with cleanup)
        if hasattr(st.session_state.rag, 'to'):
            st.session_state.rag.to('cpu')
        
        del st.session_state.rag
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
# In your main app or page selector
def switch_page(new_page):
    # Clean up before switching
    cleanup_model()
    st.session_state.current_page = new_page
    st.rerun()
    
st.set_page_config(page_title="CA-RAG", layout="wide")

st.title("CA-RAG Dashboard")
st.write("Welcome to CA-RAG pipeline: Generate and refine policies using RAG + RLHF")

st.sidebar.title("Navigation")
st.sidebar.write("Use the sidebar to navigate:")

# Streamlit will automatically create sidebar links for pages in /pages
st.sidebar.success("Select a module above")
