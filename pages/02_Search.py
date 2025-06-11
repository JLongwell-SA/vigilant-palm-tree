
import streamlit as st
from utils.utils import encode_search_rerank


# st.page_link("Chat.py", label="💬 Go to Proposal Chat", icon="💬")

st.title("🔎📄Hybrid Search over Engineering Proposals")

# --- Init session state ---
if "search_query" not in st.session_state:
    st.session_state.search_query = None
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "top_k" not in st.session_state:
    st.session_state.top_k = 5

# Separate the input area
input_container = st.container()
result_container = st.container()

with input_container:
    user_query = st.chat_input("Enter your search query:")
    top_n = st.slider("Number of results", 1, 20, st.session_state.top_n)
    st.markdown("---")

# --- Process new query ---
if user_query:
    with st.spinner("Embedding and searching..."):
        result = encode_search_rerank(user_query,  "", "", top_k=50, top_n=top_n, alpha = 0.75)
        st.session_state.search_query = user_query
        st.session_state.search_results = result[0]
        st.session_state.top_k = 50
        st.session_state.top_n = top_n

# --- Show results from session ---
if st.session_state.search_results:
    with result_container:
        st.success(f"Showing results for: {st.session_state.search_query}")
        
        for i, match in enumerate(st.session_state.search_results.data):

            doc = match['document']
            doc_id = doc['id']
            metadata = doc.get('metadata', {})

            chunk_no = doc_id.split("_")[-1]
            og_document_title = "_".join(doc_id.split("_")[:-1])
            chunk_text = metadata.get("chunk", "[No chunk text available]")
             # Use columns for better layout
            col1, col2 = st.columns([1, 3])

            with col1:
                st.markdown(f"**📘 Result {i+1}**")
                st.markdown(f"Score: `{match['score']:.4f}`")
                st.markdown(f"Doc: `{og_document_title}`")
                st.markdown(f"Chunk: `{chunk_no}`")

            with col2:
                st.markdown("**Content:**")
                st.markdown(chunk_text)

            if i < len(st.session_state.search_results.data) - 1:
                st.divider()
