import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from vectorstore import get_chroma_client, get_or_create_collection
from generation import generate_answer
from chunking import load_documents, chunk_fixed_size, chunk_recursive, chunk_by_section
from vectorstore import index_chunks

st.set_page_config(page_title="IndiGo RAG Assistant", page_icon="✈️", layout="centered")

# --- Auto-index on first run (for cloud deployment where chroma_db doesn't persist from local) ---
@st.cache_resource
def ensure_indexed():
    client = get_chroma_client()
    docs = load_documents(os.path.join(os.path.dirname(__file__), "data"))

    collections_config = {
        "airline_policy_fixed": chunk_fixed_size,
        "airline_policy_recursive_v2": chunk_recursive,
        "airline_policy_section": chunk_by_section,
    }

    for coll_name, chunk_fn in collections_config.items():
        collection = get_or_create_collection(client, coll_name)
        if collection.count() == 0:  # only index if empty
            chunks = chunk_fn(docs)
            index_chunks(collection, chunks, strategy_name=coll_name)

    return client

client = ensure_indexed()
st.title("✈️ IndiGo Customer Support Assistant")
st.caption("A RAG-based assistant answering questions from IndiGo's official policy documents — baggage, cancellation, check-in, and food.")

# Sidebar: strategy selector (demo of your chunking evaluation work)
st.sidebar.header("⚙️ Settings")
strategy_map = {
    "Recursive (recommended)": "airline_policy_recursive_v2",
    "Fixed-size": "airline_policy_fixed",
    "Section-aware": "airline_policy_section",
}
selected_strategy = st.sidebar.selectbox("Chunking strategy", list(strategy_map.keys()))
top_k = st.sidebar.slider("Number of chunks retrieved (top-k)", 1, 5, 3)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About this project**\n\n"
    "This assistant retrieves answers from IndiGo's official Conditions of Carriage, "
    "baggage policy, cancellation policy, and food menu documents using semantic search, "
    "then generates a grounded answer citing the source document. "
    "It will explicitly say 'I don't know' rather than guess, to avoid hallucination."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

collection = get_or_create_collection(client, strategy_map[selected_strategy])

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            st.caption(f"📄 Sources: {', '.join(msg['sources'])}")

# Chat input
user_query = st.chat_input("Ask about baggage, cancellation, check-in, or food...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents..."):
            result = generate_answer(user_query, collection, top_k=top_k)
        st.markdown(result["answer"])
        st.caption(f"📄 Sources: {', '.join(result['sources'])}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })

# Example questions
st.markdown("---")
st.markdown("**Try asking:**")
example_cols = st.columns(2)
examples = [
    "How much baggage can I carry on a domestic flight?",
    "What is Plan B?",
    "When does web check-in open?",
    "Do you serve hot meals on all flights?",
]
for i, example in enumerate(examples):
    if example_cols[i % 2].button(example):
        st.session_state.pending_query = example
        st.rerun()