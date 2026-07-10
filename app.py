import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from vectorstore import get_chroma_client, get_or_create_collection
from generation import generate_answer

st.set_page_config(page_title="IndiGo RAG Assistant", page_icon="✈️", layout="centered")

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

client = get_chroma_client()
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