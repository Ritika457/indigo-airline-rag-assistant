import sys, os
sys.path.append(os.path.dirname(__file__))
from chunking import load_documents, chunk_fixed_size, chunk_recursive, chunk_by_section
from vectorstore import get_chroma_client, get_or_create_collection, index_chunks

docs = load_documents('D://Airline_RAG//data')

client = get_chroma_client()

# Fixed-size
fixed_chunks = chunk_fixed_size(docs)
fixed_collection = get_or_create_collection(client, "airline_policy_fixed")
index_chunks(fixed_collection, fixed_chunks, strategy_name="fixed")

# Recursive (skip if already indexed — delete old collection first if re-running)
recursive_chunks = chunk_recursive(docs)
recursive_collection = get_or_create_collection(client, "airline_policy_recursive_v2")
index_chunks(recursive_collection, recursive_chunks, strategy_name="recursive")

# Section-aware
section_chunks = chunk_by_section(docs)
section_collection = get_or_create_collection(client, "airline_policy_section")
index_chunks(section_collection, section_chunks, strategy_name="section")

print("\nAll strategies indexed successfully.")