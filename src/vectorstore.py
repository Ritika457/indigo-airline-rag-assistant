import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "chroma_db"  # local folder jahan DB store hoga
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_or_create_collection(client, collection_name):
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )
    return collection


def index_chunks(collection, chunks, strategy_name="default"):
    """Add chunks (LangChain Document objects) into the ChromaDB collection."""
    documents = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"{strategy_name}_{i}" for i in range(len(chunks))]

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    print(f"Indexed {len(chunks)} chunks into collection '{collection.name}'.")


def query_collection(collection, query_text, top_k=3):
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
    )
    return results


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from chunking import load_documents, chunk_recursive

    # Load + chunk documents
    docs = load_documents('D://Airline_RAG//data')
    chunks = chunk_recursive(docs)

    # Setup ChromaDB
    client = get_chroma_client()
    collection = get_or_create_collection(client, "airline_policy_recursive")

    # Index chunks (only run once — comment out after first run to avoid duplicates)
    index_chunks(collection, chunks, strategy_name="recursive")

    # Test query
    test_query = "How much baggage can I carry on a domestic flight?"
    results = query_collection(collection, test_query, top_k=3)

    print(f"\n--- Query: {test_query} ---")
    for i, doc in enumerate(results["documents"][0]):
        print(f"\nResult {i+1} (source: {results['metadatas'][0][i]['source']}):")
        print(doc[:300])