from sentence_transformers import SentenceTransformer

# Multilingual model — future-proof agar kabhi Hindi query bhi test karni ho
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # fast, lightweight, good baseline


class EmbeddingModel:
    def __init__(self, model_name=MODEL_NAME):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        """Embed a list of chunk texts -> list of vectors."""
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text):
        """Embed a single query string -> vector."""
        return self.model.encode([text])[0].tolist()


if __name__ == "__main__":
    embedder = EmbeddingModel()
    sample_texts = [
        "IndiGo allows 15 kg checked baggage on domestic flights.",
        "Plan B allows free rescheduling if IndiGo delays a flight by 2 hours.",
    ]
    vectors = embedder.embed_documents(sample_texts)
    print(f"Number of vectors: {len(vectors)}")
    print(f"Vector dimension: {len(vectors[0])}")