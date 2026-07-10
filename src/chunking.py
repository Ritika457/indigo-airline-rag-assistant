import os
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_core.documents import Document


def load_documents(data_dir="data"):
    """Load all .txt files from data directory as LangChain Documents."""
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": filename}
                )
            )
    print(f"Loaded {len(documents)} documents.")
    return documents


def chunk_fixed_size(documents, chunk_size=500, chunk_overlap=50):
    """Strategy 1: Fixed-size chunking (naive, splits by character count)."""
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"fixed_{i}"
        chunk.metadata["strategy"] = "fixed_size"
    print(f"[Fixed-size] Created {len(chunks)} chunks.")
    return chunks


def chunk_recursive(documents, chunk_size=500, chunk_overlap=50):
    """Strategy 2: Recursive chunking (respects paragraph/section boundaries)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"recursive_{i}"
        chunk.metadata["strategy"] = "recursive"
    print(f"[Recursive] Created {len(chunks)} chunks.")
    return chunks


def chunk_by_section(documents):
    """Strategy 3: Section-aware chunking — splits on the ALL-CAPS headers
    already present in our policy docs (e.g. 'BAGGAGE TAGS', 'REFUND PROCESSING')."""
    import re
    chunks = []
    chunk_counter = 0
    for doc in documents:
        # Split on lines that look like section headers (all caps, followed by a dashed line)
        sections = re.split(r"\n(?=[A-Z][A-Z \-/&()]+\n-+\n)", doc.page_content)
        for section in sections:
            section = section.strip()
            if len(section) < 20:  # skip near-empty fragments
                continue
            chunks.append(
                Document(
                    page_content=section,
                    metadata={
                        "source": doc.metadata["source"],
                        "chunk_id": f"section_{chunk_counter}",
                        "strategy": "section_aware",
                    },
                )
            )
            chunk_counter += 1
    print(f"[Section-aware] Created {len(chunks)} chunks.")
    return chunks


if __name__ == "__main__":
    docs = load_documents('D://Airline_RAG//data')

    fixed_chunks = chunk_fixed_size(docs)
    recursive_chunks = chunk_recursive(docs)
    section_chunks = chunk_by_section(docs)

    print("\n--- Sample fixed-size chunk ---")
    print(fixed_chunks[0].page_content[:200])

    print("\n--- Sample recursive chunk ---")
    print(recursive_chunks[0].page_content[:200])

    print("\n--- Sample section-aware chunk ---")
    print(section_chunks[0].page_content[:200])