import os
import sys
from groq import Groq
from dotenv import load_dotenv

sys.path.append(os.path.dirname(__file__))
from vectorstore import get_chroma_client, get_or_create_collection, query_collection

load_dotenv()
client_ai = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_NAME = "llama-3.3-70b-versatile"  # free, fast, strong model on Groq


PROMPT_TEMPLATE = """You are a helpful airline customer support assistant for IndiGo.
Answer the user's question using ONLY the information in the CONTEXT below.
If the answer is not present in the context, say "I don't have that information in my knowledge base" — do NOT make up an answer.
Always mention which source document(s) you used.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (be concise, cite the source document name):
"""


def generate_answer(question, collection, top_k=3):
    results = query_collection(collection, question, top_k=top_k)
    retrieved_chunks = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]

    context = "\n\n---\n\n".join(
        f"[Source: {src}]\n{chunk}"
        for src, chunk in zip(sources, retrieved_chunks)
    )

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = client_ai.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": list(set(sources)),
        "retrieved_chunks": retrieved_chunks,
    }


if __name__ == "__main__":
    client = get_chroma_client()
    collection = get_or_create_collection(client, "airline_policy_recursive_v2")

    test_questions = [
        "How much checked baggage can I carry on a domestic flight?",
        "What happens if my flight is delayed by 3 hours?",
        "Can I carry my pet snake on board?",
    ]

    for q in test_questions:
        result = generate_answer(q, collection)
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")