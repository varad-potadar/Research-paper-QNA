import os
import re
from groq import Groq
from services.embedder import embed_chunks

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def clean_chunk(text: str) -> str:
    text = text.replace("+e", "the")
    text = text.replace("+is", "this")
    text = text.replace("ﬁ", "fi")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def answer_question(question: str, vector_store):
    # 1. Embed question
    q_embedding = embed_chunks([question])

    # 2. Retrieve MORE context (important)
    retrieved_chunks = vector_store.search(q_embedding, k=8)

    # 3. Clean + filter
    retrieved_chunks = [
        clean_chunk(c)
        for c in retrieved_chunks
        if len(c.split()) > 25
    ]

    print("DEBUG: Retrieved chunks:")
    for i, c in enumerate(retrieved_chunks):
        print(f"--- chunk {i} ---\n{c}\n")

    context = "\n\n".join(retrieved_chunks)

    # 🔑 CRITICAL PROMPT CHANGE
    prompt = f"""
You are a document-grounded question answering system.

Using ONLY the information provided in the context below,
answer the question accurately and concisely.

You may summarize or combine information from multiple
parts of the context, but you MUST NOT use any external
knowledge or assumptions.

If the answer cannot be determined strictly from the
context, respond EXACTLY with:
Not found in the document.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You answer questions using only the provided document context. Do not hallucinate."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content.strip()

    print("DEBUG: Model answer:", answer)
    return answer

