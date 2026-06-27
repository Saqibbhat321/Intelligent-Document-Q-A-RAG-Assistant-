"""Prompt builder — assembles context + question into an LLM prompt."""

from typing import List

from app.services.retrieval.retriever import RetrievedChunk

SYSTEM_PROMPT = """You are a helpful document assistant. \
Answer the user's question using ONLY the context provided below. \
If the answer cannot be found in the context, respond with exactly: \
"I could not find the answer in the uploaded documents."

Do not make up information. Do not use knowledge outside the provided context. \
Always be concise, precise, and factual."""


def build_prompt(question: str, retrieved_chunks: List[RetrievedChunk]) -> tuple[str, str]:
    """
    Construct the system and user messages for the LLM.

    Returns:
        (system_message, user_message)
    """
    if not retrieved_chunks:
        context_block = "(No relevant context found in the uploaded documents.)"
    else:
        context_parts: List[str] = []
        for i, rc in enumerate(retrieved_chunks, start=1):
            header = f"[Source {i}: {rc.source}, Page {rc.page_number}]"
            context_parts.append(f"{header}\n{rc.text}")
        context_block = "\n\n---\n\n".join(context_parts)

    user_message = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    return SYSTEM_PROMPT, user_message
