"""Prompt builder — assembles context + conversation history + question into an LLM prompt."""

from typing import List, Optional

from app.services.retrieval.retriever import RetrievedChunk

SYSTEM_PROMPT = """You are a helpful document assistant. \
Answer the user's question using ONLY the context provided below. \
If the answer cannot be found in the context, respond with exactly: \
"I could not find the answer in the uploaded documents."

You also have access to the recent conversation history. \
Use it to understand follow-up questions and references like "tell me more" or "what about that". \
Do not make up information. Do not use knowledge outside the provided context. \
Always be concise, precise, and factual."""


def build_prompt(
    question: str,
    retrieved_chunks: List[RetrievedChunk],
    conversation_history: Optional[List[dict]] = None,
) -> tuple[str, str]:
    """
    Construct the system and user messages for the LLM.

    Args:
        question: The current user question.
        retrieved_chunks: Top-K chunks retrieved from FAISS.
        conversation_history: List of dicts with keys 'question' and 'answer'
                              (most recent last, max 4 turns).

    Returns:
        (system_message, user_message)
    """
    # ── Retrieved context ──────────────────────────────────────────────────────
    if not retrieved_chunks:
        context_block = "(No relevant context found in the uploaded documents.)"
    else:
        context_parts: List[str] = []
        for i, rc in enumerate(retrieved_chunks, start=1):
            header = f"[Source {i}: {rc.source}, Page {rc.page_number}]"
            context_parts.append(f"{header}\n{rc.text}")
        context_block = "\n\n---\n\n".join(context_parts)

    # ── Conversation history ───────────────────────────────────────────────────
    history_block = ""
    if conversation_history:
        turns: List[str] = []
        for turn in conversation_history:
            turns.append(f"User: {turn['question']}\nAssistant: {turn['answer']}")
        history_block = (
            "Recent conversation:\n"
            + "\n\n".join(turns)
            + "\n\n"
        )

    user_message = (
        f"{history_block}"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    return SYSTEM_PROMPT, user_message
