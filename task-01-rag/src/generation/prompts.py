"""Prompt engineering and system templates for grounded RAG generation."""

from typing import List
from src.vectorstore.faiss_store import SearchResult

SYSTEM_PROMPT = """You are an authoritative, precise, and helpful AI Document Assistant.
Your mission is to answer the user's questions based STRICTLY and EXCLUSIVELY on the provided document passages.

CRITICAL INSTRUCTIONS & GROUNDING RULES:
1. STRICT GROUNDING: Answer ONLY using information explicitly stated in the context passages. Do not bring in external knowledge, assumptions, or unverified claims.
2. ANTI-HALLUCINATION: If the provided context does not contain enough information to answer the user's question, you MUST clearly state:
   "Based on the provided documents, I could not find information regarding [topic]."
   Never guess, fabricate facts, or speculate.
3. CITATIONS: Whenever you state a fact from a passage, append an inline citation with the document name and page number, for example: [Doc: report.pdf, Page: 3].
4. TONE & STRUCTURE: Be concise, direct, and well-structured. Use bullet points or numbered lists when explaining multi-step processes or key findings.
5. CONTRADICTIONS: If different passages provide conflicting information, explicitly point out the discrepancy with citations for each passage.
"""


def format_context_block(results: List[SearchResult]) -> str:
    """Format retrieved search results into a clean, numbered context block.

    Args:
        results: List of SearchResult instances containing text chunks and scores.

    Returns:
        Structured string block ready for LLM consumption.
    """
    if not results:
        return "No relevant context passages found."

    context_parts: List[str] = []
    for idx, res in enumerate(results, start=1):
        chunk = res.chunk
        passage_str = (
            f"--- PASSAGE {idx} ---\n"
            f"Source Document: {chunk.source_name}\n"
            f"Page Number: {chunk.page_number}\n"
            f"Chunk Index: {chunk.chunk_index}\n"
            f"Cosine Similarity: {res.score:.4f}\n"
            f"Passage Content:\n{chunk.text.strip()}\n"
        )
        context_parts.append(passage_str)

    return "\n".join(context_parts)


def build_user_prompt(query: str, context_block: str) -> str:
    """Construct the final user prompt incorporating retrieved context and query."""
    return f"""CONTEXT PASSAGES:
{context_block}

USER QUESTION:
{query}

Please provide a grounded, cited answer following the system rules:"""
