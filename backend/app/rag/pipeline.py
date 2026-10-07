import os
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv
from app.retrieval.search import search

# Load environment variables
load_dotenv()

logger = logging.getLogger("archiva.rag")

SYSTEM_PROMPT = """You are Archiva AI, an engineering knowledge assistant.
Answer the engineer's question using only the supplied engineering knowledge.
If the supplied knowledge does not contain enough information, say that the available engineering knowledge does not provide enough evidence.
Do not invent incidents, decisions, dates, teams, or technical solutions.
Mention which sources support the answer."""


def build_rag_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Construct the contextual prompt containing retrieved engineering sources."""
    context_blocks = []
    for idx, chunk in enumerate(retrieved_chunks, 1):
        context_blocks.append(
            f"[{idx}] Source: {chunk['document_id']} — {chunk['title']} ({chunk['document_type']})\n"
            f"Team: {chunk.get('team', 'N/A')} | Service: {chunk.get('service', 'N/A')}\n"
            f"Content: {chunk['content']}"
        )

    context_str = "\n\n".join(context_blocks)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== RETRIEVED ENGINEERING KNOWLEDGE ===\n"
        f"{context_str}\n\n"
        f"=== ENGINEER'S QUESTION ===\n"
        f"{query}\n\n"
        f"=== GROUNDED ANSWER WITH CITATIONS ==="
    )
    return prompt


def generate_deterministic_fallback(query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
    """
    Produce a deterministic, grounded synthesis from retrieved engineering chunks
    when an external LLM API key is not configured or unavailable.
    """
    if not relevant_chunks or (len(relevant_chunks) == 1 and relevant_chunks[0]["score"] < 0.20):
        return (
            "The available engineering knowledge in Archiva does not provide enough evidence "
            f"to answer: '{query}'. Please consult team documentation or submit an architecture review."
        )

    answer_lines = [
        f"Based on historical engineering records in Archiva, here is what was found regarding your question:\n"
    ]

    for item in relevant_chunks:
        doc_type = item.get("document_type", "Document")
        title = item.get("title", "Untitled")
        doc_id = item.get("document_id", "DOC")
        team = item.get("team", "Engineering")
        content = item.get("content", "").strip()

        if doc_type.lower() == "postmortem":
            answer_lines.append(f"• **Incident Record ({doc_id} - {title})** [{team}]: {content}")
        elif doc_type.lower() == "adr":
            answer_lines.append(f"• **Architecture Decision ({doc_id} - {title})** [{team}]: {content}")
        elif doc_type.lower() == "runbook":
            answer_lines.append(f"• **Operational Runbook ({doc_id} - {title})** [{team}]: {content}")
        else:
            answer_lines.append(f"• **Specification ({doc_id} - {title})** [{team}]: {content}")

    # Add citation summary
    citations = [f"{c['document_id']} — {c['title']} ({c['document_type']})" for c in relevant_chunks]
    answer_lines.append(f"\n**Supporting Sources:**\n" + "\n".join(f"- {cite}" for cite in citations))

    return "\n".join(answer_lines)


def generate_llm_answer(prompt: str, api_key: str) -> str:
    """Call Google Gemini API to generate a grounded answer."""
    last_err = None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        for model_name in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
            try:
                logger.info(f"Calling Gemini generate_content with model={model_name}")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                last_err = e
                logger.warning(f"Generation attempt with {model_name} failed: {e}")
                if "404" not in str(e) and "NOT_FOUND" not in str(e):
                    break
    except Exception as e_init:
        last_err = e_init
        logger.warning(f"google.genai SDK initialization failed: {e_init}")

    # Legacy google.generativeai fallback if installed
    try:
        import google.generativeai as genai_legacy
        genai_legacy.configure(api_key=api_key)
        for legacy_model in ["gemini-1.5-flash", "gemini-2.0-flash"]:
            try:
                model = genai_legacy.GenerativeModel(legacy_model)
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
            except Exception as e_leg:
                last_err = e_leg
                logger.warning(f"Legacy generation attempt with {legacy_model} failed: {e_leg}")
    except Exception as e_leg_import:
        logger.debug(f"google.generativeai fallback not available: {e_leg_import}")

    logger.warning(f"All Gemini generation models failed: {last_err}. Falling back to deterministic answer.")
    raise last_err


def answer_question(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Full RAG pipeline:
    1. Retrieve relevant engineering chunks using semantic vector search.
    2. Format structured sources that exactly match the cited documents.
    3. Generate grounded answer via Gemini LLM (if GEMINI_API_KEY set) or deterministic fallback.
    """
    # Step 1: Semantic Vector Retrieval
    retrieved_chunks = search(query=query, top_k=top_k)

    # Identify relevant chunks (matching the grounded answer selection threshold)
    if not retrieved_chunks or retrieved_chunks[0]["score"] < 0.20:
        relevant_chunks = []
    else:
        relevant_chunks = [c for c in retrieved_chunks if c["score"] >= 0.25]
        if not relevant_chunks:
            relevant_chunks = retrieved_chunks[:2]

    # Step 2: Format structured sources that exactly match the cited relevant documents
    cited_sources = []
    seen_docs = set()
    for chunk in relevant_chunks:
        doc_key = chunk["document_id"]
        if doc_key not in seen_docs:
            seen_docs.add(doc_key)
            cited_sources.append({
                "document_id": chunk["document_id"],
                "title": chunk["title"],
                "document_type": chunk["document_type"],
                "score": chunk["score"],
                "team": chunk.get("team"),
                "service": chunk.get("service")
            })

    # Step 3: Check for Gemini API key
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    answer_text = ""

    if gemini_key and relevant_chunks:
        try:
            rag_prompt = build_rag_prompt(query, relevant_chunks)
            answer_text = generate_llm_answer(rag_prompt, gemini_key)
        except Exception as err:
            logger.warning(f"LLM generation encountered error: {err}. Using deterministic synthesis.")
            answer_text = generate_deterministic_fallback(query, relevant_chunks)
    else:
        answer_text = generate_deterministic_fallback(query, relevant_chunks)

    return {
        "question": query,
        "answer": answer_text,
        "sources": cited_sources
    }
