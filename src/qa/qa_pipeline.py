# FinSightAI — QA Pipeline using Google Gemini API
import os, sys, json, re
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")   # Fast + cheap. Change to gemini-1.5-pro for higher quality.

sys.path.insert(0, str(Path(__file__).parent / "rag"))
from src.rag.retrieve import search
from src.rag.section_detector import get_priority_sections, get_section_label

SYSTEM_PROMPT = """You are FinSightAI, a precise financial document analyst.

STRICT RULES:
1. Answer ONLY using the document excerpts provided. Never use outside knowledge.
2. Cite EVERY factual claim with [Source N: TICKER Filing Year].
3. Use exact numbers and dates from the documents when available.
4. If the answer is not in the documents, say: "The provided filings do not address this."
5. Never speculate or infer beyond what is written.
6. Structure long answers with clear headings."""

FOLLOWUP_PROMPT = """Based on this financial question and answer, suggest 3 natural follow-up questions a financial analyst might ask next. Return ONLY a JSON array of 3 strings, no other text."""


def ask(
    question: str,
    ticker: str = None,
    year: int = None,
    top_k: int = 5,
    include_followups: bool = True,
) -> dict:
    """
    Main QA function. Returns grounded answer with citations,
    confidence score, and optional follow-up question suggestions.
    """

    # ── Step 1: Section-aware retrieval ──────────────────────────────────────
    priority_sections = get_priority_sections(question)
    chunks = search(question, top_k=top_k, ticker=ticker, year=year)

    # Boost chunks from priority sections to the top
    if priority_sections and chunks:
        priority_chunks = [c for c in chunks if c.get("section") in priority_sections]
        other_chunks    = [c for c in chunks if c.get("section") not in priority_sections]
        chunks = (priority_chunks + other_chunks)[:top_k]

    if not chunks:
        return {
            "answer":     "No relevant documents found for your query.",
            "citations":  [],
            "confidence": 0.0,
            "followups":  [],
        }

    # ── Step 2: Build context ─────────────────────────────────────────────────
    context_parts = []
    for i, chunk in enumerate(chunks):
        section_label = get_section_label(chunk.get("section", "unknown"))
        year_str      = str(chunk.get("filing_date", ""))[:4] or str(chunk.get("year", ""))
        context_parts.append(
            f"[Source {i+1}: {chunk.get('ticker','')} "
            f"{chunk.get('form','10-K')} {year_str} — {section_label}]\n"
            f"{chunk['text'][:800]}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # ── Step 3: Generate answer with Gemini ───────────────────────────────────
    full_prompt = f"{SYSTEM_PROMPT}\n\nDOCUMENT EXCERPTS:\n{context}\n\nQUESTION: {question}"

    try:
        response = model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=900,
            )
        )
        answer = response.text
    except Exception as e:
        answer = f"Error calling Gemini API: {e}"

    # ── Step 4: Confidence scoring ────────────────────────────────────────────
    has_citations  = bool(re.search(r"\[source\s*\d", answer.lower()))
    has_numbers    = bool(re.search(r"\$[\d,]+|\d+\.\d+%|\d{4}", answer))
    has_negation   = "do not address" in answer.lower() or "not found" in answer.lower()
    avg_chunk_score = sum(c.get("relevance_score", 0) for c in chunks) / len(chunks) if chunks else 0

    confidence = min(1.0, (
        (0.4 if has_citations else 0.0) +
        (0.2 if has_numbers   else 0.0) +
        (0.0 if has_negation  else 0.1) +
        avg_chunk_score * 0.3
    ))

    # ── Step 5: Generate follow-up questions ─────────────────────────────────
    followups = []
    if include_followups:
        try:
            fu_response = model.generate_content(
                f"{FOLLOWUP_PROMPT}\n\nQuestion: {question}\nAnswer summary: {answer[:300]}",
                generation_config=genai.GenerationConfig(temperature=0.7, max_output_tokens=200)
            )
            fu_text = fu_response.text.strip()
            fu_text = fu_text[fu_text.find("["):fu_text.rfind("]") + 1]
            followups = json.loads(fu_text)
        except Exception:
            followups = []

    # ── Step 6: Build citations ───────────────────────────────────────────────
    citations = [
        {
            "source":   i + 1,
            "ticker":   c.get("ticker", ""),
            "form":     c.get("form", "10-K"),
            "year":     str(c.get("filing_date", ""))[:4] or str(c.get("year", "")),
            "section":  get_section_label(c.get("section", "unknown")),
            "score":    round(c.get("relevance_score", 0), 3),
            "preview":  c["text"][:250],
            "url":      c.get("source_url", ""),
        }
        for i, c in enumerate(chunks)
    ]

    return {
        "answer":        answer,
        "citations":     citations,
        "confidence":    round(confidence, 2),
        "followups":     followups,
        "chunks_used":   len(chunks),
        "sections_used": list({c.get("section", "unknown") for c in chunks}),
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = ask("What are Apple's main liquidity risks?", ticker="AAPL", year=2023, include_followups=False)
    print("ANSWER:", result["answer"][:500])
    print(f"\nConfidence: {result['confidence']}")
    print(f"Sections used: {result['sections_used']}")
