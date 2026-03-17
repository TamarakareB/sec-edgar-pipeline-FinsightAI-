import os, sys
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

sys.path.insert(0, str(Path(__file__).parent / "rag"))
from retrieve import search

COMPARISON_PROMPT = """You are FinSightAI, a financial analyst comparing multiple companies' SEC filings.

You will receive excerpts from multiple companies' filings on the same topic.
Write a clear, structured COMPARISON that:
1. Identifies what each company says about the topic
2. Highlights KEY DIFFERENCES between companies
3. Ranks or scores companies where appropriate
4. Cites every claim with [COMPANY - Filing Type - Year]
5. Ends with a 2-sentence "Analyst Takeaway"

Never make up data. If a company's filing does not address the topic, say so explicitly."""


def compare_companies(
    question: str,
    tickers: list[str],
    year: int = None,
    top_k_per_company: int = 3,
) -> dict:
    """
    Answers a comparative question across multiple companies simultaneously.
    """
    all_contexts   = []
    company_chunks = {}

    # Retrieve relevant chunks for EACH company independently
    for ticker in tickers:
        chunks = search(question, top_k=top_k_per_company, ticker=ticker, year=year)
        company_chunks[ticker] = chunks

        if chunks:
            company_text  = f"\n{'='*40}\nCOMPANY: {ticker}\n{'='*40}\n"
            for chunk in chunks:
                company_text += (
                    f"\n[{ticker} - {chunk.get('form','10-K')} - "
                    f"{str(chunk.get('filing_date',''))[:4]}]\n"
                    f"{chunk['text'][:600]}\n"
                )
            all_contexts.append(company_text)
        else:
            all_contexts.append(f"\n{'='*40}\nCOMPANY: {ticker}\nNo relevant filings found.\n")

    if not any(company_chunks.values()):
        return {"answer": "No relevant documents found for any of the requested companies.", "companies": tickers}

    combined_context = "\n".join(all_contexts)
    full_prompt = (
        f"{COMPARISON_PROMPT}\n\n"
        f"FILING EXCERPTS FROM {len(tickers)} COMPANIES:\n{combined_context}\n\n"
        f"COMPARISON QUESTION: {question}\n\n"
        f"Write a detailed comparative analysis."
    )

    try:
        response = model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=1200)
        )
        answer = response.text
    except Exception as e:
        answer = f"Error calling Gemini API: {e}"

    # Build per-company citation map
    citations = {
        ticker: [
            {
                "citation": f"{c.get('ticker')} {c.get('form','10-K')} {str(c.get('filing_date',''))[:4]}",
                "score":    round(c.get("relevance_score", 0), 3),
                "preview":  c["text"][:200],
            }
            for c in chunks
        ]
        for ticker, chunks in company_chunks.items()
    }

    return {
        "question":           question,
        "companies":          tickers,
        "year":               year,
        "answer":             answer,
        "citations":          citations,
        "chunks_per_company": {t: len(c) for t, c in company_chunks.items()},
    }


def sector_comparison(question: str, sector: str, year: int = None) -> dict:
    """Compares all companies within a sector."""
    SECTOR_TICKERS = {
        "technology":  ["AAPL","MSFT","NVDA","GOOGL","META","ADBE","CRM","AMD","INTC","CSCO"],
        "financials":  ["JPM","BAC","WFC","GS","MS","V","MA","BLK","SPGI"],
        "healthcare":  ["UNH","JNJ","LLY","MRK","ABBV","TMO","BMY"],
        "energy":      ["XOM","CVX"],
        "consumer":    ["WMT","COST","PG","KO","PEP","MCD","SBUX"],
        "industrials": ["CAT","DE","RTX"],
    }
    tickers = SECTOR_TICKERS.get(sector.lower(), [])
    if not tickers:
        return {"error": f"Unknown sector '{sector}'. Choose from: {list(SECTOR_TICKERS.keys())}"}
    return compare_companies(question, tickers=tickers[:5], year=year)
