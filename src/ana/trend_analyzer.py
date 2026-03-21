import os, sys, sqlite3
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

sys.path.insert(0, str(Path(__file__).parent / "rag"))
from src.rag.retrieve import search

DB_PATH = Path("data/sql/finsightai.db")

TREND_PROMPT = """You are FinSightAI performing a 5-year trend analysis of SEC filings.

You will receive excerpts from the SAME company across multiple years on the same topic.

Your analysis must:
1. Describe how the company's language/focus on this topic CHANGED over time
2. Identify when new risks or themes FIRST appeared
3. Note any risks that DISAPPEARED or became less prominent
4. Highlight the most significant year-over-year shift
5. Cite every observation with [TICKER - Year]
6. End with a "Trend Verdict": Is this risk increasing, decreasing, or stable?

Be specific about language changes, not just general summaries."""


def analyze_trend(ticker: str, question: str, years: list[int] = None) -> dict:
    """
    Analyzes how a topic has changed for one company across 5 years.
    """
    if years is None:
        years = [2019, 2020, 2021, 2022, 2023]

    yearly_contexts = []
    yearly_chunks   = {}

    for year in sorted(years):
        chunks = search(question, top_k=2, ticker=ticker, year=year)
        yearly_chunks[year] = chunks
        if chunks:
            year_text = f"\n[{ticker} - {year}]\n{chunks[0]['text'][:700]}"
            yearly_contexts.append(year_text)

    if not any(yearly_chunks.values()):
        return {"error": f"No filings found for {ticker}"}

    combined   = "\n\n---\n\n".join(yearly_contexts)
    full_prompt = (
        f"{TREND_PROMPT}\n\n"
        f"COMPANY: {ticker}\n"
        f"YEARS ANALYZED: {', '.join(str(y) for y in years if yearly_chunks.get(y))}\n"
        f"TOPIC: {question}\n\n"
        f"FILING EXCERPTS BY YEAR:\n{combined}\n\n"
        f"Write a detailed 5-year trend analysis."
    )

    try:
        response = model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=1000)
        )
        answer = response.text
    except Exception as e:
        answer = f"Error calling Gemini API: {e}"

    return {
        "ticker":      ticker,
        "question":    question,
        "years":       years,
        "answer":      answer,
        "data_points": {y: len(c) for y, c in yearly_chunks.items()},
    }


def get_financial_trend(ticker: str) -> dict:
    """Pulls the 15 financial metrics from SQL across all years for charting."""
    if not DB_PATH.exists():
        return {"error": "Database not found. Run sql_database.py first."}

    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT year, total_revenue, net_income, gross_profit, operating_income,
               eps_diluted, total_assets, total_liabilities, total_equity,
               long_term_debt, cash_and_equivalents, operating_cash_flow,
               capital_expenditures, research_and_development
        FROM financial_metrics
        WHERE ticker = ? AND form = '10-K'
        ORDER BY year
    """, (ticker,))

    rows    = cursor.fetchall()
    columns = ["year","total_revenue","net_income","gross_profit","operating_income",
               "eps_diluted","total_assets","total_liabilities","total_equity",
               "long_term_debt","cash_and_equivalents","operating_cash_flow",
               "capital_expenditures","research_and_development"]
    conn.close()

    if not rows:
        return {"error": f"No financial data found for {ticker}"}

    records = [dict(zip(columns, row)) for row in rows]
    for i in range(1, len(records)):
        prev, curr = records[i-1], records[i]
        for field in ["total_revenue", "net_income"]:
            if prev.get(field) and curr.get(field) and prev[field] != 0:
                growth = (curr[field] - prev[field]) / abs(prev[field]) * 100
                curr[f"{field}_yoy_growth"] = round(growth, 1)

    return {
        "ticker":             ticker,
        "records":            records,
        "years":              [r["year"] for r in records],
        "available_metrics":  [c for c in columns if c != "year"],
    }


def detect_emerging_risks(ticker: str) -> dict:
    """
    Compares risk language between 2019 and 2023 to detect new, stable, and faded risks.
    """
    EMERGING_RISK_TERMS = [
        "artificial intelligence", "machine learning", "generative ai",
        "climate change", "climate risk", "carbon",
        "geopolitical", "china tariff", "russia",
        "cybersecurity", "ransomware", "data breach",
        "pandemic", "supply chain disruption",
        "inflation", "interest rate risk",
    ]

    early_chunks  = search("risk factors", top_k=10, ticker=ticker, year=2019)
    recent_chunks = search("risk factors", top_k=10, ticker=ticker, year=2023)
    early_text    = " ".join(c["text"].lower() for c in early_chunks)
    recent_text   = " ".join(c["text"].lower() for c in recent_chunks)

    emerging, stable, faded = [], [], []
    for term in EMERGING_RISK_TERMS:
        in_early  = term in early_text
        in_recent = term in recent_text
        if in_recent and not in_early:
            emerging.append(term)
        elif in_early and in_recent:
            stable.append(term)
        elif in_early and not in_recent:
            faded.append(term)

    return {
        "ticker":   ticker,
        "emerging": emerging,
        "stable":   stable,
        "faded":    faded,
        "summary": (
            f"{ticker} shows {len(emerging)} emerging risks since 2019: "
            f"{', '.join(emerging) if emerging else 'none detected'}. "
            f"{len(faded)} risk topics faded: {', '.join(faded) if faded else 'none'}."
        )
    }
