import os, sys, json, sqlite3
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
judge_model = genai.GenerativeModel("gemini-1.5-flash")

sys.path.insert(0, str(Path(__file__).parent))
from qa_pipeline import ask

DB_PATH = Path("data/sql/finsightai.db")
OUT_DIR = Path("data/dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Ground-truth test questions ───────────────────────────────────────────────
TEST_QUESTIONS = [
    {"id":"Q1","question":"What are Apple's main liquidity risks?",           "ticker":"AAPL","year":2023,"keywords":["cash","liquidity","debt","obligations","capital"],        "category":"risk_analysis"},
    {"id":"Q2","question":"What was Microsoft's total revenue in 2023?",      "ticker":"MSFT","year":2023,"keywords":["revenue","billion","2023","sales"],                       "category":"financial_metrics"},
    {"id":"Q3","question":"What supply chain risks does Tesla mention?",       "ticker":"TSLA","year":2023,"keywords":["supply","supplier","component","manufacturing","shortage"],"category":"risk_analysis"},
    {"id":"Q4","question":"What regulatory risks does JPMorgan Chase face?",   "ticker":"JPM", "year":2022,"keywords":["regulatory","compliance","federal","capital","regulation"],"category":"risk_analysis"},
    {"id":"Q5","question":"How much did Amazon spend on capital expenditures?","ticker":"AMZN","year":2022,"keywords":["capital","expenditure","billion","investment"],            "category":"financial_metrics"},
    {"id":"Q6","question":"What are NVIDIA's main market risks?",             "ticker":"NVDA","year":2023,"keywords":["market","competition","demand","semiconductor","risk"],    "category":"risk_analysis"},
    {"id":"Q7","question":"What does Apple say about R&D spending?",          "ticker":"AAPL","year":2022,"keywords":["research","development","billion","innovation"],           "category":"financial_metrics"},
    {"id":"Q8","question":"What operational risks does Walmart identify?",     "ticker":"WMT", "year":2023,"keywords":["operational","technology","cybersecurity","store"],        "category":"risk_analysis"},
]


# ── Scoring functions ─────────────────────────────────────────────────────────

def score_completeness(answer: str, keywords: list) -> float:
    answer_lower = answer.lower()
    found = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(found / len(keywords), 2) if keywords else 0.0


def score_hallucination(answer: str, citations: list) -> dict:
    answer_lower   = answer.lower()
    no_info_phrases = ["do not address","not mentioned","no information","not provided","not found in"]
    if any(p in answer_lower for p in no_info_phrases):
        return {"flag": False, "reason": "System correctly acknowledged missing info"}
    if not citations:
        return {"flag": True,  "reason": "No source chunks were retrieved"}
    if "[source" not in answer_lower and "source 1" not in answer_lower:
        return {"flag": True,  "reason": "Answer not grounded — no [Source N] citations found"}
    return {"flag": False, "reason": "No hallucination signals detected"}


def score_faithfulness_gemini(question: str, answer: str, context: str) -> float:
    """Uses Gemini as LLM-as-judge to rate faithfulness 0.0–1.0."""
    prompt = f"""You are evaluating an AI system for faithfulness to source documents.

Question: {question}

Source document excerpts:
{context[:1500]}

Generated answer:
{answer[:600]}

Rate the answer's FAITHFULNESS on a scale of 0–10:
10 = Every claim directly supported by documents
7-9 = Most claims supported, minor liberties
4-6 = Some claims supported, some questionable
1-3 = Significant unsupported claims
0   = Answer contradicts or ignores documents

Respond with ONE number only (0-10). No explanation."""

    try:
        response = judge_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0, max_output_tokens=5)
        )
        return round(float(response.text.strip()) / 10.0, 2)
    except Exception as e:
        print(f"  [Gemini scoring failed: {e}]")
        return -1.0


def score_retrieval_quality(citations: list, expected_ticker: str) -> float:
    if not citations:
        return 0.0
    correct = sum(1 for c in citations if c.get("ticker","").upper() == expected_ticker.upper())
    return round(correct / len(citations), 2)


# ── Main evaluation runner ─────────────────────────────────────────────────────

def run_evaluation():
    print("=" * 60)
    print("FinSightAI Evaluation Suite  |  Powered by Gemini")
    print(f"Running {len(TEST_QUESTIONS)} test questions...")
    print("=" * 60)

    results, all_complete, all_faith, all_retrieval = [], [], [], []
    halluc_count = 0

    for test in TEST_QUESTIONS:
        print(f"\n[{test['id']}] {test['question']}")
        print(f"  Ticker: {test['ticker']} | Year: {test['year']}")

        result     = ask(test["question"], ticker=test["ticker"], year=test["year"], include_followups=False)
        answer     = result.get("answer", "")
        citations  = result.get("citations", [])
        confidence = result.get("confidence", 0.0)
        sections   = result.get("sections_used", [])
        context    = " ".join(c.get("preview","") for c in citations[:3])

        completeness  = score_completeness(answer, test["keywords"])
        hallucination = score_hallucination(answer, citations)
        faithfulness  = score_faithfulness_gemini(test["question"], answer, context)
        retrieval_q   = score_retrieval_quality(citations, test["ticker"])

        all_complete.append(completeness)
        all_retrieval.append(retrieval_q)
        if faithfulness >= 0:
            all_faith.append(faithfulness)
        if hallucination["flag"]:
            halluc_count += 1

        print(f"  Completeness:  {completeness:.2f}")
        print(f"  Faithfulness:  {faithfulness:.2f}" if faithfulness >= 0 else "  Faithfulness:  N/A")
        print(f"  Retrieval@k:   {retrieval_q:.2f}")
        print(f"  Hallucination: {'YES ⚠️' if hallucination['flag'] else 'No ✅'}")
        print(f"  Confidence:    {confidence:.2f}")

        results.append({
            "id": test["id"], "question": test["question"],
            "ticker": test["ticker"], "year": test["year"],
            "category": test["category"], "answer": answer,
            "completeness": completeness, "faithfulness": faithfulness,
            "retrieval_k": retrieval_q, "hallucination": hallucination,
            "confidence": confidence, "sections_used": sections,
            "num_citations": len(citations),
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    avg_complete  = round(sum(all_complete)  / len(all_complete),  2) if all_complete  else 0
    avg_faith     = round(sum(all_faith)     / len(all_faith),     2) if all_faith     else 0
    avg_retrieval = round(sum(all_retrieval) / len(all_retrieval), 2) if all_retrieval else 0
    halluc_rate   = round(halluc_count / len(results), 2)             if results       else 0

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Questions tested:     {len(results)}")
    print(f"  Avg Completeness:     {avg_complete}  (target: >0.70)")
    print(f"  Avg Faithfulness:     {avg_faith}     (target: >0.75)")
    print(f"  Avg Retrieval@k:      {avg_retrieval} (target: >0.80)")
    print(f"  Hallucination Rate:   {halluc_rate:.0%} (target: <0.10)")
    print("=" * 60)

    # Save JSON report
    report = {
        "run_date":  datetime.now().isoformat(),
        "llm_model": "gemini-1.5-flash",
        "num_tests": len(results),
        "summary": {
            "avg_completeness":   avg_complete,
            "avg_faithfulness":   avg_faith,
            "avg_retrieval_at_k": avg_retrieval,
            "hallucination_rate": halluc_rate,
        },
        "results": results,
    }
    report_path = OUT_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {report_path}")

    _save_to_db(results)
    return report


def _save_to_db(results: list):
    if not DB_PATH.exists():
        print("Database not found — skipping DB save.")
        return
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rag_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT, answer TEXT, ticker TEXT,
            fiscal_year INTEGER, category TEXT,
            completeness REAL, faithfulness REAL,
            retrieval_at_k REAL, hallucination INTEGER,
            confidence REAL, num_citations INTEGER, run_date TEXT
        )
    """)
    for r in results:
        conn.execute("""
            INSERT INTO rag_evaluations
                (question,answer,ticker,fiscal_year,category,
                 completeness,faithfulness,retrieval_at_k,
                 hallucination,confidence,num_citations,run_date)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, [r["question"],r["answer"],r["ticker"],r["year"],r["category"],
              r["completeness"],r["faithfulness"],r["retrieval_k"],
              1 if r["hallucination"]["flag"] else 0,
              r["confidence"],r["num_citations"],datetime.now().isoformat()])
    conn.commit()
    conn.close()
    print("Evaluation results saved to database")


if __name__ == "__main__":
    run_evaluation()
