import os
import sys
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
judge_model = genai.GenerativeModel("gemini-1.5-flash")

# project root imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.qa.qa_pipeline import ask

DB_PATH = Path("data/sql/finsightai.db")
OUT_DIR = Path("data/eval/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_FILE = Path("data/eval/evaluation_queries.json")


def load_eval_questions():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def score_completeness(answer: str, keywords: list) -> float:
    answer_lower = answer.lower()
    found = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(found / len(keywords), 2) if keywords else 0.0


def score_hallucination(answer: str, citations: list) -> dict:
    answer_lower = answer.lower()
    no_info_phrases = [
        "do not address",
        "not mentioned",
        "no information",
        "not provided",
        "not found in"
    ]

    if any(p in answer_lower for p in no_info_phrases):
        return {"flag": False, "reason": "System correctly acknowledged missing info"}

    if not citations:
        return {"flag": True, "reason": "No source chunks were retrieved"}

    if "[source" not in answer_lower and "source 1" not in answer_lower:
        return {"flag": True, "reason": "Answer not grounded — no [Source N] citations found"}

    return {"flag": False, "reason": "No hallucination signals detected"}


def score_faithfulness_gemini(question: str, answer: str, context: str) -> float:
    prompt = f"""You are evaluating an AI system for faithfulness to source documents.

Question: {question}

Source document excerpts:
{context[:1500]}

Generated answer:
{answer[:600]}

Rate the answer's FAITHFULNESS on a scale of 0-10:
10 = Every claim directly supported by documents
7-9 = Most claims supported, minor liberties
4-6 = Some claims supported, some questionable
1-3 = Significant unsupported claims
0 = Answer contradicts or ignores documents

Respond with ONE number only (0-10). No explanation.
"""

    try:
        response = judge_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0,
                max_output_tokens=5
            )
        )
        return round(float(response.text.strip()) / 10.0, 2)
    except Exception as e:
        print(f"[Gemini faithfulness scoring failed: {e}]")
        return -1.0


def run_qa_evaluation():
    tests = load_eval_questions()

    print("=" * 60)
    print("QA Evaluation")
    print(f"Running {len(tests)} evaluation questions...")
    print("=" * 60)

    results = []
    all_complete = []
    all_faith = []
    halluc_count = 0

    for test in tests:
        print(f"\n[{test['id']}] {test['question']}")

        result = ask(
            test["question"],
            ticker=test.get("ticker"),
            year=test.get("year"),
            include_followups=False
        )

        answer = result.get("answer", "")
        citations = result.get("citations", [])
        confidence = result.get("confidence", 0.0)
        sections = result.get("sections_used", [])

        context = " ".join(c.get("preview", c.get("text", "")) for c in citations[:3])

        completeness = score_completeness(answer, test.get("keywords", []))
        hallucination = score_hallucination(answer, citations)
        faithfulness = score_faithfulness_gemini(test["question"], answer, context)

        all_complete.append(completeness)
        if faithfulness >= 0:
            all_faith.append(faithfulness)
        if hallucination["flag"]:
            halluc_count += 1

        row = {
            "id": test["id"],
            "question": test["question"],
            "ticker": test.get("ticker"),
            "year": test.get("year"),
            "category": test.get("category"),
            "answer": answer,
            "completeness": completeness,
            "faithfulness": faithfulness,
            "hallucination": hallucination,
            "confidence": confidence,
            "sections_used": sections,
            "num_citations": len(citations)
        }
        results.append(row)

        print(f"  Completeness:  {completeness:.2f}")
        print(f"  Faithfulness:  {faithfulness:.2f}" if faithfulness >= 0 else "  Faithfulness: N/A")
        print(f"  Hallucination: {'YES' if hallucination['flag'] else 'No'}")
        print(f"  Confidence:    {confidence:.2f}")

    avg_complete = round(sum(all_complete) / len(all_complete), 2) if all_complete else 0.0
    avg_faith = round(sum(all_faith) / len(all_faith), 2) if all_faith else 0.0
    halluc_rate = round(halluc_count / len(results), 2) if results else 0.0

    summary = {
        "num_tests": len(results),
        "avg_completeness": avg_complete,
        "avg_faithfulness": avg_faith,
        "hallucination_rate": halluc_rate
    }

    report = {
        "run_date": datetime.now().isoformat(),
        "evaluation_type": "qa",
        "summary": summary,
        "results": results
    }

    out_path = OUT_DIR / "qa_evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("QA EVALUATION SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_path}")

    _save_qa_to_db(results)
    return report


def _save_qa_to_db(results: list):
    if not DB_PATH.exists():
        print("Database not found — skipping DB save.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS qa_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            ticker TEXT,
            fiscal_year INTEGER,
            category TEXT,
            completeness REAL,
            faithfulness REAL,
            hallucination INTEGER,
            confidence REAL,
            num_citations INTEGER,
            run_date TEXT
        )
    """)

    for r in results:
        conn.execute("""
            INSERT INTO qa_evaluations (
                question, answer, ticker, fiscal_year, category,
                completeness, faithfulness, hallucination,
                confidence, num_citations, run_date
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["question"],
            r["answer"],
            r["ticker"],
            r["year"],
            r["category"],
            r["completeness"],
            r["faithfulness"],
            1 if r["hallucination"]["flag"] else 0,
            r["confidence"],
            r["num_citations"],
            datetime.now().isoformat()
        ))

    conn.commit()
    conn.close()
    print("QA evaluation results saved to database.")


if __name__ == "__main__":
    run_qa_evaluation()