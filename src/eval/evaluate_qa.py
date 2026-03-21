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
judge_model = genai.GenerativeModel("gemini-2.5-flash")

# project root imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.qa_pipeline import ask


DB_PATH = Path("data/sql/finsightai.db")
OUT_DIR = Path("data/eval/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_FILE = Path("data/eval/evaluation_queries.json")


def load_eval_questions():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(x):
    return str(x).strip().lower() if x is not None else ""


def get_answer_text(result: dict) -> str:
    return str(result.get("answer", "")).strip()


def get_citations(result: dict) -> list:
    citations = result.get("citations", [])
    return citations if isinstance(citations, list) else []


def score_answer_presence(answer: str) -> int:
    return 1 if answer and len(answer.strip()) >= 20 else 0


def get_matched_keywords(answer: str, keywords: list) -> list:
    answer_lower = normalize_text(answer)
    matched = []

    for kw in keywords:
        kw_norm = normalize_text(kw)
        if kw_norm and kw_norm in answer_lower:
            matched.append(kw)

    return matched


def score_keyword_coverage(answer: str, keywords: list) -> tuple[list, float]:
    if not keywords:
        return [], 0.0

    matched = get_matched_keywords(answer, keywords)
    score = round(len(matched) / len(keywords), 2)
    return matched, score


def detect_citation_presence(answer: str, citations: list) -> dict:
    answer_lower = normalize_text(answer)

    has_inline_source = (
        "[source" in answer_lower
        or "source 1" in answer_lower
        or "source 2" in answer_lower
    )

    has_retrieved_citations = len(citations) > 0

    citation_present = 1 if (has_inline_source or has_retrieved_citations) else 0

    return {
        "citation_present": citation_present,
        "has_inline_source": has_inline_source,
        "has_retrieved_citations": has_retrieved_citations
    }


def score_grounding_issue(answer: str, citations: list) -> dict:
    """
    Do NOT label every missing citation as hallucination.
    Separate grounding issues from actual unsupported answering.
    """
    answer_lower = normalize_text(answer)

    no_info_phrases = [
        "do not address",
        "not mentioned",
        "no information",
        "not provided",
        "not found in",
        "the provided filings do not address this"
    ]

    if any(p in answer_lower for p in no_info_phrases):
        return {
            "flag": False,
            "reason": "System correctly acknowledged missing information"
        }

    citation_info = detect_citation_presence(answer, citations)

    if not citations:
        return {
            "flag": True,
            "reason": "No retrieved citations available for grounding"
        }

    if citation_info["citation_present"] == 0:
        return {
            "flag": True,
            "reason": "Answer returned without visible grounding/citation signal"
        }

    return {
        "flag": False,
        "reason": "No grounding issue detected"
    }


def score_faithfulness_gemini(question: str, answer: str, context: str) -> float:
    if not answer.strip():
        return -1.0

    prompt = f"""You are evaluating an AI system for faithfulness to source documents.

Question: {question}

Source document excerpts:
{context[:1500]}

Generated answer:
{answer[:700]}

Rate the answer's FAITHFULNESS on a scale of 0-10:
10 = Every claim is directly supported by the excerpts
7-9 = Mostly supported, minor liberties
4-6 = Mixed support, some questionable claims
1-3 = Significant unsupported claims
0 = Contradicted or clearly unsupported

Respond with ONE number only.
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


def build_context_from_citations(citations: list, max_items: int = 3) -> str:
    pieces = []
    for c in citations[:max_items]:
        pieces.append(
            str(c.get("preview") or c.get("text") or c.get("content") or "")
        )
    return " ".join(pieces).strip()


def score_strict_correct(task: str, answer_present: int, citation_present: int, keyword_coverage: float) -> int:
    if task == "qa_fact":
        return 1 if (answer_present and citation_present and keyword_coverage >= 0.5) else 0

    if task == "qa_risk":
        return 1 if (answer_present and citation_present and keyword_coverage >= 0.3) else 0

    return 1 if (answer_present and citation_present and keyword_coverage >= 0.4) else 0


def print_debug_for_test(test: dict, result: dict, row: dict):
    print("\nDEBUG TEST")
    print("-" * 50)
    print("ID:               ", test.get("id"))
    print("QUESTION:         ", test.get("question"))
    print("TASK:             ", test.get("task"))
    print("EXPECTED TICKER:  ", test.get("ticker"))
    print("EXPECTED YEAR:    ", test.get("year"))
    print("KEYWORDS:         ", test.get("keywords"))
    print("\nANSWER:")
    print(row["answer"][:800] if row["answer"] else "[EMPTY ANSWER]")
    print("\nMATCHED KEYWORDS: ", row["matched_keywords"])
    print("KEYWORD COVERAGE: ", row["keyword_coverage"])
    print("ANSWER PRESENT:   ", row["answer_present"])
    print("CITATION PRESENT: ", row["citation_present"])
    print("NUM CITATIONS:    ", row["num_citations"])
    print("STRICT CORRECT:   ", row["strict_correct"])
    print("GROUNDING ISSUE:  ", row["grounding_issue"])


def run_qa_evaluation(max_tests=None, debug=False, use_gemini_judge=True):
    tests = load_eval_questions()

    # only evaluate QA tasks
    tests = [t for t in tests if str(t.get("task", "")).startswith("qa")]

    if max_tests is not None:
        tests = tests[:max_tests]

    print("=" * 60)
    print("QA Evaluation")
    print(f"Running {len(tests)} evaluation questions...")
    print("=" * 60)

    results = []

    answer_present_scores = []
    keyword_coverages = []
    strict_correct_scores = []
    grounding_issue_count = 0
    citation_present_count = 0
    faithfulness_scores = []

    for test in tests:
        print(f"\n[{test['id']}] {test['question']}")

        try:
            result = ask(
                test["question"],
                ticker=test.get("ticker"),
                year=test.get("year"),
                include_followups=False
            )
        except Exception as e:
            print(f"  QA pipeline error: {e}")
            result = {
                "answer": "",
                "citations": [],
                "confidence": 0.0,
                "sections_used": [],
                "error": str(e)
            }

        answer = get_answer_text(result)
        citations = get_citations(result)
        confidence = result.get("confidence", 0.0)
        sections = result.get("sections_used", [])

        answer_present = score_answer_presence(answer)
        matched_keywords, keyword_coverage = score_keyword_coverage(
            answer, test.get("keywords", [])
        )

        citation_info = detect_citation_presence(answer, citations)
        citation_present = citation_info["citation_present"]

        grounding_issue = score_grounding_issue(answer, citations)

        context = build_context_from_citations(citations, max_items=3)

        if use_gemini_judge:
            faithfulness = score_faithfulness_gemini(
                test["question"], answer, context
            )
        else:
            faithfulness = -1.0

        strict_correct = score_strict_correct(
            task=test.get("task", ""),
            answer_present=answer_present,
            citation_present=citation_present,
            keyword_coverage=keyword_coverage
        )

        row = {
            "id": test["id"],
            "question": test["question"],
            "ticker": test.get("ticker"),
            "year": test.get("year"),
            "task": test.get("task"),
            "answer": answer,
            "matched_keywords": matched_keywords,
            "keyword_coverage": keyword_coverage,
            "answer_present": answer_present,
            "citation_present": citation_present,
            "grounding_issue": grounding_issue,
            "faithfulness": faithfulness,
            "strict_correct": strict_correct,
            "confidence": confidence,
            "sections_used": sections,
            "num_citations": len(citations)
        }
        results.append(row)

        answer_present_scores.append(answer_present)
        keyword_coverages.append(keyword_coverage)
        strict_correct_scores.append(strict_correct)

        if grounding_issue["flag"]:
            grounding_issue_count += 1
        if citation_present:
            citation_present_count += 1
        if faithfulness >= 0:
            faithfulness_scores.append(faithfulness)

        print(f"  Answer present:   {answer_present}")
        print(f"  Keyword coverage: {keyword_coverage:.2f}")
        print(f"  Citation present: {citation_present}")
        print(f"  Strict correct:   {strict_correct}")
        print(f"  Faithfulness:     {faithfulness:.2f}" if faithfulness >= 0 else "  Faithfulness: N/A")
        print(f"  Grounding issue:  {'YES' if grounding_issue['flag'] else 'No'}")
        print(f"  Confidence:       {confidence:.2f}")

        if debug:
            print_debug_for_test(test, result, row)

    summary = {
        "num_tests": len(results),
        "answer_rate": round(sum(answer_present_scores) / len(answer_present_scores), 3) if answer_present_scores else 0.0,
        "avg_keyword_coverage": round(sum(keyword_coverages) / len(keyword_coverages), 3) if keyword_coverages else 0.0,
        "citation_rate": round(citation_present_count / len(results), 3) if results else 0.0,
        "strict_accuracy": round(sum(strict_correct_scores) / len(strict_correct_scores), 3) if strict_correct_scores else 0.0,
        "avg_faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 3) if faithfulness_scores else 0.0,
        "grounding_issue_rate": round(grounding_issue_count / len(results), 3) if results else 0.0
    }

    report = {
        "run_date": datetime.now().isoformat(),
        "evaluation_type": "qa",
        "summary": summary,
        "results": results
    }

    out_path = OUT_DIR / "qa_evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

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
            task TEXT,
            keyword_coverage REAL,
            answer_present INTEGER,
            citation_present INTEGER,
            strict_correct INTEGER,
            faithfulness REAL,
            grounding_issue INTEGER,
            confidence REAL,
            num_citations INTEGER,
            run_date TEXT
        )
    """)

    for r in results:
        conn.execute("""
            INSERT INTO qa_evaluations (
                question, answer, ticker, fiscal_year, task,
                keyword_coverage, answer_present, citation_present,
                strict_correct, faithfulness, grounding_issue,
                confidence, num_citations, run_date
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["question"],
            r["answer"],
            r["ticker"],
            r["year"],
            r["task"],
            r["keyword_coverage"],
            r["answer_present"],
            r["citation_present"],
            r["strict_correct"],
            r["faithfulness"],
            1 if r["grounding_issue"]["flag"] else 0,
            r["confidence"],
            r["num_citations"],
            datetime.now().isoformat()
        ))

    conn.commit()
    conn.close()
    print("QA evaluation results saved to database.")


if __name__ == "__main__":
    # First debug run:
    #run_qa_evaluation(max_tests=1, debug=True, use_gemini_judge=False)

    # Full run:
    run_qa_evaluation(max_tests=None, debug=False, use_gemini_judge=True)