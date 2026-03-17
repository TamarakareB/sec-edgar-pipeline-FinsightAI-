import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.retrieve import search

OUT_DIR = Path("data/eval/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_FILE = Path("data/eval/evaluation_queries.json")


def load_eval_questions():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(x):
    return str(x).strip().lower() if x is not None else ""


def extract_year(chunk):
    for key in ["year", "filing_year"]:
        if key in chunk and chunk[key] is not None:
            return str(chunk[key])

    filing_date = chunk.get("filing_date")
    if filing_date:
        return str(filing_date)[:4]

    return ""


def get_section_text(chunk):
    return normalize_text(
        chunk.get("section_hint")
        or chunk.get("section")
        or chunk.get("section_name")
        or ""
    )


def match_signals(chunk, test):
    ticker_match = normalize_text(chunk.get("ticker")) == normalize_text(test.get("ticker"))
    year_match = extract_year(chunk) == str(test.get("year"))

    expected_sections = test.get("expected_sections", [])
    section_text = get_section_text(chunk)

    section_match = any(
        normalize_text(sec) in section_text
        for sec in expected_sections
    ) if expected_sections else False

    return {
        "ticker_match": ticker_match,
        "year_match": year_match,
        "section_match": section_match,
        "strict_match": ticker_match and year_match and section_match,
        "soft_match": ticker_match and year_match
    }


def recall_at_k(results, test, k=5, strict=True):
    topk = results[:k]
    for chunk in topk:
        m = match_signals(chunk, test)
        if strict and m["strict_match"]:
            return 1
        if not strict and m["soft_match"]:
            return 1
    return 0


def reciprocal_rank(results, test, strict=True):
    for i, chunk in enumerate(results, start=1):
        m = match_signals(chunk, test)
        if strict and m["strict_match"]:
            return 1.0 / i
        if not strict and m["soft_match"]:
            return 1.0 / i
    return 0.0


def section_hit(results, test, k=5):
    for chunk in results[:k]:
        if match_signals(chunk, test)["section_match"]:
            return 1
    return 0


def summarize_top_results(results, test, k=5):
    summary = []
    for idx, chunk in enumerate(results[:k], start=1):
        m = match_signals(chunk, test)
        summary.append({
            "rank": idx,
            "chunk_id": chunk.get("chunk_id"),
            "ticker": chunk.get("ticker"),
            "year": extract_year(chunk),
            "section": chunk.get("section_hint") or chunk.get("section") or chunk.get("section_name") or "",
            "distance": chunk.get("distance"),
            "ticker_match": m["ticker_match"],
            "year_match": m["year_match"],
            "section_match": m["section_match"],
            "text_preview": str(chunk.get("text", ""))[:200]
        })
    return summary


def run_retrieval_evaluation(k=5):
    tests = load_eval_questions()

    print("=" * 60)
    print("Retrieval Evaluation")
    print(f"Running {len(tests)} evaluation questions...")
    print("=" * 60)

    results_rows = []
    strict_recalls = []
    soft_recalls = []
    strict_rrs = []
    soft_rrs = []
    section_hits = []

    for test in tests:
        print(f"\n[{test['id']}] {test['question']}")

        retrieved = search(
            query=test["question"],
            top_k=k,
            ticker=test.get("ticker"),
            year=test.get("year")
        )

        strict_r = recall_at_k(retrieved, test, k=k, strict=True)
        soft_r = recall_at_k(retrieved, test, k=k, strict=False)
        strict_mrr = reciprocal_rank(retrieved, test, strict=True)
        soft_mrr = reciprocal_rank(retrieved, test, strict=False)
        sec_hit = section_hit(retrieved, test, k=k)

        strict_recalls.append(strict_r)
        soft_recalls.append(soft_r)
        strict_rrs.append(strict_mrr)
        soft_rrs.append(soft_mrr)
        section_hits.append(sec_hit)

        row = {
            "id": test["id"],
            "question": test["question"],
            "ticker": test.get("ticker"),
            "year": test.get("year"),
            "task": test.get("task"),
            "expected_sections": test.get("expected_sections", []),
            "strict_recall_at_k": strict_r,
            "soft_recall_at_k": soft_r,
            "strict_rr": round(strict_mrr, 3),
            "soft_rr": round(soft_mrr, 3),
            "section_hit": sec_hit,
            "top_results": summarize_top_results(retrieved, test, k=k)
        }
        results_rows.append(row)

        print(f"  Strict Recall@{k}: {strict_r}")
        print(f"  Soft Recall@{k}:   {soft_r}")
        print(f"  Strict RR:         {strict_mrr:.3f}")
        print(f"  Soft RR:           {soft_mrr:.3f}")
        print(f"  Section hit:       {sec_hit}")

    summary = {
        "num_tests": len(results_rows),
        f"strict_recall_at_{k}": round(sum(strict_recalls) / len(strict_recalls), 2) if strict_recalls else 0.0,
        f"soft_recall_at_{k}": round(sum(soft_recalls) / len(soft_recalls), 2) if soft_recalls else 0.0,
        "strict_mrr": round(sum(strict_rrs) / len(strict_rrs), 2) if strict_rrs else 0.0,
        "soft_mrr": round(sum(soft_rrs) / len(soft_rrs), 2) if soft_rrs else 0.0,
        "section_hit_rate": round(sum(section_hits) / len(section_hits), 2) if section_hits else 0.0
    }

    report = {
        "run_date": datetime.now().isoformat(),
        "evaluation_type": "retrieval",
        "summary": summary,
        "results": results_rows
    }

    out_path = OUT_DIR / "retrieval_evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_path}")

    return report


if __name__ == "__main__":
    run_retrieval_evaluation()