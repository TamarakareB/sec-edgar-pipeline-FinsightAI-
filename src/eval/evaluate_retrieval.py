import sys
import json
from pathlib import Path
from datetime import datetime

# Make project root importable
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
    """
    Try several metadata fields to recover filing year.
    """
    for key in ["year", "filing_year"]:
        if key in chunk and chunk[key] is not None:
            value = str(chunk[key]).strip()

            # handle 2023.0 -> 2023
            if value.endswith(".0"):
                value = value[:-2]

            # if raw year string starts with YYYY, keep first 4 chars
            if len(value) >= 4 and value[:4].isdigit():
                return value[:4]

            if value.isdigit():
                return value

    filing_date = chunk.get("filing_date")
    if filing_date:
        filing_date = str(filing_date).strip()
        if len(filing_date) >= 4 and filing_date[:4].isdigit():
            return filing_date[:4]

    return ""


def get_section_text(chunk):
    """
    Normalize whichever section field exists in your chunk schema.
    """
    return normalize_text(
        chunk.get("section_hint")
        or chunk.get("section")
        or chunk.get("section_name")
        or ""
    )


def get_chunk_text(chunk):
    """
    Try likely text fields used in your chunk schema.
    """
    return normalize_text(
        chunk.get("text")
        or chunk.get("content")
        or chunk.get("chunk_text")
        or ""
    )


def match_signals(chunk, test):
    """
    Compare one retrieved chunk with one eval query.
    Returns diagnostic booleans for debugging.
    """
    chunk_ticker = normalize_text(chunk.get("ticker"))
    test_ticker = normalize_text(test.get("ticker"))
    ticker_match = chunk_ticker == test_ticker if test_ticker else False

    chunk_year = extract_year(chunk)
    test_year = str(test.get("year")).strip() if test.get("year") is not None else ""
    year_match = chunk_year == test_year if test_year else False

    expected_sections = test.get("expected_sections", [])
    section_text = get_section_text(chunk)
    section_match = any(
        normalize_text(sec) in section_text
        for sec in expected_sections
    ) if expected_sections else False

    keywords = test.get("keywords", [])
    chunk_text = get_chunk_text(chunk)
    keyword_match = any(
        normalize_text(kw) in chunk_text
        for kw in keywords
    ) if keywords else False

    task = test.get("task", "")

    # Task-aware strict match
    if task == "qa_fact":
        # For fact queries, keyword evidence is more reliable than section metadata
        strict_match = ticker_match and year_match and keyword_match
    elif task == "qa_risk":
        # For risk queries, either section or textual risk evidence is acceptable
        strict_match = ticker_match and year_match and (section_match or keyword_match)
    else:
        strict_match = ticker_match and year_match and (section_match or keyword_match)

    # Soft match only checks entity/time filtering
    soft_match = ticker_match and year_match

    return {
        "ticker_match": ticker_match,
        "year_match": year_match,
        "section_match": section_match,
        "keyword_match": keyword_match,
        "strict_match": strict_match,
        "soft_match": soft_match,
        "chunk_ticker": chunk_ticker,
        "chunk_year": chunk_year,
        "section_text": section_text,
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


def keyword_hit(results, test, k=5):
    for chunk in results[:k]:
        if match_signals(chunk, test)["keyword_match"]:
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
            "filing_date": chunk.get("filing_date"),
            "section": (
                chunk.get("section_hint")
                or chunk.get("section")
                or chunk.get("section_name")
                or ""
            ),
            "distance": chunk.get("distance"),
            "ticker_match": m["ticker_match"],
            "year_match": m["year_match"],
            "section_match": m["section_match"],
            "keyword_match": m["keyword_match"],
            "strict_match": m["strict_match"],
            "soft_match": m["soft_match"],
            "text_preview": (
                chunk.get("text")
                or chunk.get("content")
                or chunk.get("chunk_text")
                or ""
            )[:250]
        })

    return summary


def print_debug_for_test(test, retrieved, top_n=3):
    print("\nDEBUG TEST")
    print("-" * 50)
    print("ID:                ", test.get("id"))
    print("QUESTION:          ", test.get("question"))
    print("EXPECTED TICKER:   ", test.get("ticker"))
    print("EXPECTED YEAR:     ", test.get("year"))
    print("EXPECTED SECTIONS: ", test.get("expected_sections"))
    print("KEYWORDS:          ", test.get("keywords"))
    print("TASK:              ", test.get("task"))

    if not retrieved:
        print("\nNo retrieved chunks returned.")
        return

    for i, chunk in enumerate(retrieved[:top_n], start=1):
        m = match_signals(chunk, test)

        print(f"\n--- Top {i} ---")
        print("chunk_id:       ", chunk.get("chunk_id"))
        print("ticker(raw):    ", chunk.get("ticker"))
        print("year(raw):      ", chunk.get("year"))
        print("filing_year:    ", chunk.get("filing_year"))
        print("filing_date:    ", chunk.get("filing_date"))
        print("section_hint:   ", chunk.get("section_hint"))
        print("section:        ", chunk.get("section"))
        print("section_name:   ", chunk.get("section_name"))
        print("extract_year:   ", extract_year(chunk))
        print("section_text:   ", m["section_text"])
        print("match_signals:  ", {
            "ticker_match": m["ticker_match"],
            "year_match": m["year_match"],
            "section_match": m["section_match"],
            "keyword_match": m["keyword_match"],
            "strict_match": m["strict_match"],
            "soft_match": m["soft_match"],
        })


def run_retrieval_evaluation(k=5, max_tests=None, debug=False):
    tests = load_eval_questions()

    if max_tests is not None:
        tests = tests[:max_tests]

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
    keyword_hits = []

    for test in tests:
        print(f"\n[{test['id']}] {test['question']}")

        try:
            retrieved = search(
                query=test["question"],
                top_k=k,
                ticker=test.get("ticker"),
                year=test.get("year")
            )
        except Exception as e:
            print(f"  Search error: {e}")
            retrieved = []

        if debug:
            print_debug_for_test(test, retrieved, top_n=3)

        strict_r = recall_at_k(retrieved, test, k=k, strict=True)
        soft_r = recall_at_k(retrieved, test, k=k, strict=False)
        strict_mrr = reciprocal_rank(retrieved, test, strict=True)
        soft_mrr = reciprocal_rank(retrieved, test, strict=False)
        sec_hit = section_hit(retrieved, test, k=k)
        kw_hit = keyword_hit(retrieved, test, k=k)

        strict_recalls.append(strict_r)
        soft_recalls.append(soft_r)
        strict_rrs.append(strict_mrr)
        soft_rrs.append(soft_mrr)
        section_hits.append(sec_hit)
        keyword_hits.append(kw_hit)

        row = {
            "id": test["id"],
            "question": test["question"],
            "ticker": test.get("ticker"),
            "company": test.get("company"),
            "year": test.get("year"),
            "task": test.get("task"),
            "expected_sections": test.get("expected_sections", []),
            "keywords": test.get("keywords", []),
            "strict_recall_at_k": strict_r,
            "soft_recall_at_k": soft_r,
            "strict_rr": round(strict_mrr, 3),
            "soft_rr": round(soft_mrr, 3),
            "section_hit": sec_hit,
            "keyword_hit": kw_hit,
            "num_retrieved": len(retrieved),
            "top_results": summarize_top_results(retrieved, test, k=k)
        }
        results_rows.append(row)

        print(f"  Retrieved:         {len(retrieved)}")
        print(f"  Strict Recall@{k}: {strict_r}")
        print(f"  Soft Recall@{k}:   {soft_r}")
        print(f"  Strict RR:         {strict_mrr:.3f}")
        print(f"  Soft RR:           {soft_mrr:.3f}")
        print(f"  Section hit:       {sec_hit}")
        print(f"  Keyword hit:       {kw_hit}")

    summary = {
        "num_tests": len(results_rows),
        f"strict_recall_at_{k}": round(sum(strict_recalls) / len(strict_recalls), 3) if strict_recalls else 0.0,
        f"soft_recall_at_{k}": round(sum(soft_recalls) / len(soft_recalls), 3) if soft_recalls else 0.0,
        "strict_mrr": round(sum(strict_rrs) / len(strict_rrs), 3) if strict_rrs else 0.0,
        "soft_mrr": round(sum(soft_rrs) / len(soft_rrs), 3) if soft_rrs else 0.0,
        "section_hit_rate": round(sum(section_hits) / len(section_hits), 3) if section_hits else 0.0,
        "keyword_hit_rate": round(sum(keyword_hits) / len(keyword_hits), 3) if keyword_hits else 0.0
    }

    report = {
        "run_date": datetime.now().isoformat(),
        "evaluation_type": "retrieval",
        "k": k,
        "summary": summary,
        "results": results_rows
    }

    out_path = OUT_DIR / "retrieval_evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_path}")

    return report


if __name__ == "__main__":
    # First debug run:
    # run_retrieval_evaluation(k=5, max_tests=1, debug=True)

    # Full run:
    run_retrieval_evaluation(k=5, max_tests=None, debug=False)