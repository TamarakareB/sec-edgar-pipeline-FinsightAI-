import sys
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.retrieve import search
from src.ana.risk_classifier import classify_risk


OUT_DIR = Path("data/eval/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_FILE = Path("data/eval/evaluation_queries.json")


LABEL_ALIASES = {
    "liquidity": "liquidity_risk",
    "liquidity risk": "liquidity_risk",
    "funding risk": "liquidity_risk",

    "market": "market_risk",
    "market risk": "market_risk",

    "regulation": "regulatory_risk",
    "regulatory": "regulatory_risk",
    "regulatory risk": "regulatory_risk",

    "operations": "operational_risk",
    "operational": "operational_risk",
    "operational risk": "operational_risk",

    "competition": "competitive_risk",
    "competitive": "competitive_risk",
    "competitive risk": "competitive_risk",

    "supply chain": "supply_chain_risk",
    "supply chain risk": "supply_chain_risk",

    "cybersecurity": "cybersecurity_risk",
    "cyber risk": "cybersecurity_risk",
    "cybersecurity risk": "cybersecurity_risk",
}


def load_eval_questions():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(x):
    return str(x).strip().lower() if x is not None else ""


def normalize_label(label: str) -> str:
    label_norm = normalize_text(label)
    return LABEL_ALIASES.get(label_norm, label_norm)


def normalize_labels(labels: list) -> list:
    normalized = []
    for label in labels:
        if label is None:
            continue
        normalized.append(normalize_label(label))
    return sorted(set(normalized))


def precision_recall_f1(predicted_labels, expected_labels):
    predicted = set(normalize_labels(predicted_labels))
    expected = set(normalize_labels(expected_labels))

    if not predicted and not expected:
        return 1.0, 1.0, 1.0
    if not predicted and expected:
        return 0.0, 0.0, 0.0
    if predicted and not expected:
        return 0.0, 0.0, 0.0

    correct = len(predicted & expected)
    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / len(expected) if expected else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return round(precision, 2), round(recall, 2), round(f1, 2)


def overlap_hit(predicted_labels, expected_labels):
    predicted = set(normalize_labels(predicted_labels))
    expected = set(normalize_labels(expected_labels))
    return 1 if len(predicted & expected) > 0 else 0


def strict_correct_from_f1(f1: float, threshold: float = 0.5):
    return 1 if f1 >= threshold else 0


def summarize_top_results(results, k=3):
    rows = []
    for idx, chunk in enumerate(results[:k], start=1):
        rows.append({
            "rank": idx,
            "chunk_id": chunk.get("chunk_id"),
            "ticker": chunk.get("ticker"),
            "year": chunk.get("year"),
            "filing_date": chunk.get("filing_date"),
            "section": (
                chunk.get("section_hint")
                or chunk.get("section")
                or chunk.get("section_name")
                or ""
            ),
            "distance": chunk.get("distance"),
            "text_preview": (
                chunk.get("text")
                or chunk.get("content")
                or chunk.get("chunk_text")
                or ""
            )[:250]
        })
    return rows


def print_debug_for_test(test, retrieved, raw_clf_output, predicted_labels, expected_labels):
    print("\nDEBUG TEST")
    print("-" * 50)
    print("ID:                 ", test.get("id"))
    print("QUESTION:           ", test.get("question"))
    print("TASK:               ", test.get("task"))
    print("EXPECTED LABELS RAW:", expected_labels)
    print("EXPECTED NORMALIZED:", normalize_labels(expected_labels))
    print("PREDICTED RAW:      ", predicted_labels)
    print("PREDICTED NORMALIZED:", normalize_labels(predicted_labels))
    print("NUM RETRIEVED:      ", len(retrieved))
    print("RAW CLASSIFIER OUT: ", raw_clf_output)

    for i, chunk in enumerate(retrieved[:3], start=1):
        print(f"\n--- Top {i} ---")
        print("chunk_id:     ", chunk.get("chunk_id"))
        print("ticker:       ", chunk.get("ticker"))
        print("year:         ", chunk.get("year"))
        print("filing_date:  ", chunk.get("filing_date"))
        print("section:      ", chunk.get("section_hint") or chunk.get("section") or chunk.get("section_name"))
        print("text_preview: ", (chunk.get("text") or chunk.get("content") or chunk.get("chunk_text") or "")[:200])


def run_risk_evaluation(k=5, max_tests=None, debug=False):
    tests = load_eval_questions()

    # only use tests that actually define risk labels
    tests = [t for t in tests if "expected_labels" in t]

    if max_tests is not None:
        tests = tests[:max_tests]

    print("=" * 60)
    print("Risk Classification Evaluation")
    print(f"Running {len(tests)} evaluation questions...")
    print("=" * 60)

    results = []
    precisions = []
    recalls = []
    f1s = []
    overlap_hits = []
    strict_corrects = []

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

        try:
            clf_output = classify_risk(test["question"], retrieved[:k])
        except Exception as e:
            print(f"  Risk classifier error: {e}")
            clf_output = {"risk_labels": [], "error": str(e)}

        predicted_labels = [
            x.get("label")
            for x in clf_output.get("risk_labels", [])
            if isinstance(x, dict) and x.get("label")
        ]
        expected_labels = test.get("expected_labels", [])

        precision, recall, f1 = precision_recall_f1(predicted_labels, expected_labels)
        hit = overlap_hit(predicted_labels, expected_labels)
        strict_correct = strict_correct_from_f1(f1, threshold=0.5)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        overlap_hits.append(hit)
        strict_corrects.append(strict_correct)

        row = {
            "id": test["id"],
            "question": test["question"],
            "ticker": test.get("ticker"),
            "year": test.get("year"),
            "task": test.get("task"),
            "expected_labels_raw": expected_labels,
            "expected_labels_normalized": normalize_labels(expected_labels),
            "predicted_labels_raw": predicted_labels,
            "predicted_labels_normalized": normalize_labels(predicted_labels),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "overlap_hit": hit,
            "strict_correct": strict_correct,
            "num_retrieved": len(retrieved),
            "top_results": summarize_top_results(retrieved, k=3)
        }
        results.append(row)

        print(f"  Expected raw:        {expected_labels}")
        print(f"  Expected normalized: {normalize_labels(expected_labels)}")
        print(f"  Predicted raw:       {predicted_labels}")
        print(f"  Predicted normalized:{normalize_labels(predicted_labels)}")
        print(f"  Precision:           {precision:.2f}")
        print(f"  Recall:              {recall:.2f}")
        print(f"  F1:                  {f1:.2f}")
        print(f"  Overlap hit:         {hit}")
        print(f"  Strict correct:      {strict_correct}")

        if debug:
            print_debug_for_test(test, retrieved, clf_output, predicted_labels, expected_labels)

    summary = {
        "num_tests": len(results),
        "avg_precision": round(sum(precisions) / len(precisions), 3) if precisions else 0.0,
        "avg_recall": round(sum(recalls) / len(recalls), 3) if recalls else 0.0,
        "avg_f1": round(sum(f1s) / len(f1s), 3) if f1s else 0.0,
        "overlap_hit_rate": round(sum(overlap_hits) / len(overlap_hits), 3) if overlap_hits else 0.0,
        "strict_accuracy": round(sum(strict_corrects) / len(strict_corrects), 3) if strict_corrects else 0.0
    }

    report = {
        "run_date": datetime.now().isoformat(),
        "evaluation_type": "risk_classification",
        "k": k,
        "summary": summary,
        "results": results
    }

    out_path = OUT_DIR / "risk_evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("RISK EVALUATION SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_path}")

    return report


if __name__ == "__main__":
    # First debug run:
    #run_risk_evaluation(k=5, max_tests=1, debug=True)

    # Full run:
    run_risk_evaluation(k=5, max_tests=None, debug=False)