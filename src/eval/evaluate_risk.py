import sys
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.retrieve import search
from src.qa.risk_classifier import classify_risk

OUT_DIR = Path("data/eval/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_FILE = Path("data/eval/evaluation_queries.json")


def load_eval_questions():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def precision_recall_f1(predicted_labels, expected_labels):
    predicted = set(predicted_labels)
    expected = set(expected_labels)

    if not predicted and not expected:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 0.0, 0.0

    correct = len(predicted & expected)
    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / len(expected) if expected else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return round(precision, 2), round(recall, 2), round(f1, 2)


def run_risk_evaluation(k=5):
    tests = load_eval_questions()

    print("=" * 60)
    print("Risk Classification Evaluation")
    print(f"Running {len(tests)} evaluation questions...")
    print("=" * 60)

    results = []
    precisions = []
    recalls = []
    f1s = []

    for test in tests:
        if "expected_labels" not in test:
            continue

        print(f"\n[{test['id']}] {test['question']}")

        retrieved = search(
            query=test["question"],
            top_k=k,
            ticker=test.get("ticker"),
            year=test.get("year")
        )

        clf_output = classify_risk(test["question"], retrieved[:k])
        predicted_labels = [x["label"] for x in clf_output.get("risk_labels", [])]
        expected_labels = test.get("expected_labels", [])

        precision, recall, f1 = precision_recall_f1(predicted_labels, expected_labels)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        row = {
            "id": test["id"],
            "question": test["question"],
            "expected_labels": expected_labels,
            "predicted_labels": predicted_labels,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        results.append(row)

        print(f"  Expected:  {expected_labels}")
        print(f"  Predicted: {predicted_labels}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall:    {recall:.2f}")
        print(f"  F1:        {f1:.2f}")

    summary = {
        "num_tests": len(results),
        "avg_precision": round(sum(precisions) / len(precisions), 2) if precisions else 0.0,
        "avg_recall": round(sum(recalls) / len(recalls), 2) if recalls else 0.0,
        "avg_f1": round(sum(f1s) / len(f1s), 2) if f1s else 0.0
    }

    report = {
        "run_date": datetime.now().isoformat(),
        "evaluation_type": "risk_classification",
        "summary": summary,
        "results": results
    }

    out_path = OUT_DIR / "risk_evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("RISK EVALUATION SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_path}")

    return report


if __name__ == "__main__":
    run_risk_evaluation()