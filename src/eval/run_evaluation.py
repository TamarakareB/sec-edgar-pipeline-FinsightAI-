import json
from pathlib import Path
from datetime import datetime

from evaluate_retrieval import run_retrieval_evaluation
from evaluate_qa import run_qa_evaluation
from evaluate_risk import run_risk_evaluation

OUT_DIR = Path("data/eval/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_all_evaluations():
    retrieval_report = run_retrieval_evaluation(k=5)
    qa_report = run_qa_evaluation()
    risk_report = run_risk_evaluation(k=5)

    final_report = {
        "run_date": datetime.now().isoformat(),
        "project": "FinSightAI",
        "reports": {
            "retrieval": retrieval_report.get("summary", {}),
            "qa": qa_report.get("summary", {}),
            "risk": risk_report.get("summary", {})
        }
    }

    out_path = OUT_DIR / "full_evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\n" + "=" * 60)
    print("FULL EVALUATION COMPLETE")
    print("=" * 60)
    print(json.dumps(final_report, indent=2))
    print(f"Saved to {out_path}")

    return final_report


if __name__ == "__main__":
    run_all_evaluations()