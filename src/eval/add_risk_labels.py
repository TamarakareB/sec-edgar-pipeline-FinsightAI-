import json
from pathlib import Path

EVAL_FILE = Path("data/eval/evaluation_queries.json")

RISK_LABEL_MAP = {
    "AAPL": ["competitive_risk", "regulatory_risk", "operational_risk"],
    "AMZN": ["competitive_risk", "operational_risk", "regulatory_risk"],
    "CSCO": ["competitive_risk", "cybersecurity_risk", "operational_risk"],
    "CVX": ["market_risk", "regulatory_risk", "operational_risk"],
    "JNJ": ["regulatory_risk", "operational_risk", "competitive_risk"],
    "META": ["regulatory_risk", "competitive_risk", "cybersecurity_risk"],
    "NFLX": ["competitive_risk", "market_risk", "operational_risk"],
    "NVDA": ["supply_chain_risk", "competitive_risk", "market_risk"],
    "ORCL": ["competitive_risk", "cybersecurity_risk", "operational_risk"],
    "WMT": ["operational_risk", "supply_chain_risk", "competitive_risk"],
}

def main():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        queries = json.load(f)

    updated = 0
    for q in queries:
        if q.get("task") == "qa_risk":
            ticker = q.get("ticker")
            if ticker in RISK_LABEL_MAP:
                q["expected_labels"] = RISK_LABEL_MAP[ticker]
                updated += 1

    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)

    print(f"Updated {updated} risk queries with expected_labels.")

if __name__ == "__main__":
    main()