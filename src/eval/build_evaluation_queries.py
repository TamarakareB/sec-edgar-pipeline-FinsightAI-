import json
from pathlib import Path

OUT_PATH = Path("data/eval/evaluation_queries.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

COMPANIES = {
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "CSCO": "Cisco",
    "CVX": "Chevron",
    "JNJ": "Johnson & Johnson",
    "META": "Meta",
    "NFLX": "Netflix",
    "NVDA": "NVIDIA",
    "ORCL": "Oracle",
    "WMT": "Walmart",
}

YEARS = [2022, 2023]

QUERY_TEMPLATES = [
    {
        "task": "qa_fact",
        "question": "What was {company}'s total revenue in {year}?",
        "expected_sections": [
            "financial statements",
            "income statement",
            "results of operations",
            "management discussion"
        ],
        "keywords": [
            "revenue",
            "net sales",
            "total revenue"
        ]
    },
    {
        "task": "qa_fact",
        "question": "What was {company}'s net income in {year}?",
        "expected_sections": [
            "financial statements",
            "income statement",
            "results of operations",
            "management discussion"
        ],
        "keywords": [
            "net income",
            "net earnings",
            "profit"
        ]
    },
    {
        "task": "qa_risk",
        "question": "What were the main risk factors reported by {company} in {year}?",
        "expected_sections": [
            "risk factors",
            "item 1a",
            "management discussion"
        ],
        "keywords": [
            "risk",
            "uncertainty",
            "competition",
            "regulation",
            "operations",
            "market"
        ]
    }
]


def build_queries():
    queries = []
    qid = 1

    for ticker, company in COMPANIES.items():
        for year in YEARS:
            for template in QUERY_TEMPLATES:
                queries.append({
                    "id": f"q{qid:03d}",
                    "ticker": ticker,
                    "company": company,
                    "year": year,
                    "task": template["task"],
                    "question": template["question"].format(company=company, year=year),
                    "expected_sections": template["expected_sections"],
                    "keywords": template["keywords"],
                })
                qid += 1

    return queries


def main():
    queries = build_queries()

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(queries)} queries to {OUT_PATH}")

    by_ticker = {}
    for q in queries:
        by_ticker[q["ticker"]] = by_ticker.get(q["ticker"], 0) + 1

    print("\nQueries per ticker:")
    for ticker, count in sorted(by_ticker.items()):
        print(f"  {ticker}: {count}")

    years = sorted(set(q["year"] for q in queries))
    print(f"\nYears covered: {years}")

    tasks = {}
    for q in queries:
        tasks[q["task"]] = tasks.get(q["task"], 0) + 1

    print("\nQueries by task:")
    for task, count in sorted(tasks.items()):
        print(f"  {task}: {count}")


if __name__ == "__main__":
    main()