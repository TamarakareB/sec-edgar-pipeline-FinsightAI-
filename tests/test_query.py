from src.rag.qa_gemini import answer_question
from src.rag.risk_classifier import classify_risk

chunks = [
    {
        "chunk_id": "c1",
        "ticker": "AAPL",
        "filing_date": "2023",
        "text": "Apple faces regulatory risks related to antitrust investigations in the EU."
    },
    {
        "chunk_id": "c2",
        "ticker": "AAPL",
        "filing_date": "2023",
        "text": "The company depends on global supply chains which may cause operational risks."
    }
]

query = "What risks does Apple face?"

qa = answer_question(query, chunks)

risk = classify_risk(query, chunks)

import json

print("QA RESULT")
print(json.dumps(qa, indent=2))

print("\nRISK LABELS")
print(json.dumps(risk, indent=2))