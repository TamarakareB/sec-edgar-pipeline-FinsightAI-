import os
import json
from google import genai
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-2.5-flash"

RISK_LABELS = [
    "Financial/Liquidity",
    "Regulatory/Legal",
    "Operational",
    "Market/Macro",
    "Cyber/Technology",
    "Competition",
    "Strategy",
    "Reputation/ESG"
]


def classify_risk(query, chunks):

    api_key = os.getenv("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)

    context = ""

    for ch in chunks:
        context += f"""
chunk_id: {ch['chunk_id']}
text: {ch['text']}
"""

    prompt = f"""
You are a financial risk classifier.

Choose relevant labels from:

{RISK_LABELS}

User question:
{query}

Evidence:
{context}

Return JSON:

{{
"risk_labels":[
{{"label":"Regulatory/Legal","chunk_id":"..."}}
]
}}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    text = response.text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(text)
        labels = parsed.get("risk_labels", [])

        if not labels:
            labels.append({
                "label": "Strategy",
                "chunk_id": chunks[0]["chunk_id"]
            })

        return {"risk_labels": labels[:3]}

    except:
        return {
            "risk_labels": [
                {
                    "label": "Strategy",
                    "chunk_id": chunks[0]["chunk_id"]
                }
            ]
        }