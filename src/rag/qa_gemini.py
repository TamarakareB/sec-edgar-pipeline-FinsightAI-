import os
import json
from google import genai
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-2.5-flash"


def format_chunks(chunks):
    context = ""
    for i, ch in enumerate(chunks):
        context += f"""
Chunk {i}
chunk_id: {ch['chunk_id']}
ticker: {ch['ticker']}
filing_date: {ch['filing_date']}

{ch['text']}
"""
    return context


def answer_question(query, chunks):

    api_key = os.getenv("GEMINI_API_KEY")
    print("API KEY LOADED:", api_key[:10] if api_key else None)

    client = genai.Client(api_key=api_key)

    context = format_chunks(chunks)

    prompt = f"""
You are a financial filings assistant.

Answer the user's question ONLY using the evidence.

If evidence is insufficient say:
"Insufficient evidence"

Question:
{query}

Evidence:
{context}

Return JSON:

{{
"answer": "...",
"citations": ["chunk_id"]
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
        return parsed
    except:
        return {
            "answer": text,
            "citations": []
        }