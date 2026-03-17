from retrieve import search

query = "What risks does Apple mention about supply chain?"

results = search(query)

for i, r in enumerate(results):
    print("\n" + "="*60)
    print("RESULT", i + 1)
    print("Chunk ID:", r["chunk_id"])
    print("Ticker:", r["ticker"])
    print("Form:", r["form"])
    print("Date:", r["filing_date"])
    print("Source:", r["source_url"])
    print("Text:", r["text"][:400])