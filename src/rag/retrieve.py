import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("data/index")

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(str(INDEX_DIR / "faiss_index.bin"))

with open(INDEX_DIR / "metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


def _extract_year(record):
    # try common field names
    for key in ["year", "filing_year"]:
        if key in record and record[key] is not None:
            return str(record[key])

    filing_date = record.get("filing_date", "")
    if filing_date:
        return str(filing_date)[:4]

    return None


def search(query, top_k=5, ticker=None, year=None):
    q_embedding = model.encode([query])
    distances, indices = index.search(q_embedding, top_k * 5)  # over-retrieve for filtering

    results = []

    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx < 0:
            continue

        item = dict(metadata[idx])  # copy
        item["rank"] = rank
        item["distance"] = float(dist)

        # optional filtering
        if ticker is not None:
            item_ticker = str(item.get("ticker", "")).upper()
            if item_ticker != str(ticker).upper():
                continue

        if year is not None:
            item_year = _extract_year(item)
            if item_year != str(year):
                continue

        results.append(item)

        if len(results) >= top_k:
            break

    return results