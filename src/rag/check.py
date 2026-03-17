import pickle
from collections import Counter
from pathlib import Path

meta_path = Path("data/index/metadata.pkl")

with open(meta_path, "rb") as f:
    metadata = pickle.load(f)

print("Total chunks:", len(metadata))

tickers = Counter()
years = Counter()
pairs = Counter()

for m in metadata:
    ticker = m.get("ticker")
    year = m.get("year")
    tickers[ticker] += 1
    years[year] += 1
    pairs[(ticker, year)] += 1

print("\nTop tickers:")
print(tickers.most_common(20))

print("\nYears:")
print(years.most_common())

print("\nSample ticker-year pairs:")
print(pairs.most_common(30))