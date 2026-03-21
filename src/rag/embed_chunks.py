import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle

DATA_PATH = Path("data/dataset/edgar_chunks.jsonl")
OUTPUT_DIR = Path("data/index")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []
metadata = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        row = json.loads(line)
        texts.append(row["text"])
        metadata.append(row)

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

np.save(OUTPUT_DIR / "embeddings.npy", embeddings)

with open(OUTPUT_DIR / "metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("Embeddings saved to data/index/")
# Verify the contents of the dataset