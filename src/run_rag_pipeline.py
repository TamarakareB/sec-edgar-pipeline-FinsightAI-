import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RAG_DIR = PROJECT_ROOT / "rag"
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
DATASET_DIR = DATA_DIR / "dataset"

DATASET_PATH = DATASET_DIR / "edgar_chunks.jsonl"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
METADATA_PATH = INDEX_DIR / "metadata.pkl"
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"

STEP_SCRIPTS = {
    "embed": RAG_DIR / "embed_chunks.py",
    "index": RAG_DIR / "build_index.py",
    "retrieve": RAG_DIR / "retrieve.py",
}

BUILD_STEPS = ["embed", "index"]


def run_python_script(script_path: Path, extra_args=None):
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print("\n" + "=" * 80)
    print(f"Running step: {script_path.name}")
    print("Command:", " ".join(cmd))
    print("=" * 80)

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_path.name} failed with exit code {result.returncode}"
        )


def validate_for_embed():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {DATASET_PATH}\n"
            f"Please make sure edgar_chunks.jsonl exists before running embed."
        )


def validate_for_index():
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Missing embeddings file: {EMBEDDINGS_PATH}\n"
            f"Run the embed step first."
        )
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {METADATA_PATH}\n"
            f"Run the embed step first."
        )


def validate_for_retrieve():
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Missing FAISS index: {FAISS_INDEX_PATH}\n"
            f"Run the index step first."
        )
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {METADATA_PATH}\n"
            f"Run the embed step first."
        )


def step_embed():
    validate_for_embed()
    run_python_script(STEP_SCRIPTS["embed"])
    print("[DONE] embed completed.")


def step_index():
    validate_for_index()
    run_python_script(STEP_SCRIPTS["index"])
    print("[DONE] index completed.")


def step_retrieve(query: str, top_k: int = 5, ticker: str = None, year: str = None):
    validate_for_retrieve()

    extra_args = ["--query", query, "--top_k", str(top_k)]

    if ticker:
        extra_args.extend(["--ticker", ticker])

    if year:
        extra_args.extend(["--year", str(year)])

    run_python_script(STEP_SCRIPTS["retrieve"], extra_args=extra_args)
    print("[DONE] retrieve completed.")


def run_build_pipeline(from_step: str = "embed"):
    start_idx = BUILD_STEPS.index(from_step)
    for step in BUILD_STEPS[start_idx:]:
        if step == "embed":
            step_embed()
        elif step == "index":
            step_index()


def main():
    parser = argparse.ArgumentParser(description="Run the RAG pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_all = subparsers.add_parser("all", help="Run embed + index")
    parser_all.add_argument(
        "--from-step",
        choices=BUILD_STEPS,
        default="embed",
        help="Resume the pipeline from a specific step"
    )

    subparsers.add_parser("embed", help="Run embedding step only")
    subparsers.add_parser("index", help="Run index-building step only")

    parser_retrieve = subparsers.add_parser("retrieve", help="Run retrieval")
    parser_retrieve.add_argument("--query", required=True, type=str, help="Search query")
    parser_retrieve.add_argument("--top_k", default=5, type=int, help="Top-k results")
    parser_retrieve.add_argument("--ticker", default=None, type=str, help="Optional ticker filter")
    parser_retrieve.add_argument("--year", default=None, type=str, help="Optional year filter")

    args = parser.parse_args()

    if args.command == "all":
        run_build_pipeline(from_step=args.from_step)
    elif args.command == "embed":
        step_embed()
    elif args.command == "index":
        step_index()
    elif args.command == "retrieve":
        step_retrieve(
            query=args.query,
            top_k=args.top_k,
            ticker=args.ticker,
            year=args.year,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()