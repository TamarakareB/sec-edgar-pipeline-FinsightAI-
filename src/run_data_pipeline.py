import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # because file is in src/
PYTHON = sys.executable

STEPS = ["clean", "fetch", "build-docs", "build-chunks"]


# ---------------------------
# Basic utilities
# ---------------------------
def run_command(cmd: list[str], step_name: str):
    print("\n" + "=" * 70)
    print(f"[RUNNING] {step_name}")
    print("=" * 70)
    print("Command:", " ".join(cmd))
    print("Working directory:", ROOT)

    result = subprocess.run(cmd, cwd=ROOT)

    if result.returncode != 0:
        raise RuntimeError(f"{step_name} failed with exit code {result.returncode}")

    print(f"[DONE] {step_name}")


def file_exists_and_not_empty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def jsonl_has_content(path: Path) -> bool:
    if not file_exists_and_not_empty(path):
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return True
    except Exception:
        return False
    return False


def meta_is_valid(meta_path: Path) -> bool:
    """
    Valid only if:
    1) file exists and is non-empty
    2) JSON can be parsed into a non-empty list
    3) at least one local_path exists on disk
    """
    if not file_exists_and_not_empty(meta_path):
        return False

    try:
        rows = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if not isinstance(rows, list) or len(rows) == 0:
        return False

    existing_raw_files = 0
    for r in rows:
        local_path = r.get("local_path")
        if local_path and Path(local_path).exists():
            existing_raw_files += 1

    return existing_raw_files > 0


def print_status(clean_csv: Path, meta_path: Path, docs_path: Path, chunks_path: Path):
    print("\n" + "-" * 70)
    print("Current artifact status")
    print("-" * 70)
    print(f"clean csv     : {clean_csv} -> {'OK' if file_exists_and_not_empty(clean_csv) else 'MISSING/EMPTY'}")
    print(f"fetch meta    : {meta_path} -> {'VALID' if meta_is_valid(meta_path) else 'INVALID/MISSING'}")
    print(f"docs jsonl    : {docs_path} -> {'OK' if jsonl_has_content(docs_path) else 'MISSING/EMPTY'}")
    print(f"chunks jsonl  : {chunks_path} -> {'OK' if jsonl_has_content(chunks_path) else 'MISSING/EMPTY'}")
    print("-" * 70)


def should_run(step: str, start_from: str) -> bool:
    return STEPS.index(step) >= STEPS.index(start_from)


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Robust FinSightAI data pipeline runner")

    parser.add_argument(
        "--start-from",
        choices=STEPS,
        default="clean",
        help="Start pipeline from a specific step",
    )

    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--force-fetch", action="store_true")
    parser.add_argument("--force-build-docs", action="store_true")
    parser.add_argument("--force-build-chunks", action="store_true")

    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)

    args = parser.parse_args()

    clean_csv = ROOT / "data" / "top_sp500.csv"
    meta_path = ROOT / "data" / "dataset" / "edgar_meta.json"
    docs_path = ROOT / "data" / "dataset" / "edgar_docs.jsonl"
    chunks_path = ROOT / "data" / "dataset" / "edgar_chunks.jsonl"

    print("\n" + "#" * 70)
    print("FinSightAI Data Pipeline")
    print("#" * 70)
    print(f"Project root: {ROOT}")
    print(f"Python executable: {PYTHON}")
    print(f"Start from step: {args.start_from}")

    print_status(clean_csv, meta_path, docs_path, chunks_path)

    # ---------------------------
    # Step 1: clean
    # ---------------------------
    if should_run("clean", args.start_from):
        if file_exists_and_not_empty(clean_csv) and not args.force_clean:
            print(f"[AUTO-SKIP] clean already available: {clean_csv}")
        else:
            run_command(
                [PYTHON, "-m", "src.data.clean_top50_csv"],
                "Clean / validate top50 CSV",
            )
            if not file_exists_and_not_empty(clean_csv):
                raise RuntimeError(f"Clean step completed but output missing/empty: {clean_csv}")
    else:
        print("[SKIP-BY-START] clean")

    # ---------------------------
    # Step 2: fetch
    # ---------------------------
    if should_run("fetch", args.start_from):
        if meta_is_valid(meta_path) and not args.force_fetch:
            print(f"[AUTO-SKIP] fetch artifacts already valid: {meta_path}")
        else:
            run_command(
                [PYTHON, "-m", "src.data.fetch_top50"],
                "Fetch EDGAR filings for top companies",
            )
            if not meta_is_valid(meta_path):
                raise RuntimeError(
                    f"Fetch step completed but metadata/raw files are still invalid: {meta_path}"
                )
    else:
        print("[SKIP-BY-START] fetch")

    # ---------------------------
    # Step 3: build-docs
    # ---------------------------
    if should_run("build-docs", args.start_from):
        if jsonl_has_content(docs_path) and not args.force_build_docs:
            print(f"[AUTO-SKIP] docs already available: {docs_path}")
        else:
            if not meta_is_valid(meta_path):
                raise RuntimeError(
                    f"Cannot run build-docs because fetch artifacts are invalid: {meta_path}"
                )

            run_command(
                [PYTHON, "-m", "src.data.sec_edgar_pipeline", "build-docs"],
                "Build document-level dataset",
            )

            if not jsonl_has_content(docs_path):
                raise RuntimeError(
                    f"build-docs finished but docs output is missing/empty: {docs_path}"
                )
    else:
        print("[SKIP-BY-START] build-docs")

    # ---------------------------
    # Step 4: build-chunks
    # ---------------------------
    if should_run("build-chunks", args.start_from):
        if jsonl_has_content(chunks_path) and not args.force_build_chunks:
            print(f"[AUTO-SKIP] chunks already available: {chunks_path}")
        else:
            if not jsonl_has_content(docs_path):
                raise RuntimeError(
                    f"Cannot run build-chunks because docs output is missing/empty: {docs_path}"
                )

            run_command(
                [
                    PYTHON,
                    "-m",
                    "src.data.sec_edgar_pipeline",
                    "build-chunks",
                    "--chunk-size",
                    str(args.chunk_size),
                    "--overlap",
                    str(args.overlap),
                ],
                "Build chunk-level dataset",
            )

            if not jsonl_has_content(chunks_path):
                raise RuntimeError(
                    f"build-chunks finished but chunks output is missing/empty: {chunks_path}"
                )
    else:
        print("[SKIP-BY-START] build-chunks")

    print_status(clean_csv, meta_path, docs_path, chunks_path)

    print("\n" + "#" * 70)
    print("Data pipeline finished successfully.")
    print("#" * 70)


if __name__ == "__main__":
    main()