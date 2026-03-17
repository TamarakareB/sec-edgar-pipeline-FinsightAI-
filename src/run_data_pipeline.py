import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PYTHON = sys.executable


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


def main():
    parser = argparse.ArgumentParser(description="Run FinSightAI data pipeline")

    # manual skip
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-build-docs", action="store_true")
    parser.add_argument("--skip-build-chunks", action="store_true")

    # force rerun
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--force-fetch", action="store_true")
    parser.add_argument("--force-build-docs", action="store_true")
    parser.add_argument("--force-build-chunks", action="store_true")

    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)

    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("FinSightAI Data Pipeline")
    print("#" * 70)
    print(f"Project root: {ROOT}")
    print(f"Python executable: {PYTHON}")

    # expected outputs
    clean_output = ROOT / "data" / "top_sp500.csv"
    meta_output = ROOT / "data" / "dataset" / "edgar_meta.json"
    docs_output = ROOT / "data" / "dataset" / "edgar_docs.jsonl"
    chunks_output = ROOT / "data" / "dataset" / "edgar_chunks.jsonl"

    # Step 1: clean / validate CSV
    if args.skip_clean:
        print("[SKIP] Clean / validate top50 CSV (manual skip)")
    elif file_exists_and_not_empty(clean_output) and not args.force_clean:
        print(f"[AUTO-SKIP] Clean step skipped because file already exists: {clean_output}")
    else:
        run_command(
            [PYTHON, "-m", "src.data.clean_top50_csv"],
            "Clean / validate top50 CSV",
        )

    # Step 2: fetch EDGAR filings
    if args.skip_fetch:
        print("[SKIP] Fetch EDGAR filings (manual skip)")
    elif file_exists_and_not_empty(meta_output) and not args.force_fetch:
        print(f"[AUTO-SKIP] Fetch step skipped because file already exists: {meta_output}")
    else:
        run_command(
            [PYTHON, "-m", "src.data.fetch_top50"],
            "Fetch EDGAR filings for top companies",
        )

    # Step 3: build docs
    if args.skip_build_docs:
        print("[SKIP] Build document-level dataset (manual skip)")
    elif file_exists_and_not_empty(docs_output) and not args.force_build_docs:
        print(f"[AUTO-SKIP] Build-docs step skipped because file already exists: {docs_output}")
    else:
        if not file_exists_and_not_empty(meta_output):
            raise RuntimeError(
                f"❌ Cannot build docs because metadata file is missing: {meta_output}"
            )

        run_command(
            [PYTHON, "-m", "src.data.sec_edgar_pipeline", "build-docs"],
            "Build document-level dataset",
        )

    # Step 4: build chunks
    if args.skip_build_chunks:
        print("[SKIP] Build chunk-level dataset (manual skip)")
    elif file_exists_and_not_empty(chunks_output) and not args.force_build_chunks:
        print(f"[AUTO-SKIP] Build-chunks step skipped because file already exists: {chunks_output}")
    else:
        if not file_exists_and_not_empty(docs_output):
            raise RuntimeError(
                f"❌ Cannot build chunks because docs file is missing: {docs_output}"
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

    print("\n" + "#" * 70)
    print("Data pipeline finished successfully.")
    print("#" * 70)


if __name__ == "__main__":
    main()