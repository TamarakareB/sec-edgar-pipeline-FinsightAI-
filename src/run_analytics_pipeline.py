import argparse
import subprocess
import sys
from pathlib import Path


# =============================================================================
# Paths
# =============================================================================

# This file is expected to live at: project_root/src/run_analytics_pipeline.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
ANA_DIR = SRC_DIR / "ana"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SQL_DIR = DATA_DIR / "sql"

# Optional recommended output directory for future standardization
OUTPUT_DIR = DATA_DIR / "outputs"
ANALYTICS_DIR = OUTPUT_DIR / "analytics"

# Recommended output files (warnings only by default, not hard requirements)
RECOMMENDED_OUTPUTS = {
    "extract": ANALYTICS_DIR / "financial_metrics.json",
    "risk": ANALYTICS_DIR / "risk_summary.json",
    "compare": ANALYTICS_DIR / "comparison_results.json",
    "trend": ANALYTICS_DIR / "trend_results.json",
}

STEP_SCRIPTS = {
    "extract": ANA_DIR / "financial_extractor.py",
    "risk": ANA_DIR / "risk_classifier.py",
    "compare": ANA_DIR / "compare_engine.py",
    "trend": ANA_DIR / "trend_analyzer.py",
}

PIPELINE_STEPS = ["extract", "risk", "compare", "trend"]


# =============================================================================
# Helpers
# =============================================================================

def print_header(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def run_python_script(script_path: Path, extra_args=None):
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    module_path = script_path.relative_to(PROJECT_ROOT).with_suffix("")
    module_name = ".".join(module_path.parts)

    cmd = [sys.executable, "-m", module_name] 
    if extra_args:
        cmd.extend(extra_args)

    print_header(f"Running step: {script_path.name}")
    print("Command:", " ".join(cmd))
    print("Working directory:", PROJECT_ROOT)

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_path.name} failed with exit code {result.returncode}"
        )


def ensure_script_exists(step_name: str):
    script_path = STEP_SCRIPTS[step_name]
    if not script_path.exists():
        raise FileNotFoundError(
            f"Missing script for step '{step_name}': {script_path}"
        )


def warn_if_recommended_output_missing(step_name: str):
    output_path = RECOMMENDED_OUTPUTS.get(step_name)
    if output_path and not output_path.exists():
        print(
            f"[WARN] Recommended output for '{step_name}' not found: {output_path}\n"
            f"       This is only a warning. If your module saves somewhere else, update\n"
            f"       RECOMMENDED_OUTPUTS in run_analytics_pipeline.py."
        )


# =============================================================================
# Validation
# =============================================================================

def validate_common():
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Missing src directory: {SRC_DIR}")

    if not ANA_DIR.exists():
        raise FileNotFoundError(f"Missing analytics module directory: {ANA_DIR}")

    if not DATA_DIR.exists():
        print(f"[WARN] Data directory not found yet: {DATA_DIR}")

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not ANALYTICS_DIR.exists():
        ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)


def validate_for_extract():
    validate_common()
    ensure_script_exists("extract")

    if not RAW_DIR.exists():
        print(
            f"[WARN] Raw data directory not found: {RAW_DIR}\n"
            f"       If financial_extractor.py reads from another location, this is okay."
        )


def validate_for_risk():
    validate_common()
    ensure_script_exists("risk")

    # Soft dependency warning only
    warn_if_recommended_output_missing("extract")


def validate_for_compare():
    validate_common()
    ensure_script_exists("compare")

    # Soft dependency warnings only
    warn_if_recommended_output_missing("extract")
    warn_if_recommended_output_missing("risk")


def validate_for_trend():
    validate_common()
    ensure_script_exists("trend")

    # Soft dependency warnings only
    warn_if_recommended_output_missing("extract")
    warn_if_recommended_output_missing("risk")
    warn_if_recommended_output_missing("compare")


# =============================================================================
# Step runners
# =============================================================================

def step_extract():
    validate_for_extract()
    run_python_script(STEP_SCRIPTS["extract"])
    print("[DONE] extract completed.")
    warn_if_recommended_output_missing("extract")


def step_risk():
    validate_for_risk()
    run_python_script(STEP_SCRIPTS["risk"])
    print("[DONE] risk completed.")
    warn_if_recommended_output_missing("risk")


def step_compare():
    validate_for_compare()
    run_python_script(STEP_SCRIPTS["compare"])
    print("[DONE] compare completed.")
    warn_if_recommended_output_missing("compare")


def step_trend():
    validate_for_trend()
    run_python_script(STEP_SCRIPTS["trend"])
    print("[DONE] trend completed.")
    warn_if_recommended_output_missing("trend")


# =============================================================================
# Pipeline orchestration
# =============================================================================

def run_analytics_pipeline(from_step: str = "extract"):
    if from_step not in PIPELINE_STEPS:
        raise ValueError(
            f"Invalid from_step: {from_step}. Expected one of {PIPELINE_STEPS}"
        )

    start_idx = PIPELINE_STEPS.index(from_step)

    for step in PIPELINE_STEPS[start_idx:]:
        if step == "extract":
            step_extract()
        elif step == "risk":
            step_risk()
        elif step == "compare":
            step_compare()
        elif step == "trend":
            step_trend()
        else:
            raise ValueError(f"Unknown step: {step}")

    print_header("Analytics pipeline finished successfully")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the non-QA analytics pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_all = subparsers.add_parser(
        "all",
        help="Run extract + risk + compare + trend"
    )
    parser_all.add_argument(
        "--from-step",
        choices=PIPELINE_STEPS,
        default="extract",
        help="Resume pipeline from a specific step"
    )

    subparsers.add_parser("extract", help="Run financial extraction only")
    subparsers.add_parser("risk", help="Run risk classification only")
    subparsers.add_parser("compare", help="Run compare engine only")
    subparsers.add_parser("trend", help="Run trend analyzer only")

    args = parser.parse_args()

    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("ANA_DIR =", ANA_DIR)
    print("DATA_DIR =", DATA_DIR)

    if args.command == "all":
        run_analytics_pipeline(from_step=args.from_step)
    elif args.command == "extract":
        step_extract()
    elif args.command == "risk":
        step_risk()
    elif args.command == "compare":
        step_compare()
    elif args.command == "trend":
        step_trend()
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()