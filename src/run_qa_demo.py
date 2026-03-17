# run_QAdemo.py
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# Path setup (compatible with your current project structure)
# -----------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
RAG_DIR = CURRENT_DIR / "rag"

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

# Local imports
from qa_pipeline import ask
from risk_classifier import classify_risk
from trend_analyzer import (
    analyze_trend,
    get_financial_trend,
    detect_emerging_risks,
)
from compare_engine import compare_companies, sector_comparison
from retrieve import search


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
VALID_MODES = {"auto", "qa", "risk", "trend", "compare"}
VALID_TREND_SUBMODES = {
    "auto",
    "narrative",
    "financial",
    "emerging_risk",
}


def log(msg: str, debug: bool = False) -> None:
    if debug:
        print(msg)


def normalize_ticker(ticker: Optional[str]) -> Optional[str]:
    if ticker is None:
        return None
    ticker = ticker.strip().upper()
    return ticker or None


def normalize_tickers(tickers: Optional[str]) -> List[str]:
    if not tickers:
        return []
    parts = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    return list(dict.fromkeys(parts))  # dedupe, keep order


def parse_years(years_str: Optional[str]) -> Optional[List[int]]:
    if not years_str:
        return None
    result = []
    for part in years_str.split(","):
        part = part.strip()
        if not part:
            continue
        result.append(int(part))
    return result if result else None


def infer_mode(question: str, mode: str) -> str:
    if mode != "auto":
        return mode

    q = question.lower()

    compare_keywords = ["compare", "comparison", "versus", "vs", "difference between", "differ"]
    trend_keywords = ["trend", "over time", "changed", "change over", "increase", "decrease", "grew", "declined"]
    risk_keywords = ["risk type", "classify risk", "risk category", "what risks", "main risks", "risk labels"]

    if any(k in q for k in compare_keywords):
        return "compare"
    if any(k in q for k in trend_keywords):
        return "trend"
    if any(k in q for k in risk_keywords):
        return "risk"
    return "qa"


def infer_trend_submode(question: str, trend_submode: str) -> str:
    if trend_submode != "auto":
        return trend_submode

    q = question.lower()

    financial_keywords = [
        "revenue", "net income", "gross profit", "operating income", "eps",
        "cash flow", "debt", "assets", "liabilities", "equity", "financial trend"
    ]
    emerging_keywords = [
        "emerging risk", "emerged", "new risks", "faded risks", "stable risks",
        "appeared", "disappeared"
    ]

    if any(k in q for k in financial_keywords):
        return "financial"
    if any(k in q for k in emerging_keywords):
        return "emerging_risk"
    return "narrative"


def validate_inputs(
    *,
    question: str,
    mode: str,
    trend_submode: str,
    ticker: Optional[str],
    tickers: List[str],
    sector: Optional[str],
) -> None:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Choose from {sorted(VALID_MODES)}.")

    if trend_submode not in VALID_TREND_SUBMODES:
        raise ValueError(
            f"Invalid trend_submode '{trend_submode}'. Choose from {sorted(VALID_TREND_SUBMODES)}."
        )

    # mode-specific checks
    effective_mode = infer_mode(question, mode)

    if effective_mode == "compare":
        if not tickers and not sector:
            raise ValueError("Compare mode requires either --tickers or --sector.")

    if effective_mode == "trend":
        effective_submode = infer_trend_submode(question, trend_submode)
        if effective_submode in {"narrative", "financial", "emerging_risk"} and not ticker:
            raise ValueError(f"Trend submode '{effective_submode}' requires --ticker.")

    if effective_mode == "risk":
        # Strong recommendation, not hard failure if you want broad retrieval
        pass


def chunk_preview(text: str, n: int = 220) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text[:n]


def build_chunk_index(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index = {}
    for ch in chunks:
        cid = ch.get("chunk_id")
        if cid is not None:
            index[str(cid)] = ch
    return index


def safe_search(
    question: str,
    top_k: int,
    ticker: Optional[str] = None,
    year: Optional[int] = None,
) -> List[Dict[str, Any]]:
    results = search(question, top_k=top_k, ticker=ticker, year=year)
    return results or []


def format_risk_result(
    question: str,
    ticker: Optional[str],
    year: Optional[int],
    chunks: List[Dict[str, Any]],
    classifier_output: Dict[str, Any],
) -> Dict[str, Any]:
    chunk_map = build_chunk_index(chunks)
    labels = classifier_output.get("risk_labels", []) or []

    enriched = []
    for item in labels:
        cid = str(item.get("chunk_id", ""))
        ch = chunk_map.get(cid, {})
        enriched.append({
            "label": item.get("label", ""),
            "chunk_id": item.get("chunk_id", ""),
            "ticker": ch.get("ticker", ticker or ""),
            "form": ch.get("form", ""),
            "year": str(ch.get("filing_date", ""))[:4] or str(ch.get("year", "")),
            "section": ch.get("section", ""),
            "relevance_score": ch.get("relevance_score", None),
            "preview": chunk_preview(ch.get("text", "")),
        })

    warnings = []
    if classifier_output.get("warning"):
        warnings.append(classifier_output["warning"])

    return {
        "mode": "risk",
        "submode": None,
        "question": question,
        "ticker": ticker,
        "year": year,
        "answer": None,
        "result": {
            "risk_labels": enriched
        },
        "citations": enriched,
        "confidence": None,
        "warnings": warnings,
        "debug": {
            "chunks_retrieved": len(chunks)
        }
    }


def format_qa_result(raw: Dict[str, Any], question: str, ticker: Optional[str], year: Optional[int]) -> Dict[str, Any]:
    return {
        "mode": "qa",
        "submode": None,
        "question": question,
        "ticker": ticker,
        "year": year,
        "answer": raw.get("answer"),
        "result": None,
        "citations": raw.get("citations", []),
        "confidence": raw.get("confidence"),
        "warnings": [],
        "debug": {
            "chunks_used": raw.get("chunks_used"),
            "sections_used": raw.get("sections_used", []),
            "followups_count": len(raw.get("followups", [])),
            "followups": raw.get("followups", []),
        }
    }


def format_trend_narrative_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    warnings = []
    if raw.get("error"):
        warnings.append(raw["error"])

    return {
        "mode": "trend",
        "submode": "narrative",
        "question": raw.get("question"),
        "ticker": raw.get("ticker"),
        "year": None,
        "answer": raw.get("answer"),
        "result": {
            "years": raw.get("years", []),
            "data_points": raw.get("data_points", {}),
        },
        "citations": None,
        "confidence": None,
        "warnings": warnings,
        "debug": {
            "years": raw.get("years", []),
            "data_points": raw.get("data_points", {}),
        }
    }


def format_trend_financial_result(question: str, raw: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    warnings = []
    if raw.get("error"):
        warnings.append(raw["error"])

    return {
        "mode": "trend",
        "submode": "financial",
        "question": question,
        "ticker": ticker,
        "year": None,
        "answer": None,
        "result": raw,
        "citations": None,
        "confidence": None,
        "warnings": warnings,
        "debug": {
            "record_count": len(raw.get("records", [])) if isinstance(raw.get("records"), list) else 0,
            "years": raw.get("years", []),
        }
    }


def format_trend_emerging_result(question: str, raw: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    warnings = []
    if raw.get("error"):
        warnings.append(raw["error"])

    return {
        "mode": "trend",
        "submode": "emerging_risk",
        "question": question,
        "ticker": ticker,
        "year": None,
        "answer": raw.get("summary"),
        "result": raw,
        "citations": None,
        "confidence": None,
        "warnings": warnings,
        "debug": {
            "emerging_count": len(raw.get("emerging", [])),
            "stable_count": len(raw.get("stable", [])),
            "faded_count": len(raw.get("faded", [])),
        }
    }


def format_compare_result(raw: Dict[str, Any], submode: str) -> Dict[str, Any]:
    warnings = []
    if raw.get("error"):
        warnings.append(raw["error"])

    return {
        "mode": "compare",
        "submode": submode,
        "question": raw.get("question"),
        "ticker": None,
        "year": raw.get("year"),
        "answer": raw.get("answer"),
        "result": {
            "companies": raw.get("companies", []),
            "chunks_per_company": raw.get("chunks_per_company", {}),
        },
        "citations": raw.get("citations", {}),
        "confidence": None,
        "warnings": warnings,
        "debug": {
            "companies": raw.get("companies", []),
            "chunks_per_company": raw.get("chunks_per_company", {}),
        }
    }


# -----------------------------------------------------------------------------
# Main orchestrator
# -----------------------------------------------------------------------------
def run_qa_demo(
    question: str,
    mode: str = "auto",
    trend_submode: str = "auto",
    ticker: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    sector: Optional[str] = None,
    year: Optional[int] = None,
    years: Optional[List[int]] = None,
    top_k: int = 5,
    include_followups: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    ticker = normalize_ticker(ticker)
    tickers = tickers or []

    validate_inputs(
        question=question,
        mode=mode,
        trend_submode=trend_submode,
        ticker=ticker,
        tickers=tickers,
        sector=sector,
    )

    effective_mode = infer_mode(question, mode)
    log(f"[1/5] Effective mode: {effective_mode}", debug)

    # -------------------------------------------------------------------------
    # QA MODE
    # -------------------------------------------------------------------------
    if effective_mode == "qa":
        log("[2/5] Running QA pipeline...", debug)
        raw = ask(
            question=question,
            ticker=ticker,
            year=year,
            top_k=top_k,
            include_followups=include_followups,
        )
        return format_qa_result(raw, question, ticker, year)

    # -------------------------------------------------------------------------
    # RISK MODE
    # -------------------------------------------------------------------------
    if effective_mode == "risk":
        log("[2/5] Retrieving chunks for risk classification...", debug)
        chunks = safe_search(question, top_k=top_k, ticker=ticker, year=year)

        if not chunks:
            return {
                "mode": "risk",
                "submode": None,
                "question": question,
                "ticker": ticker,
                "year": year,
                "answer": "No relevant documents found for risk classification.",
                "result": {"risk_labels": []},
                "citations": [],
                "confidence": None,
                "warnings": ["No chunks retrieved."],
                "debug": {"chunks_retrieved": 0},
            }

        log(f"[3/5] Retrieved {len(chunks)} chunks. Running classifier...", debug)
        classifier_output = classify_risk(question, chunks)
        return format_risk_result(question, ticker, year, chunks, classifier_output)

    # -------------------------------------------------------------------------
    # TREND MODE
    # -------------------------------------------------------------------------
    if effective_mode == "trend":
        effective_submode = infer_trend_submode(question, trend_submode)
        log(f"[2/5] Trend submode: {effective_submode}", debug)

        if effective_submode == "narrative":
            log("[3/5] Running narrative trend analysis...", debug)
            raw = analyze_trend(
                ticker=ticker,
                question=question,
                years=years,
            )
            return format_trend_narrative_result(raw)

        if effective_submode == "financial":
            log("[3/5] Running financial trend analysis from SQL...", debug)
            raw = get_financial_trend(ticker)
            return format_trend_financial_result(question, raw, ticker)

        if effective_submode == "emerging_risk":
            log("[3/5] Running emerging risk detection...", debug)
            raw = detect_emerging_risks(ticker)
            return format_trend_emerging_result(question, raw, ticker)

        raise ValueError(f"Unsupported trend submode: {effective_submode}")

    # -------------------------------------------------------------------------
    # COMPARE MODE
    # -------------------------------------------------------------------------
    if effective_mode == "compare":
        if sector:
            log(f"[2/5] Running sector comparison for sector='{sector}'...", debug)
            raw = sector_comparison(question=question, sector=sector, year=year)
            return format_compare_result(raw, submode="sector")
        else:
            log(f"[2/5] Running company comparison for tickers={tickers}...", debug)
            raw = compare_companies(
                question=question,
                tickers=tickers,
                year=year,
                top_k_per_company=top_k,
            )
            return format_compare_result(raw, submode="companies")

    raise ValueError(f"Unhandled mode: {effective_mode}")


# -----------------------------------------------------------------------------
# Pretty print
# -----------------------------------------------------------------------------
def print_result(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(f"MODE: {result.get('mode')}")
    if result.get("submode"):
        print(f"SUBMODE: {result.get('submode')}")
    print(f"QUESTION: {result.get('question')}")
    if result.get("ticker"):
        print(f"TICKER: {result.get('ticker')}")
    if result.get("year"):
        print(f"YEAR: {result.get('year')}")
    print("=" * 80)

    if result.get("answer"):
        print("\nANSWER / SUMMARY:\n")
        print(result["answer"])

    if result.get("result") is not None:
        print("\nRESULT:\n")
        print(json.dumps(result["result"], indent=2, ensure_ascii=False, default=str))

    if result.get("citations"):
        print("\nCITATIONS:\n")
        print(json.dumps(result["citations"], indent=2, ensure_ascii=False, default=str))

    if result.get("confidence") is not None:
        print(f"\nCONFIDENCE: {result['confidence']}")

    warnings = result.get("warnings", [])
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"- {w}")

    debug = result.get("debug", {})
    if debug:
        print("\nDEBUG INFO:\n")
        print(json.dumps(debug, indent=2, ensure_ascii=False, default=str))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FinSightAI QA Demo Runner")

    parser.add_argument("--question", type=str, required=True, help="User question")
    parser.add_argument("--mode", type=str, default="auto",
                        choices=sorted(VALID_MODES),
                        help="Run mode")
    parser.add_argument("--trend_submode", type=str, default="auto",
                        choices=sorted(VALID_TREND_SUBMODES),
                        help="Trend submode")
    parser.add_argument("--ticker", type=str, default=None, help="Single ticker, e.g. AAPL")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated tickers for compare mode, e.g. AAPL,MSFT,NVDA")
    parser.add_argument("--sector", type=str, default=None,
                        help="Sector for sector comparison, e.g. technology")
    parser.add_argument("--year", type=int, default=None, help="Single filing year filter")
    parser.add_argument("--years", type=str, default=None,
                        help="Comma-separated years for narrative trend, e.g. 2019,2020,2021,2022,2023")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k retrieval size")
    parser.add_argument("--include_followups", action="store_true",
                        help="Enable follow-up question generation for QA mode")
    parser.add_argument("--debug", action="store_true", help="Print debug logs")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    tickers = normalize_tickers(args.tickers)
    years = parse_years(args.years)

    result = run_qa_demo(
        question=args.question,
        mode=args.mode,
        trend_submode=args.trend_submode,
        ticker=args.ticker,
        tickers=tickers,
        sector=args.sector,
        year=args.year,
        years=years,
        top_k=args.top_k,
        include_followups=args.include_followups,
        debug=args.debug,
    )

    print_result(result)


if __name__ == "__main__":
    main()