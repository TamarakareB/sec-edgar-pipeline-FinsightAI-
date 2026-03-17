# src/sec_edgar_pipeline.py

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from src.config import settings


# -----------------------
# Helpers
# -----------------------
def pad_cik(cik: str | int) -> str:
    # SEC submissions endpoint uses 10-digit zero-padded CIK
    return str(cik).lstrip("0").zfill(10)


def cik_nopad(cik: str | int) -> str:
    # For /Archives/edgar/data/<CIK>/ path (no leading zeros)
    return str(int(str(cik)))


def accession_no_dashes(accession: str) -> str:
    return accession.replace("-", "")


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def make_headers(host: str) -> dict:
    # SEC requires a descriptive User-Agent (see src/config.py)
    return {
        "User-Agent": settings.user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": host,
    }


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20))
def http_get(url: str, headers: dict, timeout: int = 30) -> requests.Response:
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r


def html_to_text(html: str) -> str:
    # Use lxml parser if installed; bs4 will fall back if not.
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    text = normalize_text(text)
    n = len(text)
    out = []
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        out.append((text[start:end], start, end))
        if end == n:
            break
        start = max(0, end - overlap)
    return out


# -----------------------
# EDGAR: submissions + url building
# -----------------------
def get_submissions_json(cik: str | int) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{pad_cik(cik)}.json"
    r = http_get(url, headers=make_headers("data.sec.gov"))
    time.sleep(settings.sleep_seconds)
    return r.json()


def iter_recent_filings(submissions_json: dict, forms: set[str], max_scan: int = 800):
    recent = submissions_json.get("filings", {}).get("recent", {})
    form_list = recent.get("form", [])
    accession_list = recent.get("accessionNumber", [])
    filing_date_list = recent.get("filingDate", [])
    report_date_list = recent.get("reportDate", [])
    primary_doc_list = recent.get("primaryDocument", [])
    primary_desc_list = recent.get("primaryDocDescription", [])

    n = min(len(form_list), max_scan)
    for i in range(n):
        form = form_list[i]
        if form not in forms:
            continue
        yield {
            "form": form,
            "accessionNumber": accession_list[i] if i < len(accession_list) else None,
            "filingDate": filing_date_list[i] if i < len(filing_date_list) else None,
            "reportDate": report_date_list[i] if i < len(report_date_list) else None,
            "primaryDocument": primary_doc_list[i] if i < len(primary_doc_list) else None,
            "primaryDocDescription": primary_desc_list[i] if i < len(primary_desc_list) else None,
        }


def build_primary_doc_url(cik: str | int, accession: str, primary_doc: str) -> str:
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_nopad(cik)}/{accession_no_dashes(accession)}/{primary_doc}"
    )


def download_bytes(url: str, out_path: Path) -> bytes:
    r = http_get(url, headers=make_headers("www.sec.gov"), timeout=60)
    time.sleep(settings.sleep_seconds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(r.content)
    return r.content


# -----------------------
# Stage 1: fetch (download only)
# -----------------------
def stage_fetch(cik: str | int, ticker: str, company: str, forms: set[str], max_filings: int):
    settings.raw_dir.mkdir(parents=True, exist_ok=True)

    subs = get_submissions_json(cik)

    filings = []
    for f in iter_recent_filings(subs, forms=forms):
        if f.get("accessionNumber") and f.get("primaryDocument"):
            filings.append(f)
        if len(filings) >= max_filings:
            break

    print(f"[fetch] Selected filings: {len(filings)}")

    meta_rows = []
    for f in tqdm(filings, desc="fetch"):
        accession = f["accessionNumber"]
        primary_doc = f["primaryDocument"]
        source_url = build_primary_doc_url(cik, accession, primary_doc)

        local_name = safe_filename(
            f"{ticker}_{f['form']}_{f.get('filingDate','')}_{accession}_{primary_doc}"
        )
        local_path = settings.raw_dir / ticker / f["form"] / local_name

        if not local_path.exists():
            try:
                download_bytes(source_url, local_path)
            except Exception as e:
                print(f"[WARN] download failed: {source_url} ({e})")
                continue

        meta_rows.append(
            {
                "company": company,
                "ticker": ticker,
                "cik": cik_nopad(cik),
                "form": f["form"],
                "filing_date": f.get("filingDate"),
                "report_date": f.get("reportDate"),
                "accession_number": accession,
                "primary_document": primary_doc,
                "primary_doc_description": f.get("primaryDocDescription"),
                "source_url": source_url,
                "local_path": str(local_path),
            }
        )

    # ✅ Save meta (append + dedupe) so running fetch repeatedly doesn't overwrite previous companies
    meta_path = settings.dataset_dir / "edgar_meta.json"
    settings.dataset_dir.mkdir(parents=True, exist_ok=True)

    existing = []
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []

    combined = existing + meta_rows
    seen = set()
    deduped = []
    for r in combined:
        key = (r.get("accession_number"), r.get("primary_document"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    meta_path.write_text(
        json.dumps(deduped, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[fetch] Saved meta: {meta_path} (rows={len(deduped)})")


# -----------------------
# Stage 2: build docs dataset from raw + meta
# -----------------------
def stage_build_docs():
    meta_path = settings.dataset_dir / "edgar_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Run: python -m src.data.sec_edgar_pipeline fetch ..."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    records = []
    for m in tqdm(meta, desc="build-docs"):
        p = Path(m["local_path"])
        if not p.exists():
            continue

        content = p.read_bytes()
        suffix = p.suffix.lower()

        if suffix in {".htm", ".html"}:
            text = html_to_text(content.decode("utf-8", errors="ignore"))
        else:
            # NOTE: Many SEC primary docs are .txt but contain HTML-like text; decoding still works.
            text = content.decode("utf-8", errors="ignore")

        text = normalize_text(text)
        rec = dict(m)
        rec["text"] = text
        records.append(rec)

    out_jsonl = settings.dataset_dir / "edgar_docs.jsonl"
    out_parquet = settings.dataset_dir / "edgar_docs.parquet"

    with out_jsonl.open("w", encoding="utf-8") as fw:
        for r in records:
            fw.write(json.dumps(r, ensure_ascii=False) + "\n")

    pd.DataFrame(records).to_parquet(out_parquet, index=False)
    print(f"[build-docs] Saved: {out_jsonl}")
    print(f"[build-docs] Saved: {out_parquet}")
    print(f"[build-docs] Docs written: {len(records)}")


# -----------------------
# Stage 3: build chunks dataset from docs
# -----------------------
def stage_build_chunks(chunk_size: int = 1200, overlap: int = 200):
    docs_path = settings.dataset_dir / "edgar_docs.jsonl"
    if not docs_path.exists():
        raise FileNotFoundError(
            f"Missing {docs_path}. Run: python -m src.data.sec_edgar_pipeline build-docs"
        )

    chunk_rows = []
    with docs_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="build-chunks"):
            d = json.loads(line)
            doc_id = d["accession_number"]
            chunks = chunk_text(d["text"], chunk_size=chunk_size, overlap=overlap)
            for k, (ch, a, b) in enumerate(chunks):
                chunk_rows.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}_c{str(k).zfill(6)}",
                        "company": d["company"],
                        "ticker": d["ticker"],
                        "cik": d["cik"],
                        "form": d["form"],
                        "filing_date": d["filing_date"],
                        "year": int(d["filing_date"][:4]) if d.get("filing_date") else None,
                        "source_url": d["source_url"],
                        "local_path": d["local_path"],
                        "char_start": a,
                        "char_end": b,
                        "text": ch,
                    }
                )

    out_jsonl = settings.dataset_dir / "edgar_chunks.jsonl"
    out_parquet = settings.dataset_dir / "edgar_chunks.parquet"

    with out_jsonl.open("w", encoding="utf-8") as fw:
        for r in chunk_rows:
            fw.write(json.dumps(r, ensure_ascii=False) + "\n")

    pd.DataFrame(chunk_rows).to_parquet(out_parquet, index=False)
    print(f"[build-chunks] Saved: {out_jsonl}")
    print(f"[build-chunks] Saved: {out_parquet}")
    print(f"[build-chunks] Chunks written: {len(chunk_rows)}")


def parse_forms(s: str) -> set[str]:
    return {x.strip() for x in s.split(",") if x.strip()}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch submissions + download raw primary documents")
    p_fetch.add_argument("--cik", required=True, type=str)
    p_fetch.add_argument("--ticker", required=True, type=str)
    p_fetch.add_argument("--company", required=True, type=str)
    p_fetch.add_argument("--forms", default="10-K,10-Q,8-K", type=str)
    p_fetch.add_argument("--max-filings", default=10, type=int)

    sub.add_parser("build-docs", help="Build doc-level dataset from raw files")
    p_chunks = sub.add_parser("build-chunks", help="Build chunk-level dataset from doc dataset")
    p_chunks.add_argument("--chunk-size", default=1200, type=int)
    p_chunks.add_argument("--overlap", default=200, type=int)

    args = parser.parse_args()

    if args.cmd == "fetch":
        stage_fetch(
            cik=args.cik,
            ticker=args.ticker,
            company=args.company,
            forms=parse_forms(args.forms),
            max_filings=args.max_filings,
        )
    elif args.cmd == "build-docs":
        stage_build_docs()
    elif args.cmd == "build-chunks":
        stage_build_chunks(chunk_size=args.chunk_size, overlap=args.overlap)


if __name__ == "__main__":
    main()