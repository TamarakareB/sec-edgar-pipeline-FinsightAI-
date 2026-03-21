import json
import sqlite3
from pathlib import Path


# =============================================================================
# Paths
# =============================================================================

# This file is expected to live at: project_root/src/sql_database.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SQL_DIR = DATA_DIR / "sql"
DATASET_DIR = DATA_DIR / "dataset"

DB_PATH = SQL_DIR / "finsightai.db"
METRICS_PATH = DATASET_DIR / "financial_metrics.jsonl"

SQL_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Config
# =============================================================================

FIELDS = [
    "total_revenue",
    "net_income",
    "gross_profit",
    "operating_income",
    "eps_basic",
    "eps_diluted",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "long_term_debt",
    "cash_and_equivalents",
    "operating_cash_flow",
    "capital_expenditures",
    "research_and_development",
    "dividend_per_share",
]


# =============================================================================
# Formatting helpers
# =============================================================================

def fmt_money(x):
    """Format money-like numeric values safely."""
    if x is None:
        return "N/A"
    try:
        return f"${x:,.0f}M"
    except (TypeError, ValueError):
        return f"{x}"


def fmt_num(x, decimals=2):
    """Format a generic numeric value safely."""
    if x is None:
        return "N/A"
    try:
        return f"{x:,.{decimals}f}"
    except (TypeError, ValueError):
        return f"{x}"


# =============================================================================
# DB setup
# =============================================================================

def create_tables(conn):
    cur = conn.cursor()

    field_cols = ",\n        ".join([f"{field} REAL" for field in FIELDS])

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS financial_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            company TEXT,
            form TEXT,
            filing_date TEXT,
            year INTEGER,
            accession TEXT,
            {field_cols},
            fields_found INTEGER,
            confidence REAL,
            UNIQUE(ticker, filing_date, form)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            ticker TEXT PRIMARY KEY,
            company TEXT,
            sector TEXT
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticker
        ON financial_metrics(ticker)
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_year
        ON financial_metrics(year)
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_form
        ON financial_metrics(form)
    """)

    conn.commit()
    print("Tables created.")


# =============================================================================
# Load data
# =============================================================================

def load_metrics(conn):
    if not METRICS_PATH.exists():
        print(f"[ERROR] Metrics file not found: {METRICS_PATH}")
        print("Run financial_extractor.py first.")
        return 0

    cur = conn.cursor()

    cols = ", ".join(FIELDS)
    val_placeholders = ", ".join(["?" for _ in FIELDS])

    loaded = 0
    skipped = 0

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                skipped += 1
                print(f"[WARN] Line {line_num}: invalid JSON, skipped. Error: {e}")
                continue

            field_vals = [record.get(field) for field in FIELDS]

            try:
                cur.execute(
                    f"""
                    INSERT OR REPLACE INTO financial_metrics (
                        ticker,
                        company,
                        form,
                        filing_date,
                        year,
                        accession,
                        {cols},
                        fields_found,
                        confidence
                    )
                    VALUES (?, ?, ?, ?, ?, ?, {val_placeholders}, ?, ?)
                    """,
                    [
                        record.get("ticker"),
                        record.get("company"),
                        record.get("form"),
                        record.get("filing_date"),
                        record.get("year"),
                        record.get("accession"),
                    ]
                    + field_vals
                    + [
                        record.get("fields_found"),
                        record.get("confidence"),
                    ],
                )
                loaded += 1

            except Exception as e:
                skipped += 1
                print(f"[WARN] Line {line_num}: insert skipped. Error: {e}")

    conn.commit()
    print(f"Loaded {loaded} records.")
    if skipped > 0:
        print(f"Skipped {skipped} records.")
    return loaded


# =============================================================================
# Diagnostics
# =============================================================================

def data_quality_checks(conn):
    cur = conn.cursor()

    print("\n" + "=" * 80)
    print("DATA QUALITY CHECKS")
    print("=" * 80)

    # Total records
    cur.execute("SELECT COUNT(*) FROM financial_metrics")
    total_records = cur.fetchone()[0]
    print(f"Total records in financial_metrics: {total_records}")

    # Distinct tickers
    cur.execute("SELECT COUNT(DISTINCT ticker) FROM financial_metrics")
    distinct_tickers = cur.fetchone()[0]
    print(f"Distinct tickers: {distinct_tickers}")

    # Missing key fields
    cur.execute("""
        SELECT COUNT(*)
        FROM financial_metrics
        WHERE total_revenue IS NULL
    """)
    missing_revenue = cur.fetchone()[0]
    print(f"Rows with missing total_revenue: {missing_revenue}")

    cur.execute("""
        SELECT COUNT(*)
        FROM financial_metrics
        WHERE net_income IS NULL
    """)
    missing_net_income = cur.fetchone()[0]
    print(f"Rows with missing net_income: {missing_net_income}")

    # Year range
    cur.execute("""
        SELECT MIN(year), MAX(year)
        FROM financial_metrics
        WHERE year IS NOT NULL
    """)
    min_year, max_year = cur.fetchone()
    print(f"Year range: {min_year} to {max_year}")

    # Suspiciously large revenues
    cur.execute("""
        SELECT ticker, year, total_revenue
        FROM financial_metrics
        WHERE total_revenue IS NOT NULL
          AND total_revenue > 1000000
        ORDER BY total_revenue DESC
        LIMIT 10
    """)
    suspicious = cur.fetchall()

    if suspicious:
        print("\n[WARN] Potentially suspicious total_revenue values (> 1,000,000):")
        for row in suspicious:
            print(f"  {row[0]} ({row[1]}): {fmt_money(row[2])}")
        print("  Check unit normalization in financial_extractor.py.")
    else:
        print("\nNo obviously suspicious total_revenue values found under current threshold.")


# =============================================================================
# Sample queries
# =============================================================================

def sample_queries(conn):
    cur = conn.cursor()

    print("\n" + "=" * 80)
    print("SAMPLE QUERIES")
    print("=" * 80)

    print("\nTop 5 by Revenue (10-K only, non-null revenue):")
    cur.execute("""
        SELECT ticker, year, total_revenue
        FROM financial_metrics
        WHERE form = '10-K'
          AND total_revenue IS NOT NULL
        ORDER BY total_revenue DESC
        LIMIT 5
    """)
    rows = cur.fetchall()

    if not rows:
        print("  No rows found.")
    else:
        for row in rows:
            print(f"  {row[0]} ({row[1]}): {fmt_money(row[2])}")

    print("\nApple metrics over time:")
    cur.execute("""
        SELECT year, total_revenue, net_income, eps_diluted
        FROM financial_metrics
        WHERE ticker = 'AAPL'
          AND form = '10-K'
        ORDER BY year
    """)
    rows = cur.fetchall()

    if not rows:
        print("  No AAPL 10-K rows found.")
    else:
        for row in rows:
            year, revenue, net_income, eps_diluted = row
            print(
                f"  {year}: "
                f"Revenue={fmt_money(revenue)}  "
                f"NI={fmt_money(net_income)}  "
                f"EPS={fmt_num(eps_diluted, decimals=2)}"
            )

    print("\nMost complete records (by fields_found):")
    cur.execute("""
        SELECT ticker, year, form, fields_found, confidence
        FROM financial_metrics
        ORDER BY fields_found DESC, confidence DESC
        LIMIT 5
    """)
    rows = cur.fetchall()

    if not rows:
        print("  No rows found.")
    else:
        for row in rows:
            print(
                f"  {row[0]} ({row[1]}, {row[2]}): "
                f"fields_found={row[3]}, confidence={fmt_num(row[4], 3)}"
            )


# =============================================================================
# Main
# =============================================================================

def main():
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("DB_PATH =", DB_PATH)
    print("METRICS_PATH =", METRICS_PATH)

    conn = sqlite3.connect(DB_PATH)

    try:
        create_tables(conn)
        load_metrics(conn)
        data_quality_checks(conn)
        sample_queries(conn)
    finally:
        conn.close()

    print(f"\nDatabase saved: {DB_PATH}")
    print("You can inspect it with DB Browser for SQLite.")


if __name__ == "__main__":
    main()