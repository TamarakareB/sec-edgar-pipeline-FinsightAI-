import pandas as pd

INP = "data/top_sp500.csv"
OUT = "data/top50_sp500.csv"

df = pd.read_csv(INP)

# 1) Clean column names
df.columns = [c.strip() for c in df.columns]

# 2) Drop unnamed/empty columns (e.g., "Unnamed: 3")
df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]

# 3) Fix weird company column name variations
if "company _name" in df.columns:
    df = df.rename(columns={"company _name": "company_name"})
elif "company_name" not in df.columns and "company-name" in df.columns:
    df = df.rename(columns={"company-name": "company_name"})
elif "company_name" not in df.columns:
    raise ValueError(f"Could not find company column. Columns: {df.columns.tolist()}")

# 4) Normalize values
df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
df["company_name"] = df["company_name"].astype(str).str.strip()

# CIK: convert to string, remove .0 if present, pad to 10 digits
df["cik"] = (
    df["cik"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
    .str.strip()
    .str.zfill(10)
)

# 5) Keep only required columns
df = df[["ticker", "company_name", "cik"]]

df.to_csv(OUT, index=False)
print(f"✅ Cleaned CSV saved to {OUT} with {len(df)} rows")