# Setup Guide

This document explains how to set up the SEC-EDGAR-Pipeline project locally.

---

## 1. Environment Requirements

- Python 3.10+
- pip / virtualenv
- (Recommended) VS Code

---

## 2. Clone the Repository

```bash
git clone <your-repo-url>
cd sec-edgar-pipeline

```

---


## 3.Create Virtual Environment

```bash
python -m venv .venv

```
### Activate

-Windows

```bash
.venv\Scripts\activate

```

-Mac/Linux

```bash
source .venv/bin/activate

```
---

## 4.Install Dependencies

```bash
pip install -r requirements.txt

```
---

## 5.Environment Variables (IMPORTANT)
Create a .env file in the project root:

```bash
touch .env
```

Add the following:

# =========================
# Gemini API (REQUIRED)
# =========================
GEMINI_API_KEY=your_gemini_api_key_here

# =========================
# SEC EDGAR Contact Info (REQUIRED)
# =========================
SEC_EDGAR_EMAIL=your_email@example.com

---

### 🔑 Gemini API Key

Get it from:
👉 https://ai.google.dev/

Required for:
QA pipeline
Risk classification (if using LLM)
Any LLM-based extraction

### 📩 SEC EDGAR Email Requirement

SEC requires a valid user agent when accessing EDGAR data.

We use:

your_email@example.com

### Why this matters:

- Prevents request blocking
- Required by SEC fair access policy
- Ensures compliance

## 6. Initial Data Preparation

Before running analytics, you must have:

data/dataset/financial_metrics.jsonl

If not available, run:

```bash
python -m src.run_data_pipeline
```

---

## 7.Build RAG Index 

```bash
python -m src.run_rag_pipeline all
```

---

## 8. Load SQL Database
```bash
python -m src.sql_database
```

This creates:

data/sql/finsightai.db

## 9. Verify Outputs

Check:

```
data/
├── dataset/
├── sql/
└── outputs/   (if configured)
```

---

## 10. Common Issues
### ❗ Missing API Key

Error:

GEMINI_API_KEY not found

Fix:

Add it to .env

### ❗ EDGAR Request Blocked

Fix:

Ensure SEC_EDGAR_EMAIL is set
Use a valid email

### ❗ Path Errors

Fix:

Always run from project root:
python -m src.run_analytics_pipeline all
---

## 11. Optional Tools

SQLite Viewer

👉 https://sqlitebrowser.org

Open:

data/sql/finsightai.db

---

Setup Complete 🎉

You are now ready to run the full pipeline.