TEAM_CONTRIBUTIONS.md

# Team Contributions

This project was developed collaboratively as part of BANA 275.  
Each team member was responsible for a key component of the pipeline, covering data engineering, NLP, modeling, and evaluation.

---

## Tamarakare Chinonso Edwin-Biayeibo: : Data Collection & Preprocessing Lead

**Responsibilities**
- Defined the Top 50 S&P 500 companies based on market capitalization
- Collected SEC EDGAR filings (primarily 10-K) for the past 5 years
- Stored raw filings (HTML/PDF/text format)
- Converted filings into clean text
- Removed noise such as tables, headers, and duplicated content
- Chunked documents for downstream RAG pipeline
- Generated metadata:
  - ticker
  - filing date
  - fiscal year
  - section labels (e.g., Risk Factors, MD&A)
- Delivered a clean and structured dataset to the team

**Deliverables**
- Clean dataset (JSONL / Parquet)
- Data pipeline scripts
- Dataset documentation
- Summary statistics

---

## Vishal Srivastava: RAG & Retrieval System Lead

**Responsibilities**
- Built embedding pipeline for document chunks
- Implemented vector database (FAISS / Chroma)
- Developed retrieval system for relevant section search
- Tuned chunk size and top-k retrieval
- Implemented citation tracking for retrieved content
- Optimized retrieval performance
- Integrated retrieval system with QA pipeline

**Deliverables**
- Vector database
- Retrieval system code (indexing + search)
- Retrieval evaluation results (Recall@k, MRR)

---

## Qihua Liu: QA System & Risk Classification Lead

**Responsibilities**
- Developed LLM-based question answering pipeline
- Implemented grounded answer generation with citations
- Built risk classification system:
  - regulatory risk
  - operational risk
  - financial risk
  - market/liquidity risk
- Reduced hallucinations using retrieval grounding
- Connected retrieval → LLM → final answer pipeline

**Deliverables**
- QA pipeline
- Risk classification module
- Example queries and outputs

---

## Gema Zhu: Structured Data Extraction & SQL Database Lead

**Responsibilities**
- Defined key financial metrics (e.g., revenue, net income, assets)
- Extracted structured values using regex/pattern matching
- Built SQL database to store financial data
- Designed database schema
- Linked extracted values to source document evidence
- Implemented validation for extracted data

**Deliverables**
- SQL database schema
- Financial extraction pipeline
- Populated database
- Extraction validation results

---

## Ruofan Yang: Evaluation, Testing & Integration Lead

**Responsibilities**
- Designed evaluation framework for the pipeline
- Created labeled test dataset
- Evaluated system performance:
  - Recall@k (retrieval)
  - groundedness (QA)
  - hallucination rate
  - classification accuracy / F1
- Conducted experiments and compared model settings
- Integrated full pipeline components
- Supported final report and presentation

**Deliverables**
- Evaluation framework
- Performance metrics and experiment results
- Final integrated pipeline

---

## 🤝 Team Collaboration

- Collaborated on system design and architecture
- Participated in debugging, testing, and integration
- Held regular meetings to align progress
- Contributed to documentation and final deliverables

---

## 📌 Summary

This project represents a full end-to-end system combining:
- data engineering
- natural language processing
- retrieval-augmented generation (RAG)
- structured data extraction
- evaluation and experimentation

Each team member contributed to building a complete AI-powered financial document analysis pipeline.