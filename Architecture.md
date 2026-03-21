# 1 Architecture Overview

FinSightAI is an end-to-end financial intelligence system that integrates SEC EDGAR data ingestion, NLP-based analytics, RAG-powered QA, and an interactive Streamlit UI.

# 2 System Architecture

```
                                ┌────────────────────────────┐
                                │        Streamlit UI        │
                                │         (app.py)           │
                                └────────────┬───────────────┘
                                             │
        ┌────────────────────────────────────┼────────────────────────────────────┐
        │                                    │                                    │
        ▼                                    ▼                                    ▼
┌──────────────────┐             ┌────────────────────┐              ┌────────────────────┐
│   QA System      │             │ Analytics Layer     │              │ SQL Explorer / UI  │
│ (RAG + Gemini)   │             │ (Batch Pipeline)    │              │                    │
└───────┬──────────┘             └─────────┬──────────┘              └─────────┬──────────┘
        │                                  │                                   │
        ▼                                  ▼                                   ▼
┌──────────────────┐        ┌────────────────────────────────────┐     ┌────────────────────┐
│ Retrieval Engine │        │  Financial Extractor               │     │ SQLite Database     │
│ (FAISS + Embeds) │        │  Risk Classifier                   │     │ (finsightai.db)     │
└───────┬──────────┘        │  Compare Engine                    │     └────────────────────┘
        │                   │  Trend Analyzer                    │
        ▼                   └────────────────────────────────────┘
┌────────────────────────────┐
│ Embeddings + FAISS Index   │
│ data/index/                │
└────────────┬───────────────┘
             ▼
┌────────────────────────────┐
│ Processed Dataset          │
│ (chunks / metrics)         │
│ data/dataset/              │
└────────────┬───────────────┘
             ▼
┌────────────────────────────┐
│ SEC EDGAR Data             │
│ (raw filings)              │
│ data/raw/                  │
└────────────────────────────┘

```

# 3 Architecture Layers

1. Data Ingestion Layer

The Data Ingestion Layer is responsible for collecting and preprocessing raw SEC EDGAR filings.

It performs:

Retrieval of SEC filings via EDGAR
Cleaning and normalization of filing text
Document segmentation and chunking for downstream processing

Outputs:

edgar_docs.jsonl (cleaned filings)
edgar_chunks.jsonl (chunked text for NLP/RAG)
financial_metrics.jsonl (structured financial extraction results)

This layer ensures that unstructured regulatory filings are transformed into machine-readable formats suitable for both analytics and retrieval.

2. Data Storage Layer

The Data Storage Layer provides structured persistence for processed data.

It includes:

A SQLite database (finsightai.db)
A normalized table (financial_metrics) for financial indicators
Indexing on key fields such as ticker and year

Purpose:

Enable fast querying and aggregation
Serve as a reliable backend for analytics and UI components
Provide reproducibility across pipeline runs

This layer acts as the central source of truth for structured financial data.

3. RAG (Retrieval-Augmented Generation) Layer

The RAG Layer enables semantic search and context-aware question answering.

It consists of:

Text embeddings (Sentence Transformers)
FAISS vector index for similarity search
Retrieval logic (Top-K relevant chunks)

Workflow:

User query is embedded
Relevant document chunks are retrieved
Retrieved context is passed to the LLM

This layer bridges unstructured SEC text with LLM reasoning, enabling accurate and explainable answers.

4. Analytics Layer

The Analytics Layer is a batch-processing system that generates structured insights from SEC data.

It follows a sequential pipeline:

Financial Extraction → Risk Classification → Company Comparison → Trend Analysis

Modules include:

Financial Extractor: extracts key financial metrics from filings
Risk Classifier: identifies and categorizes risk factors
Compare Engine: analyzes differences across companies
Trend Analyzer: evaluates multi-year financial and risk trends

Characteristics:

Precomputed (non-real-time)
Deterministic and reproducible
Designed for UI consumption

This layer ensures that complex analytics are computed once and reused efficiently.

5. QA (Question Answering) Layer

The QA Layer provides real-time, natural language interaction with SEC data.

It integrates:

RAG retrieval outputs
Gemini LLM for reasoning and response generation

Capabilities:

Risk explanation
Strategy interpretation
Financial insights in natural language

Unlike the Analytics Layer, this layer is query-driven and operates dynamically at runtime.

6. Application Layer (UI)

The Application Layer is implemented using Streamlit (app.py) and serves as the user-facing interface.

Features include:

Natural language query input
Predefined analytical prompts
Company and year filters
Visualization of results
Confidence and source tracking

The UI does not perform heavy computation directly; instead, it consumes outputs from:

Analytics pipeline (precomputed data)
RAG + QA system (real-time responses)

This separation improves performance and maintainability.

7. Pipeline Orchestration Layer

The Pipeline Layer coordinates the execution of different system components.

It includes:

Data Pipeline
```bash
python -m src.run_data_pipeline
```
RAG Pipeline
```bash
python -m src.run_rag_pipeline all
```
Analytics Pipeline
```bash
python -m src.run_analytics_pipeline all
```

Responsibilities:

Enforce execution order
Prevent missing dependencies
Enable reproducibility
Support partial and full pipeline runs

This layer ensures that all system components are properly synchronized.

8. External Services Layer

The system depends on two external services:

SEC EDGAR
Source of all financial filings
Requires a valid user email (SEC_EDGAR_EMAIL)
Enforces fair access policies
Gemini API
Provides LLM capabilities
Used for QA and advanced text reasoning
Requires API key (GEMINI_API_KEY)

These services extend the system beyond local computation and enable intelligent analysis.