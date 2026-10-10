# Munasib Khareed (مناسب خرید)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: Academic](https://img.shields.io/badge/license-Academic%20FYP-green.svg)](#)

> **AI-Driven E-Commerce Decision-Support System (Virtual Consumer Advocate)**  
> Final Year Project — FAST National University of Computer and Emerging Sciences (FAST-NUCES), Karachi.

---

## 📌 Project Overview

**Munasib Khareed** is an intelligent decision-support system designed to empower online shoppers. By analyzing product metadata and customer reviews from the **UCSD Amazon Review Data (2018)**, it transforms unstructured user feedback into transparent, evidence-grounded shopping recommendations.

### Key Capabilities
- **Product Profile & Intelligence**: Comprehensive metadata normalization, verified purchase ratio, and sales rank tracking.
- **Review Intelligence**:
  - DistilBERT-based sentiment classification (weak supervision baseline from star ratings: 1–2 negative, 3 neutral, 4–5 positive).
  - Aspect-based sentiment analysis and attribute extraction (spaCy + domain heuristics).
  - Evidence-strength indicator based on review length, verified purchase status, and helpfulness metrics.
- **Consumer Risk Assessment (0–100)**: Transparent fuzzy inference system evaluating risk across sentiment polarity, negative aspect intensity, and evidence volatility.
- **Multi-Criteria Decision Analysis (TOPSIS)**: Vector-normalized TOPSIS ranking evaluating the target product against viable alternatives and sampled category peers.
- **Explainable AI (XAI)**: SHAP tree explainer on a surrogate regressor model, reporting feature attribution and surrogate fidelity.
- **Evidence-Based Alternatives**: Discovery of higher-confidence, lower-risk candidate substitutes within the same product category.
- **Final Verdict Engine**: Structured, rule-enforced verdicts: `CONSIDER`, `CONSIDER WITH CAUTION`, or `CONSUMER RISK` (products with weak evidence cannot receive a clean `CONSIDER`).
- **Interactive UI**: Clean, dual-view Streamlit dashboard backed strictly by FastAPI REST services.

---

## 🏗️ Layered Architecture

Munasib Khareed follows a strictly decoupled, layered architecture:

```text
Data Layer (Raw/Interim/Processed Parquet)
   │
   ▼
Database Cache (SQLAlchemy 2.0 + SQLite serving cache)
   │
   ▼
Domain Modules (product_intel, review_intel, risk, decision, xai, alternatives, verdict)
   │
   ▼
Pipeline Orchestrator (src/munasib/pipeline)
   │
   ▼
FastAPI REST API (src/munasib/api)
   │
   ▼
Streamlit Dashboard (app/) [UI never calls ML modules directly; only REST API]
```

---

## 📂 Repository Structure

```text
Munasib-Khareed/
├── configs/                  # Externalized configuration files (.yaml)
├── data/
│   ├── raw/                  # Original UCSD Amazon Review datasets (git-ignored)
│   ├── interim/              # Intermediate transformations (git-ignored)
│   ├── processed/            # Cleaned Parquet partitions (git-ignored)
│   └── db/                   # SQLite serving database (git-ignored)
├── models/                   # Serialized model weights & checkpoints (git-ignored)
├── notebooks/                # Exploratory data analysis & prototyping notebooks
├── scripts/                  # Data downloading, ETL, training, and evaluation scripts
├── src/munasib/              # Core application package
│   ├── core/                 # Shared configurations, logging, and constants
│   ├── data/                 # Ingestion, schema validation, and preprocessing
│   ├── db/                   # SQLAlchemy ORM models, session management
│   ├── product_intel/        # Product metadata extraction and profiling
│   ├── review_intel/         # Sentiment, aspect extraction, and evidence scoring
│   │   ├── sentiment/        # DistilBERT sentiment modeling
│   │   ├── aspects/          # Aspect extraction & aspect-level polarity
│   │   └── evidence/         # Credibility & evidence-strength scoring
│   ├── risk/                 # Scikit-fuzzy consumer risk assessment (0-100)
│   ├── decision/             # Custom NumPy TOPSIS multi-criteria ranking
│   ├── xai/                  # Surrogate regression & SHAP explainability
│   ├── alternatives/         # Peer sampling and alternative recommendation
│   ├── verdict/              # Rule-enforced final verdict synthesis
│   ├── pipeline/             # Pipeline orchestration engine
│   └── api/                  # FastAPI web server
│       ├── routers/          # API route definitions
│       └── schemas/          # Pydantic v2 data models
├── app/                      # Streamlit frontend application
│   ├── components/           # Reusable UI widgets and Plotly charts
│   └── pages/                # Multi-page dashboard views
├── tests/                    # Automated test suite
│   ├── unit/                 # Fast, isolated unit tests
│   ├── integration/          # Database, pipeline, and API integration tests
│   ├── e2e/                  # Full-flow end-to-end scenarios
│   └── fixtures/             # Shared pytest mocks and sample data
├── reports/
│   ├── figures/              # Generated plots, diagrams, and figures
│   └── evaluation/           # Performance tables and metric artifacts
└── docs/                     # Project documentation and proposal PDF
    ├── proposal.pdf          # Approved FYP project proposal
    └── contributing.md       # Team workflow & branch conventions
```

---

## 🛠️ Tech Stack

- **Language & Runtime**: Python 3.11
- **Deep Learning & NLP**: PyTorch, Hugging Face Transformers (`distilbert-base-uncased`), spaCy
- **Machine Learning & Stats**: scikit-learn, NumPy, pandas, PyArrow, SHAP
- **Fuzzy Logic & MCDM**: scikit-fuzzy, Custom Vector-Normalized TOPSIS
- **Web API**: FastAPI, Uvicorn, Pydantic v2
- **Persistence**: SQLAlchemy 2.0, SQLite
- **Dashboard**: Streamlit, Plotly
- **Quality & Testing**: pytest, ruff, mypy

---

## 👥 Team & Contribution Guidelines

Refer to [docs/contributing.md](docs/contributing.md) for team branch conventions (`main`, `dev`, and feature branches for our 3 teammates), commit style guides, and PR requirements.
