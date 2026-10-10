# Munasib Khareed: Project Rules

## What this project is
"Munasib Khareed" is an AI-driven e-commerce decision-support system (a virtual consumer advocate) for a university Final Year Project (FAST NU Karachi, BS CS). It analyzes ONE category of the UCSD Amazon Review Data (2018). For a selected product it produces: product profile, review sentiment and aspects, evidence-strength indicators, a fuzzy-logic + TOPSIS consumer risk assessment (0-100), a SHAP explanation, evidence-based alternatives, and a final verdict shown in a Streamlit dashboard. The proposal is at docs/proposal.pdf; consult it when a task touches methodology.

## Stack
Python 3.11, PyTorch, Transformers (DistilBERT), spaCy, scikit-learn, pandas, NumPy, PyArrow, scikit-fuzzy, custom NumPy TOPSIS, SHAP, FastAPI, SQLAlchemy 2.0 + SQLite, Pydantic v2, Streamlit, Plotly, pytest, ruff, mypy, Docker.

## Locked design decisions (do not change without asking me)
1. Dataset: UCSD Amazon Review Data (2018) (it has verified purchase, rank, also_buy/also_view).
2. Sentiment labels are weak supervision from star ratings (1-2 negative, 3 neutral, 4-5 positive). Document this wherever metrics appear.
3. SHAP explains a surrogate regressor trained on the pipeline's risk scores, not fuzzy/TOPSIS directly. Always report the surrogate's fidelity.
4. TOPSIS ranks the selected product against its alternatives plus a sampled peer set from the same category.
5. Verdict labels are exactly: CONSIDER, CONSIDER WITH CAUTION, CONSUMER RISK. A product with very weak evidence can never get a clean CONSIDER.
6. The SQLite DB is a serving cache for the processed subset, not a product database.
7. Architecture is layered: data -> db -> module logic (product_intel, review_intel, risk, decision, xai, alternatives, verdict) -> pipeline orchestrator -> FastAPI -> Streamlit. The UI never imports ML modules; it only calls the API.

## Code rules
- Source lives in src/munasib/ (src layout); the frontend is in app/. Follow the existing structure and conventions.
- Type hints and docstrings everywhere. Formulas are documented in docstrings and docs/architecture.md.
- All thresholds, weights, paths, model names and seeds come from configs/*.yaml or settings, never hardcoded.
- No placeholders, no TODOs, no "rest of code here", no stubbed logic outside the explicitly designated StubAnalysisService.
- Handle missing data explicitly (None plus a flag); never fabricate values, scores or explanations. All narrative text must cite real computed numbers.
- Every module gets tests. Tests must not need network or GPU.

## Working rules
- Before editing, inspect the repo and read the files relevant to the task.
- After editing, run the task's verification commands and fix failures until they pass. Do not claim success without running them.
- Never invent or hardcode evaluation metrics; report only what scripts actually produce.
- If something needs a manual step (large download, GPU training, credentials), say so clearly instead of faking it.
- Do not modify files unrelated to the current task. Do not add dependencies without saying so.
- Finish each task with: a summary of changes, commands run with results, manual steps for me, then commit with the message I give.