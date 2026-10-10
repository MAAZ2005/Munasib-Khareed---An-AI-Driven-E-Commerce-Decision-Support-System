# Contributing & Git Branch Conventions

This document outlines the collaborative workflow, Git branching model, and code quality standards for the **Munasib Khareed** Final Year Project (BS CS, FAST-NUCES Karachi).

---

## 🌳 Branching Strategy

To keep the codebase stable and enable smooth parallel development across our **three teammates**, we use a GitFlow-inspired branching strategy centered around `main`, `dev`, and scoped feature branches.

```text
       ┌───────────┐
       │   main    │  (Production / Evaluation Milestone Releases)
       └─────▲─────┘
             │ (Release PR / Tag)
       ┌─────┴─────┐
       │    dev    │  (Primary Integration Branch)
       └─────▲─────┘
             │
   ┌─────────┼─────────┐
   │         │         │
┌──┴──┐   ┌──┴──┐   ┌──┴──┐
│feat/│   │feat/│   │feat/│  (Individual Feature Branches per Teammate)
│ TM1 │   │ TM2 │   │ TM3 │
└─────┘   └─────┘   └─────┘
```

### 1. `main` Branch
- **Purpose**: Stable, production-ready codebase representing verified project milestones (e.g., Proposal submission, Mid-Term defense, Final defense).
- **Rules**:
  - **Direct commits are strictly prohibited**.
  - Merges into `main` occur only from `dev` after full integration testing and team sign-off.
  - Every merge to `main` must be tagged with a semantic version (e.g., `v0.1.0-scaffold`, `v0.2.0-pipeline`).

### 2. `dev` Branch
- **Purpose**: Active shared integration branch where all developed modules, pipelines, and UI views converge.
- **Rules**:
  - All feature branches branch off `dev` and merge back into `dev`.
  - Must remain in a functional state (automated test suite must pass).
  - Merging into `dev` requires a Pull Request (PR) and at least one teammate review.

### 3. Feature & Task Branches (Three Teammates)
Every task or module must be developed in an isolated branch branched off `dev`.

#### Naming Convention:
```
<type>/<teammate-id>/<short-description>
```

#### Types:
- `feat/`: New feature or module component (e.g., `feat/t1/distilbert-sentiment`)
- `fix/`: Bug fix or calculation adjustment (e.g., `fix/t2/topsis-zero-division`)
- `refactor/`: Code reorganization without behavior change (e.g., `refactor/t3/api-schemas`)
- `test/`: Adding or updating test suites (e.g., `test/t1/fuzzy-risk-unit-tests`)
- `docs/`: Documentation and architecture updates (e.g., `docs/t2/architecture-diagrams`)
- `chore/`: Configuration, build setup, dependencies (e.g., `chore/t3/ruff-config`)

#### Teammate Identifiers:
- `t1` or `teammate1` (e.g., Maaz)
- `t2` or `teammate2`
- `t3` or `teammate3`

#### Examples:
- `feat/t1/evidence-strength-scoring`
- `feat/t2/fuzzy-risk-inference-engine`
- `feat/t3/streamlit-product-intelligence-view`
- `fix/t1/rating-weak-supervision-labels`

---

## 🔄 Daily Development Workflow

### Step 1: Sync with `dev`
Before starting work on any feature, ensure local `dev` is up to date:
```bash
git checkout dev
git pull origin dev
```

### Step 2: Create a Feature Branch
```bash
git checkout -b feat/t1/aspect-extraction-spacy
```

### Step 3: Implement & Verify Locally
- Follow the layered architecture: `data -> db -> module -> pipeline -> api -> app`.
- Follow locked design rules (document weak supervision for sentiment, SHAP fidelity on surrogate regressor, TOPSIS category peer sampling).
- Keep all parameters, weights, and thresholds in `configs/*.yaml`.
- Include docstrings and type annotations.
- Run tests locally:
  ```bash
  pytest
  ruff check .
  mypy src/
  ```
- Tests **must never require network access or a dedicated GPU**.

### Step 4: Commit Your Work
Write concise, descriptive commit messages:
```bash
git commit -m "feat(aspects): implement spacy noun-chunk aspect extraction with frequency ranking"
```

### Step 5: Push and Open a Pull Request
```bash
git push -u origin feat/t1/aspect-extraction-spacy
```
Open a PR on GitHub targeting the `dev` branch. Outline:
1. Summary of changes.
2. Verification commands run and test outputs.
3. Any required configuration or manual steps.

### Step 6: Code Review & Merge
- At least one teammate must review the PR.
- Once approved and checks pass, merge into `dev` using **Squash & Merge** or **Rebase & Merge** to keep the Git history clean.
- Delete the feature branch after merging.

---

## 🔒 Working Rules & Guardrails
1. **Never commit raw data, processed datasets, or model weights**: Ensure `data/` and `models/` contents remain git-ignored (retaining only `.gitkeep`).
2. **Never commit `.env` or secrets**: Keep environment configurations local.
3. **No fabricated data or stubbed results**: Every narrative report and dashboard metric must link to computed pipeline outputs.
4. **Preserve `docs/proposal.pdf`**: The approved proposal is version-controlled and remains the definitive reference for project scope and methodology.
