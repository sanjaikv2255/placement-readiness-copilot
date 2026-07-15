# Placement Readiness Copilot — Module 1

## Resume & Skill-Gap Scoring Engine

**Stack:** XGBoost, MLflow, Python 3.10

**Status:** Environment setup complete (Day 1)

## Structure
- `data/raw` — synthetic resumes & JDs (JSON)
- `data/processed` — cleaned/feature-engineered data
- `notebooks` — EDA & experimentation
- `src` — core modules (scoring logic, feature extraction)
- `models` — saved XGBoost models
- `mlruns` — MLflow tracking

## Setup
\`\`\`
conda activate placement-module1
pip install -r requirements.txt
\`\`\`