# Predicting Insurance Drug Spending

A machine learning project that forecasts **average spending per dosage unit** for Medicare Part D drugs, enabling proactive budget planning, cost control, and data‑driven reimbursement policies.

## Overview
- **Business goal:** help insurers and healthcare systems anticipate drug costs, identify high‑cost outliers, and support pricing/reimbursement decisions.
- **Methodology:** CRISP‑DM — business understanding → data understanding → preparation → modeling → evaluation → (optional) deployment.
- **Final model:** LightGBM, after comparing multiple regressors (Linear/Tree/KNN/Random Forest/XGBoost/LightGBM).

## Dataset
- **Source:** data.gov — *Medicare Part D Spending by Drug*.
- **Scope:** 2018–2022; **13,889** records; **46** features (spending, claims, dosage units, CAGR, outlier flags, etc.).
- **Target:** `Avg_Spnd_Per_Dsg_Unt_Wghtd_2022` (average spending per dosage unit, 2022).

## Tech Stack
Python 3.12 • pandas • numpy • scikit‑learn • XGBoost • LightGBM • matplotlib • Jupyter

## Repo Structure (suggested)
```
project-insurance-drug-spending/
├─ notebooks/                 # exploratory & modeling notebooks
├─ src/                       # reusable code (preprocessing, training)
├─ data/                      # place raw/processed data (gitignored)
├─ models/                    # saved models/metrics
├─ docs/                      # figures, screenshots
├─ environment.yml            # conda env (optional)
├─ requirements.txt           # pip deps (optional)
└─ README.md
```

> Add `data/` to `.gitignore`. Never commit secrets or large raw files.

## Getting Started

### Option A — Conda (recommended)
```bash
conda env create -f environment.yml
conda activate insurance-ml
jupyter lab
```

### Option B — Pip
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

## Usage
1) Open `notebooks/01_eda.ipynb` to explore the dataset.  
2) Run `notebooks/02_modeling.ipynb` to train and compare models.  
3) (Optional) Move reusable steps into `src/` and expose a CLI (e.g., `python -m src.train --model lightgbm`).

## Results (high level)
- LightGBM delivered the strongest predictive performance and fastest training among the evaluated models. Keep this section updated with your RMSE/R² once you run the notebooks on your machine.

## Roadmap
- [ ] Add cross‑validation and model tracking (e.g., MLflow).
- [ ] Package `src/` as an installable module.
- [ ] Create a simple Streamlit/Flask demo for what‑if analysis.

## Course / Credit (optional)
- IS 670 – Machine Learning for Business Analytics • Group project

## License
MIT — see `LICENSE`.
