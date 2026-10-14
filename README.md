# Biogas prediction with LightGBM and SHAP interaction analysis (LSINA)

Code accompanying the manuscript

> Pugazhendi P., Panneerselvam A., Vasanthakumari O., Murali D.
> *Machine learning prediction of biogas production from multi-feedstock anaerobic digestion based on LightGBM and SHAP interaction analysis.*
> Submitted to *Biomass Conversion and Biorefinery* (revision, 2026).

The scripts reproduce every table and figure of the manuscript and its Supplementary Information from the public dataset.

## Data

The dataset is **not redistributed** in this repository. Download `biogas_dataset.csv` from Kaggle:

https://www.kaggle.com/datasets/ucimachinelearning/indian-biogas-production-dataset

and place it at `data/biogas_dataset.csv` (15,298 records, 22 columns).

The data provider does not document the originating facility or the measurement protocol; see the manuscript (Section 2.2 and Section 4.4) for the resulting limitations.

## Installation

Python 3.13 was used. Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

## Running the analysis

Run the scripts in order from the repository root:

| Script | Purpose | Main outputs |
|---|---|---|
| `01_audit.py` | Data audit: records per date, ranges, trend, autocorrelation, seasonality | `results/table2_descriptives.csv` (Table 2) |
| `02_models.py` | Reproduction of the originally submitted model; Optuna tuning of all models with 5-fold expanding-window CV; test evaluation with seed repeats and date-block bootstrap CIs | `results/table4_models.csv`, `results/model_meta.json` (Tables 3–4) |
| `03_linear_check.py` | Linear regression baseline coefficients | printed coefficients (Table 5, last column) |
| `04_shap.py` | SHAP main effects; signed and absolute interaction values with bootstrap CIs; additive null (100 refits) with Benjamini–Hochberg FDR; group contributions; Kruskal–Wallis/Dunn seasonal tests | `results/table5_shap.csv`, `results/table6_interactions.csv`, `results/analysis.json` (Tables 5–7) |
| `05_figures.py` | All figures | `figures/Figure_2` … `Figure_7b`, `Figure_S1`–`S3` (PNG and PDF) |

`common.py` holds the feature lists, symbols, data loading (humidity capped at 100 %) and the date-based split.

`02_models.py` uses 30 Optuna trials per model (set the environment variable `N_TRIALS` to change it). The full pipeline takes about 45 minutes on two CPU cores. All random seeds are fixed.

## Protocol summary

- Development set: 1 Jan 2010 – 31 Dec 2021 (12,240 records, 4,383 dates). Test set: 1 Jan 2022 – 31 Dec 2024 (3,058 records, 1,096 dates). The split is made on whole dates.
- Tuning: five-fold expanding-window time-series cross-validation on whole dates within the development set; the test set is used once.
- Models: linear regression, Random Forest, XGBoost, CatBoost, LightGBM and an additive (depth-one) LightGBM.
- SHAP: TreeExplainer on the LightGBM model (seed 42); interaction values on 1,500 test records.

## Repository contents

- `results/` – tables produced by the scripts for the submitted revision (for comparison when re-running).
- `figures/` – figures as submitted; `Figure_1_LSINA_workflow.drawio` is the editable workflow diagram (Fig. 1).

## Citation

Please cite release v1.0.0 of this repository (see `CITATION.cff`) and the article once published.

## License

Code: MIT License (see `LICENSE`). The dataset remains subject to the terms of its Kaggle page.
