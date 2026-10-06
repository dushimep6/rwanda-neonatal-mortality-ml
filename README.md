# Predicting neonatal mortality in Rwanda: logistic regression versus machine learning (2025 DHS)

Analysis code for the manuscript **"Predicting neonatal mortality in Rwanda using the 2025 Demographic and Health Survey: a comparison of logistic regression and machine learning algorithms"** (Dushimimana P, Sumwiza K, Uwamahoro R, Twizere C, Tumusiime DK, Rushingabigwi G, Mugisha E). The manuscript is in preparation for journal submission.

## What this code does

1. **Stata** builds the analytic samples from the 2025 Rwanda Demographic and Health Survey (DHS) Births Recode file, defines neonatal death (death before 28 days), and fits the survey-adjusted logistic regressions (Tables 1, 2 and 4 of the paper).
2. **Python** compares logistic regression, random forest, XGBoost, LightGBM and CatBoost under five-fold stratified cross-validation, across five class-imbalance conditions. It adds out-of-fold SHAP values, a nested analysis of how the condition was selected, subgroup performance by infant sex and household wealth, and calibration.

Model 1 is the primary analysis (births in the five years before the survey). Model 2 is a sensitivity analysis (births in the three years before the survey, with three added care-related predictors).

## What is not included

- **The DHS data.** The DHS licence does not allow redistribution, so no survey data are in this repository and none should be added. See `data/README.md` for how to obtain the file.
- **Trained models.** The scripts retrain everything from the data.

## Repository layout

```
stata/
  01_model1_regression_and_export.do       Model 1 sample, regression, AUC, export for Python
  02_model2_regression_and_export.do       Model 2 sample, exclusions, regression, AUC, export for Python
  03_optional_antenatal_breastfeeding_missingness.do   optional snippet (see file header)
python/
  01_model1_ml_pipeline.py                 Model 1 machine learning, nested selection, subgroups, calibration
  02_model2_ml_pipeline.py                 Model 2 machine learning
data/                                      put the DHS file here (not tracked by git)
results/                                   outputs are written here (not tracked by git)
requirements.txt                           Python packages
```

## How to run

1. Obtain `RWBR91FL.dta` and place it in `data/` (see `data/README.md`).
2. In Stata (version 18 was used), first set the working folder to the repository (`cd` followed by its path), then run
   `do stata/01_model1_regression_and_export.do` and `do stata/02_model2_regression_and_export.do`.
   By default each do-file looks for `RWBR91FL.dta` in the `data/` folder of the repository (so `cd` to the repository folder first). If your data are elsewhere, do not edit the do-file: type `global datadir "C:/full/path/to/folder"` in the Stata Command window (the folder, not the file name), then run the whole do-file with `do`. Run the whole file, not a selection of lines. The do-file stops with a clear message if the survey file is not found. Each do-file writes an analytic file (`model1_analytic_data_improved.dta`, `model2_analytic_data_improved.dta`) into that folder. Regression tables and AUC values are printed in the Results window; use `log using` if you want a saved log.
3. Install the Python packages: `pip install -r requirements.txt`.
4. From the repository root, run `python python/01_model1_ml_pipeline.py` and `python python/02_model2_ml_pipeline.py`. Inputs are read from `data/` and outputs written to `results/model1/` and `results/model2/`. To use other folders, set the environment variables `DATA_DIR` and `OUT_DIR`.

The Python pipelines are computationally heavy: they tune five algorithms under five conditions in each of five folds, including 30 Bayesian-optimisation (Optuna) trials per algorithm per fold. Expect a long run.

## Which file produces which result in the paper

**Python, Model 1 (`results/model1/`)**

| Output | Used for |
|---|---|
| `five_condition_comparison_revised.csv` | Table 3 (ClassWeight_Strong rows); Supplementary Table S1 (all rows) |
| `nested_selection_model1.csv` | Table 3 (nested AUC column); Supplementary Table S3 |
| `brier_scores_revised.csv` | Table 3 (Brier score column) |
| `fig1_study_flow_revised.png` | Figure 1 |
| `fig2_roc_curves_revised.png` | Figure 2 |
| `fig9_overfitting_check_revised.png`, `overfitting_check_revised.csv` | Figure 3 |
| `fig_shap_bar_revised.png`, `shap_importance_rf_oof_revised.csv` | Figure 4 |
| `fig_shap_beeswarm_revised.png` | Figure 5 |
| `fig7_calibration_revised.png` | Supplementary Figure S3 |
| `fig_fairness_combined_model1.png`, `fairness_by_sex_model1.csv`, `fairness_by_wealth_model1.csv` | Supplementary Figure S4 |
| `calibration_overall_model1.csv`, `calibration_by_subgroup_model1.csv` | Calibration results in the Results section |
| `fig8_confusion_matrix_revised.png` | Confusion-matrix counts quoted in the Results (threshold 0.5) |

**Python, Model 2 (`results/model2/`)**

| Output | Used for |
|---|---|
| `five_condition_comparison_model2_revised.csv` | Table 5 (ClassWeight_Strong rows); Supplementary Table S2 (all rows) |
| `fig1_study_flow_model2_revised.png` | Supplementary Figure S1 |
| `fig2_roc_curves_model2_revised.png` | Supplementary Figure S2 |

The other files in each folder (Brier scores, calibration, overfitting, confusion matrix and SHAP outputs for Model 2) are additional outputs not reported in the paper.

**Stata (printed in the Results window)**

| Output | Used for |
|---|---|
| Section 3 of the Model 1 do-file (`svy: tab neonatal_death, percent ci`) | Weighted prevalence (1.7%) checked against the official rate |
| Table 1 sections (`svy: tab ..., count` and `col`) | Table 1 |
| Final `svy: logistic` and `roctab` | Tables 2 and 4; regression AUC |
| Model 2 do-file, complete-case loop (Section 5) | Supplementary Table S4 (new exclusions at each step) |

## Reproducibility notes

- All random seeds are fixed (`RANDOM_STATE = 42`), and rerunning the Model 1 pipeline reproduced the reported values.
- Results can differ slightly with other package versions. To record your exact versions, run `pip freeze > requirements-lock.txt`.
- The machine learning models do not use the DHS sampling weights during training. This is stated in the paper. The regression models do account for the survey design.
- Only paths were changed from the versions used to produce the paper: Windows folder paths were replaced by the `DATA_DIR`, `OUT_DIR` and `datadir` settings so the code runs on any system. The analysis code is unchanged.

## Licence and citation

The code is released under the MIT licence (see `LICENSE`). The licence covers the code only, not the DHS data. If you use this code, please cite the paper (see `CITATION.cff`). The code is archived on Zenodo: [10.5281/zenodo.23191383](https://doi.org/10.5281/zenodo.23191383) (this DOI always resolves to the latest version).

## Contact

Pasteur Dushimimana (corresponding author), dushimep6@gmail.com.




