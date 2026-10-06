# ----------------------------------------------------------------------
# 01_model1_ml_pipeline.py
# Model 1 (primary analysis): logistic regression vs random forest, XGBoost,
# LightGBM and CatBoost for neonatal mortality, Rwanda 2025 DHS, births in the
# five years before the survey.
#
# INPUT : data/model1_analytic_data_improved.dta  (written by stata/01_model1_regression_and_export.do)
# OUTPUT: tables and figures in results/model1/  (see README.md for the output map)
# RUN   : from the repository root, `python python/01_model1_ml_pipeline.py`
#         (set DATA_DIR / OUT_DIR environment variables to use other folders)
#
# Sections: 0-8 five-condition comparison, SHAP, figures; 9 nested selection of the
# modelling condition; 10 subgroup performance (sex, wealth); 11 calibration.
# ----------------------------------------------------------------------
# ======================================================================
# NEONATAL MORTALITY PREDICTION - COMPLETE PIPELINE
# Rwanda 2025 RDHS - Model 1 predictor set
# Logistic Regression vs Random Forest, XGBoost, LightGBM, CatBoost
# 5 conditions compared: class-weight/original, class-weight/strong,
# SMOTEN/original, SMOTEN/strong, Optuna-tuned/class-weight
# ======================================================================

import re
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                              precision_score, recall_score, accuracy_score, roc_curve,
                              confusion_matrix)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTEN
import optuna
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

optuna.logging.set_verbosity(optuna.logging.WARNING)
RANDOM_STATE = 42
import os
# Folders can be set with environment variables; defaults are relative to the repository root.
DATA_DIR = os.environ.get("DATA_DIR", "data")      # holds the analytic .dta file written by the Stata do-file
OUT_DIR = os.environ.get("OUT_DIR", "results/model1")  # all tables and figures are written here
os.makedirs(OUT_DIR, exist_ok=True)


# ======================================================================
# SECTION 0: LOAD AND VERIFY DATA
# ======================================================================

df = pd.read_stata(os.path.join(DATA_DIR, "model1_analytic_data_improved.dta"))
print("Shape:", df.shape)
print("\n--- Outcome variable check ---")
print(df['neonatal_death'].value_counts(dropna=False))
print("\n--- Any missing values remaining? ---")
print(df.isnull().sum())


# ======================================================================
# SECTION 1: PREPARE OUTCOME AND ONE-HOT ENCODE PREDICTORS
# ======================================================================

y = (df['neonatal_death'] == 'Neonatal death').astype(int)
print("\nOutcome distribution:")
print(y.value_counts())
print("Event rate: {:.4f}".format(y.mean()))

categorical_vars = ['bord_cat', 'b0', 'v190', 'v106', 'b4', 'v501',
                     'short_interval', 'matage_cat']

X_raw = df[categorical_vars].copy()
X_raw_cat = X_raw.copy()
for col in categorical_vars:
    X_raw_cat[col] = X_raw_cat[col].astype(str)

def sanitize_colname(col):
    col = col.replace('<', 'lt').replace('>', 'gt')
    col = re.sub(r'[\[\]]', '', col)
    return col

X = pd.get_dummies(X_raw, columns=categorical_vars, drop_first=True)
X.columns = [sanitize_colname(c) for c in X.columns]

bad_chars = [c for c in X.columns if any(ch in c for ch in ['<', '>', '[', ']'])]
print("\nOK: no problematic characters remain." if not bad_chars else f"WARNING: {bad_chars}")
print("\nEncoded feature matrix shape:", X.shape)

assert len(X) == len(y)
assert X.isnull().sum().sum() == 0
print("\nAll checks passed - X and y are aligned and complete.")


def encode_fixed(cat_df):
    """One-hot encode using categories fixed from the FULL dataset, so
    resampled (SMOTEN) and original data always produce identical columns."""
    df_enc = cat_df.copy()
    for col in categorical_vars:
        df_enc[col] = pd.Categorical(df_enc[col], categories=sorted(X_raw_cat[col].unique()))
    enc = pd.get_dummies(df_enc, columns=categorical_vars, drop_first=True)
    enc.columns = [sanitize_colname(c) for c in enc.columns]
    return enc.astype(int)


# ======================================================================
# SECTION 2: CROSS-VALIDATION SETUP
# ======================================================================

X = X.astype(int)
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
folds = list(skf.split(X, y))

print(f"\nNumber of folds: {N_SPLITS}\n")
for i, (train_idx, test_idx) in enumerate(folds):
    print(f"Fold {i+1}: train n={len(train_idx)} (events={y.iloc[train_idx].sum()}), "
          f"test n={len(test_idx)} (events={y.iloc[test_idx].sum()})")


# ======================================================================
# SECTION 3: MODEL NAMES
# ======================================================================

MODEL_NAMES = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'CatBoost']
MODEL_COLORS = {
    'Logistic Regression': '#D73027', 'Random Forest': '#4575B4',
    'XGBoost': '#1B7837', 'LightGBM': '#F46D43', 'CatBoost': '#762A83'
}


# ======================================================================
# SECTION 4: CANDIDATE PARAMETER SETS (original / strong regularization)
# ======================================================================

CANDIDATE_PARAMS_ORIG = {
    'Logistic Regression': [{'C': 0.01}, {'C': 0.1}, {'C': 1.0}, {'C': 10.0}],
    'Random Forest': [
        {'n_estimators': 200, 'max_depth': 4, 'min_samples_leaf': 5},
        {'n_estimators': 300, 'max_depth': 6, 'min_samples_leaf': 5},
        {'n_estimators': 300, 'max_depth': 6, 'min_samples_leaf': 10},
        {'n_estimators': 400, 'max_depth': 8, 'min_samples_leaf': 10},
    ],
    'XGBoost': [
        {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05},
        {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05},
        {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.1},
        {'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.03},
    ],
    'LightGBM': [
        {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05},
        {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05},
        {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.1},
        {'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.03},
    ],
    'CatBoost': [
        {'iterations': 200, 'depth': 3, 'learning_rate': 0.05},
        {'iterations': 300, 'depth': 4, 'learning_rate': 0.05},
        {'iterations': 300, 'depth': 4, 'learning_rate': 0.1},
        {'iterations': 400, 'depth': 3, 'learning_rate': 0.03},
    ],
}

CANDIDATE_PARAMS_STRONG = {
    'Logistic Regression': [{'C': 0.001}, {'C': 0.01}, {'C': 0.05}, {'C': 0.1}],
    'Random Forest': [
        {'n_estimators': 100, 'max_depth': 2, 'min_samples_leaf': 15},
        {'n_estimators': 150, 'max_depth': 3, 'min_samples_leaf': 20},
        {'n_estimators': 200, 'max_depth': 3, 'min_samples_leaf': 25},
        {'n_estimators': 100, 'max_depth': 2, 'min_samples_leaf': 30},
    ],
    'XGBoost': [
        {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.03,
         'reg_alpha': 1.0, 'reg_lambda': 5.0, 'min_child_weight': 10},
        {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.03,
         'reg_alpha': 2.0, 'reg_lambda': 8.0, 'min_child_weight': 15},
        {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.02,
         'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_weight': 20},
        {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.02,
         'reg_alpha': 1.0, 'reg_lambda': 10.0, 'min_child_weight': 10},
    ],
    'LightGBM': [
        {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.03,
         'min_child_samples': 30, 'reg_alpha': 1.0, 'reg_lambda': 5.0},
        {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.03,
         'min_child_samples': 40, 'reg_alpha': 2.0, 'reg_lambda': 8.0},
        {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.02,
         'min_child_samples': 50, 'reg_alpha': 3.0, 'reg_lambda': 10.0},
        {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.02,
         'min_child_samples': 30, 'reg_alpha': 1.0, 'reg_lambda': 10.0},
    ],
    'CatBoost': [
        {'iterations': 50, 'depth': 2, 'learning_rate': 0.03, 'l2_leaf_reg': 10.0},
        {'iterations': 100, 'depth': 2, 'learning_rate': 0.03, 'l2_leaf_reg': 15.0},
        {'iterations': 50, 'depth': 3, 'learning_rate': 0.02, 'l2_leaf_reg': 20.0},
        {'iterations': 100, 'depth': 2, 'learning_rate': 0.02, 'l2_leaf_reg': 10.0},
    ],
}


def build_model_generic(name, params, spw, use_class_weight):
    cw = 'balanced' if use_class_weight else None
    spw_eff = spw if use_class_weight else 1.0
    if name == 'Logistic Regression':
        return LogisticRegression(C=params['C'], class_weight=cw, max_iter=1000, random_state=RANDOM_STATE)
    if name == 'Random Forest':
        return RandomForestClassifier(**params, class_weight=cw, random_state=RANDOM_STATE, n_jobs=-1)
    if name == 'XGBoost':
        return XGBClassifier(**params, scale_pos_weight=spw_eff, eval_metric='logloss',
                              random_state=RANDOM_STATE, verbosity=0)
    if name == 'LightGBM':
        return LGBMClassifier(**params, class_weight=cw, random_state=RANDOM_STATE, verbose=-1)
    if name == 'CatBoost':
        cb_weight = 'Balanced' if use_class_weight else None
        return CatBoostClassifier(**params, auto_class_weights=cb_weight, verbose=0, random_state=RANDOM_STATE)


# ======================================================================
# SECTION 5: OPTUNA SEARCH SPACES (30 trials/model/fold - see note below)
# Trial count kept modest: inner validation folds have ~25 events, so
# beyond ~30 trials TPE increasingly fits validation noise rather than
# finding genuinely better configs. This is an exploratory comparison
# condition, not the primary analysis.
# ======================================================================

N_OPTUNA_TRIALS = 30

def optuna_objective(name, X_tr, y_tr, X_val, y_val, spw):
    def objective(trial):
        if name == 'Logistic Regression':
            C = trial.suggest_float('C', 1e-4, 10.0, log=True)
            model = LogisticRegression(C=C, class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
        elif name == 'Random Forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                'max_depth': trial.suggest_int('max_depth', 2, 8),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 30),
            }
            model = RandomForestClassifier(**params, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
        elif name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                'max_depth': trial.suggest_int('max_depth', 2, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            }
            model = XGBClassifier(**params, scale_pos_weight=spw, eval_metric='logloss',
                                   random_state=RANDOM_STATE, verbosity=0)
        elif name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                'max_depth': trial.suggest_int('max_depth', 2, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            }
            model = LGBMClassifier(**params, class_weight='balanced', random_state=RANDOM_STATE, verbose=-1)
        elif name == 'CatBoost':
            params = {
                'iterations': trial.suggest_int('iterations', 50, 400),
                'depth': trial.suggest_int('depth', 2, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0, log=True),
            }
            model = CatBoostClassifier(**params, auto_class_weights='Balanced', verbose=0, random_state=RANDOM_STATE)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, probs)
    return objective

def build_from_optuna_params(name, params):
    if name == 'Logistic Regression':
        return LogisticRegression(C=params['C'], class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
    if name == 'Random Forest':
        return RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                                       min_samples_leaf=params['min_samples_leaf'],
                                       class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    if name == 'XGBoost':
        return XGBClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                              learning_rate=params['learning_rate'], reg_alpha=params['reg_alpha'],
                              reg_lambda=params['reg_lambda'], min_child_weight=params['min_child_weight'],
                              scale_pos_weight=params.get('_spw', 1.0), eval_metric='logloss',
                              random_state=RANDOM_STATE, verbosity=0)
    if name == 'LightGBM':
        return LGBMClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                               learning_rate=params['learning_rate'], min_child_samples=params['min_child_samples'],
                               reg_alpha=params['reg_alpha'], reg_lambda=params['reg_lambda'],
                               class_weight='balanced', random_state=RANDOM_STATE, verbose=-1)
    if name == 'CatBoost':
        return CatBoostClassifier(iterations=params['iterations'], depth=params['depth'],
                                   learning_rate=params['learning_rate'], l2_leaf_reg=params['l2_leaf_reg'],
                                   auto_class_weights='Balanced', verbose=0, random_state=RANDOM_STATE)


# ======================================================================
# SECTION 6: RUN ALL 5 CONDITIONS ACROSS ALL 5 FOLDS
# ======================================================================

CONDITIONS = ['ClassWeight_Original', 'ClassWeight_Strong', 'SMOTEN_Original', 'SMOTEN_Strong', 'Optuna_ClassWeight']

condition_results = {cond: {name: {'auc': [], 'pr_auc': [], 'f1': [], 'precision': [],
                                     'recall': [], 'accuracy': [], 'train_auc': []} for name in MODEL_NAMES}
                      for cond in CONDITIONS}
condition_best_params = {cond: {name: [] for name in MODEL_NAMES} for cond in CONDITIONS}
condition_inner_auc = {cond: {name: [] for name in MODEL_NAMES} for cond in CONDITIONS}

# also collect pooled OOF probabilities for the best-performing condition
# (ClassWeight_Strong - lowest overfitting gap, highest test AUC) for
# ROC/calibration/confusion-matrix figures
oof_probs = {name: np.full(len(X), np.nan) for name in MODEL_NAMES}
oof_train_auc = {name: [] for name in MODEL_NAMES}

for fold_i, (train_idx, test_idx) in enumerate(folds):
    Xcat_train, Xcat_test = X_raw_cat.iloc[train_idx], X_raw_cat.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    Xcat_inner_tr, Xcat_inner_val, y_inner_tr, y_inner_val = train_test_split(
        Xcat_train, y_train, test_size=0.25, stratify=y_train, random_state=RANDOM_STATE)

    X_test_enc = encode_fixed(Xcat_test)
    X_inner_val_enc = encode_fixed(Xcat_inner_val)
    X_train_enc_orig = encode_fixed(Xcat_train)
    X_inner_tr_enc = encode_fixed(Xcat_inner_tr)
    X_train_enc = encode_fixed(Xcat_train)  # same as X_train_enc_orig, kept for clarity

    spw_inner = (y_inner_tr == 0).sum() / (y_inner_tr == 1).sum()
    spw_full_train = (y_train == 0).sum() / (y_train == 1).sum()

    print(f"\n{'='*70}\nFOLD {fold_i+1}\n{'='*70}")

    for cond in CONDITIONS:
        if cond == 'ClassWeight_Original':
            param_grid, use_smoten, use_optuna = CANDIDATE_PARAMS_ORIG, False, False
        elif cond == 'ClassWeight_Strong':
            param_grid, use_smoten, use_optuna = CANDIDATE_PARAMS_STRONG, False, False
        elif cond == 'SMOTEN_Original':
            param_grid, use_smoten, use_optuna = CANDIDATE_PARAMS_ORIG, True, False
        elif cond == 'SMOTEN_Strong':
            param_grid, use_smoten, use_optuna = CANDIDATE_PARAMS_STRONG, True, False
        elif cond == 'Optuna_ClassWeight':
            param_grid, use_smoten, use_optuna = None, False, True

        if use_smoten:
            smn = SMOTEN(random_state=RANDOM_STATE, k_neighbors=5)
            Xcat_inner_tr_fit, y_inner_tr_fit = smn.fit_resample(Xcat_inner_tr, y_inner_tr)
            X_inner_tr_fit_enc = encode_fixed(Xcat_inner_tr_fit)
        else:
            X_inner_tr_fit_enc, y_inner_tr_fit = X_inner_tr_enc, y_inner_tr

        for name in MODEL_NAMES:
            if use_optuna:
                sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
                study = optuna.create_study(direction='maximize', sampler=sampler)
                obj = optuna_objective(name, X_inner_tr_enc, y_inner_tr, X_inner_val_enc, y_inner_val, spw_inner)
                study.optimize(obj, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
                best_params = dict(study.best_params)
                best_auc = study.best_value
                if name == 'XGBoost':
                    best_params['_spw'] = spw_full_train
                final_model = build_from_optuna_params(name, best_params)
                X_train_fit_enc, y_train_fit = X_train_enc, y_train
            else:
                best_auc, best_params = -1, None
                for params in param_grid[name]:
                    model = build_model_generic(name, params, spw_inner, use_class_weight=not use_smoten)
                    model.fit(X_inner_tr_fit_enc, y_inner_tr_fit)
                    val_probs = model.predict_proba(X_inner_val_enc)[:, 1]
                    val_auc = roc_auc_score(y_inner_val, val_probs)
                    if val_auc > best_auc:
                        best_auc, best_params = val_auc, params

                if use_smoten:
                    smn_full = SMOTEN(random_state=RANDOM_STATE, k_neighbors=5)
                    Xcat_train_fit, y_train_fit = smn_full.fit_resample(Xcat_train, y_train)
                    X_train_fit_enc = encode_fixed(Xcat_train_fit)
                else:
                    X_train_fit_enc, y_train_fit = X_train_enc, y_train

                final_model = build_model_generic(name, best_params, spw_full_train, use_class_weight=not use_smoten)

            condition_best_params[cond][name].append(best_params)
            condition_inner_auc[cond][name].append(best_auc)
            final_model.fit(X_train_fit_enc, y_train_fit)

            train_probs = final_model.predict_proba(X_train_enc_orig)[:, 1]
            test_probs = final_model.predict_proba(X_test_enc)[:, 1]
            test_preds = (test_probs >= 0.5).astype(int)

            train_auc = roc_auc_score(y_train, train_probs)
            test_auc = roc_auc_score(y_test, test_probs)
            pr_auc = average_precision_score(y_test, test_probs)
            f1 = f1_score(y_test, test_preds, zero_division=0)
            prec = precision_score(y_test, test_preds, zero_division=0)
            rec = recall_score(y_test, test_preds, zero_division=0)
            acc = accuracy_score(y_test, test_preds)

            condition_results[cond][name]['auc'].append(test_auc)
            condition_results[cond][name]['pr_auc'].append(pr_auc)
            condition_results[cond][name]['f1'].append(f1)
            condition_results[cond][name]['precision'].append(prec)
            condition_results[cond][name]['recall'].append(rec)
            condition_results[cond][name]['accuracy'].append(acc)
            condition_results[cond][name]['train_auc'].append(train_auc)

            # keep ClassWeight_Strong's OOF predictions for downstream figures
            # (best-performing condition: lowest overfitting gap + highest test AUC
            # across the 5-condition comparison)
            if cond == 'ClassWeight_Strong':
                oof_probs[name][test_idx] = test_probs
                oof_train_auc[name].append(train_auc)

        print(f"  [{cond}] fold {fold_i+1} done")

print("\n\nAll 5 conditions complete across all folds.")


# ======================================================================
# SECTION 7: FINAL CONDITION COMPARISON TABLE
# ======================================================================

def compute_ci(values, confidence=0.95):
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    se = stats.sem(values)
    margin = se * stats.t.ppf((1 + confidence) / 2, n - 1) if n > 1 and se > 0 else 0
    return mean, mean - margin, mean + margin

final_condition_rows = []
for cond in CONDITIONS:
    for name in MODEL_NAMES:
        train_auc_mean = np.mean(condition_results[cond][name]['train_auc'])
        test_auc_mean, lo, hi = compute_ci(condition_results[cond][name]['auc'])
        pr_auc_mean = np.mean(condition_results[cond][name]['pr_auc'])
        recall_mean = np.mean(condition_results[cond][name]['recall'])
        accuracy_mean = np.mean(condition_results[cond][name]['accuracy'])
        gap = train_auc_mean - test_auc_mean
        final_condition_rows.append({
            'Condition': cond, 'Model': name,
            'Train_AUC': round(train_auc_mean, 3),
            'Test_AUC': round(test_auc_mean, 3),
            'Test_AUC_95CI': f"[{lo:.3f}, {hi:.3f}]",
            'Gap': round(gap, 3),
            'PR_AUC': round(pr_auc_mean, 3),
            'Recall': round(recall_mean, 3),
            'Accuracy': round(accuracy_mean, 3),
        })

final_condition_df = pd.DataFrame(final_condition_rows)
naive_baseline_accuracy = 1 - y.mean()  # always predicting "no death"
print(f"\nNAIVE BASELINE ACCURACY (always predict 'no neonatal death'): {naive_baseline_accuracy:.3f}")
print("Accuracy is reported for completeness but is NOT informative given severe class")
print("imbalance (event rate = {:.3f}) - a model with no discriminative ability at all".format(y.mean()))
print("would already score {:.1%} accuracy. AUC, PR-AUC, precision and recall are the".format(naive_baseline_accuracy))
print("primary metrics for model comparison in this study.")
print("\n" + "="*110)
print("FULL 5-CONDITION COMPARISON (all models x all conditions)")
print("="*110)
print(final_condition_df.to_string(index=False))

print("\n" + "="*100)
print("BEST CONDITION PER MODEL (lowest Gap, among Test_AUC >= 0.55)")
print("="*100)
viable = final_condition_df[final_condition_df['Test_AUC'] >= 0.55]
best_per_model = viable.loc[viable.groupby('Model')['Gap'].idxmin()]
print(best_per_model.to_string(index=False))

print("\n" + "="*100)
print("TOP 5 ROWS OVERALL BY TEST_AUC")
print("="*100)
print(final_condition_df.sort_values('Test_AUC', ascending=False).head(5).to_string(index=False))

final_condition_df.to_csv(os.path.join(OUT_DIR, "five_condition_comparison_revised.csv"), index=False)
print(f"\nSaved: five_condition_comparison_revised.csv")
print(f"\nReference: Stata logistic regression (in-sample, full data) AUC = 0.694 [0.643, 0.745]")


# ======================================================================
# SECTION 8: OUT-OF-FOLD SHAP - RANDOM FOREST (ClassWeight_Strong condition,
# the best-performing condition: lowest overfitting gap + highest test AUC)
# ======================================================================

oof_shap_rows = []
feature_names_ref = X.columns.tolist()

for fold_i, (train_idx, test_idx) in enumerate(folds):
    Xcat_train, Xcat_test = X_raw_cat.iloc[train_idx], X_raw_cat.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    X_train_enc = encode_fixed(Xcat_train)
    X_test_enc = encode_fixed(Xcat_test)

    params = condition_best_params['ClassWeight_Strong']['Random Forest'][fold_i]
    model = RandomForestClassifier(**params, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train_enc, y_train)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test_enc)
    if isinstance(sv, list):
        sv_pos = sv[1]
    elif sv.ndim == 3:
        sv_pos = sv[:, :, 1]
    else:
        sv_pos = sv

    for row_i, orig_idx in enumerate(test_idx):
        oof_shap_rows.append((orig_idx, sv_pos[row_i]))
    print(f"Fold {fold_i+1}: SHAP computed for {len(test_idx)} out-of-fold observations")

oof_shap_rows.sort(key=lambda r: r[0])
oof_shap_array = np.array([r[1] for r in oof_shap_rows])
oof_order_idx = np.array([r[0] for r in oof_shap_rows])
X_full_shap = X.iloc[oof_order_idx].reset_index(drop=True)

mean_abs_shap = np.abs(oof_shap_array).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'feature': feature_names_ref, 'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

print("\nFeature importance ranking (out-of-fold mean |SHAP|):")
print(shap_importance_df.to_string(index=False))
shap_importance_df.to_csv(os.path.join(OUT_DIR, "shap_importance_rf_oof_revised.csv"), index=False)


# ======================================================================
# SECTION 9: SHAP FIGURES (no titles)
# ======================================================================

LABEL_MAP = {
    'bord_cat_2nd-4th': 'Birth order: 2nd-4th', 'bord_cat_5th+': 'Birth order: 5th+',
    'b0_1st of multiple': 'Multiple birth: 1st of multiple',
    'b0_2nd of multiple': 'Multiple birth: 2nd of multiple',
    'b0_3rd of multiple': 'Multiple birth: 3rd of multiple',
    'v190_poorer': 'Wealth: Poorer', 'v190_middle': 'Wealth: Middle',
    'v190_richer': 'Wealth: Richer', 'v190_richest': 'Wealth: Richest',
    'v106_primary': "Mother's education: Primary", 'v106_secondary': "Mother's education: Secondary",
    'v106_higher': "Mother's education: Higher", 'b4_female': 'Sex: Female',
    'v501_married': 'Marital status: Married', 'v501_living with partner': 'Marital status: Living with partner',
    'v501_widowed': 'Marital status: Widowed', 'v501_divorced': 'Marital status: Divorced',
    'v501_no longer living together/separated': 'Marital status: Separated',
    'short_interval_Short interval (lt24mo)': 'Birth interval: Short (<24mo)',
    'matage_cat_20-34': 'Maternal age: 20-34', 'matage_cat_35+': 'Maternal age: 35+',
}
def relabel(cols):
    return [LABEL_MAP.get(c, c) for c in cols]

shap_importance_labeled = shap_importance_df.copy()
shap_importance_labeled['feature'] = shap_importance_labeled['feature'].map(lambda x: LABEL_MAP.get(x, x))
plot_data = shap_importance_labeled.sort_values('mean_abs_shap')

fig, ax = plt.subplots(figsize=(9, 8))
bars = ax.barh(plot_data['feature'], plot_data['mean_abs_shap'], color='#2166AC')
for bar, val in zip(bars, plot_data['mean_abs_shap']):
    ax.text(val + 0.0006, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9, weight='bold')
ax.set_xlabel('Mean |SHAP value| (out-of-fold)', fontsize=11)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig_shap_bar_revised.png"), dpi=300, bbox_inches='tight'); plt.close()
print("Saved: fig_shap_bar_revised.png")

X_full_shap_labeled = X_full_shap.copy()
X_full_shap_labeled.columns = relabel(X_full_shap.columns)
plt.figure()
shap.summary_plot(oof_shap_array, X_full_shap_labeled, show=False)
plt.gca().set_title('')
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig_shap_beeswarm_revised.png"), dpi=300, bbox_inches='tight'); plt.close()
print("Saved: fig_shap_beeswarm_revised.png")


# ======================================================================
# SECTION 10: REMAINING FIGURES (study flow, ROC, comparison, forest,
# calibration, confusion matrix, overfitting check) - using
# ClassWeight_Strong condition's pooled OOF predictions (best-performing
# condition: lowest overfitting gap + highest test AUC in Section 7)
# ======================================================================

# ---- Figure 1: Study flow (STROBE-style: straight main chain, elbow
# connectors to exclusion boxes) ----
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(8.5, 8.5))
ax.axis('off')
ax.set_xlim(0, 10.5)
ax.set_ylim(2.6, 11.8)

def draw_box(ax, x, y, w, h, text, facecolor):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.04,rounding_size=0.12",
                          facecolor=facecolor, edgecolor='black', linewidth=1.2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=11)

main_x = 4
main_boxes = [
    (11.0, "Total births\nRWBR91FL, n = 28,878"),
    (8.8,  "Births in 5-year recall period\n(excl. interview month), n = 7,214"),
    (6.2,  "Outcome available\nn = 7,197"),
    (3.6,  "Analytic sample (Model 1)\nn = 7,167\n(121 neonatal deaths)"),
]
box_w, box_h = 3.9, 0.85

for y_box, text in main_boxes:
    draw_box(ax, main_x, y_box, box_w, box_h, text, '#DEEBF7')

for i in range(len(main_boxes) - 1):
    y_top = main_boxes[i][0] - box_h/2
    y_bot = main_boxes[i+1][0] + box_h/2
    ax.annotate('', xy=(main_x, y_bot), xytext=(main_x, y_top),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.4))

excl_x = 7.9
excl_w, excl_h = 3.2, 0.8
exclusions = [
    (7.5, "Excluded: unknown\nage at death, n = 17"),
    (4.9, "Excluded: missing\npredictor data, n = 30"),
]
for y_box, text in exclusions:
    draw_box(ax, excl_x, y_box, excl_w, excl_h, text, '#FEE0D2')
    ax.plot([main_x, excl_x - excl_w/2], [y_box, y_box], color='black', lw=1.2)
    ax.annotate('', xy=(excl_x - excl_w/2, y_box), xytext=(main_x + 0.05, y_box),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig1_study_flow_revised.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig1_study_flow_revised.png")

# ---- Figure 2: Mean ROC curves across folds - CLEAN LINES ONLY,
# no shaded SD band. For a single-model plot, shading is standard;
# for a 5-model overlay like this, overlapping shaded bands are unusual
# and create visual clutter - the accepted convention for multi-model
# comparison plots is plain lines with AUC in the legend (uncertainty
# is already reported numerically in the comparison table/CI) ----
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(7, 7))
for name in MODEL_NAMES:
    tprs, aucs = [], []
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        y_fold = y.iloc[test_idx].values
        probs_fold = oof_probs[name][test_idx]
        fpr, tpr, _ = roc_curve(y_fold, probs_fold)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc_score(y_fold, probs_fold))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color=MODEL_COLORS[name], lw=2.2,
            label=f"{name} (AUC={mean_auc:.3f}$\\pm${std_auc:.3f})")

ax.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.legend(loc='lower right', fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig2_roc_curves_revised.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig2_roc_curves_revised.png (clean lines, no shading)")

# ---- Figure 3: Brier scores (replaces model comparison AUC bar chart) ----
from sklearn.metrics import brier_score_loss

brier_rows = []
for name in MODEL_NAMES:
    valid = ~np.isnan(oof_probs[name])
    bs = brier_score_loss(y[valid], oof_probs[name][valid])
    brier_rows.append({'Model': name, 'Brier': bs})
brier_df = pd.DataFrame(brier_rows).sort_values('Brier')  # lower = better

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(brier_df['Model'], brier_df['Brier'],
                color=[MODEL_COLORS[n] for n in brier_df['Model']], edgecolor='black')
# points-based offset (not data-value offset) so labels land tight against
# the bar end regardless of the axis's value scale - matches the SHAP bar style
for bar, val in zip(bars, brier_df['Brier']):
    ax.annotate(f'{val:.4f}', xy=(val, bar.get_y() + bar.get_height()/2),
                xytext=(4, 0), textcoords='offset points',
                va='center', ha='left', fontsize=9, weight='bold')
ax.set_xlabel('Brier score (out-of-fold, lower = better)', fontsize=11)
ax.set_xlim(0, brier_df['Brier'].max() * 1.18)  # headroom so labels never clip
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig3_brier_scores_revised.png"), dpi=300, bbox_inches='tight')
plt.close()
brier_df.to_csv(os.path.join(OUT_DIR, "brier_scores_revised.csv"), index=False)
print("Saved: fig3_brier_scores_revised.png")

# ---- Figure 7: Calibration ----
fig, axes = plt.subplots(2, 3, figsize=(14, 9)); axes = axes.flatten()
for i, name in enumerate(MODEL_NAMES):
    valid = ~np.isnan(oof_probs[name])
    frac_pos, mean_pred = calibration_curve(y[valid], oof_probs[name][valid], n_bins=10, strategy='quantile')
    axes[i].plot(mean_pred, frac_pos, marker='o', color=MODEL_COLORS[name])
    axes[i].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axes[i].set_xlabel('Mean Predicted Risk', fontsize=9); axes[i].set_ylabel('Observed Frequency', fontsize=9)
    axes[i].text(0.05, 0.9, name, transform=axes[i].transAxes, fontsize=10, weight='bold')
for j in range(len(MODEL_NAMES), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig7_calibration_revised.png"), dpi=300, bbox_inches='tight'); plt.close()
print("Saved: fig7_calibration_revised.png")

# ---- Figure 8: Confusion matrix (best model in ClassWeight_Strong by Test_AUC) ----
cond_df = final_condition_df[final_condition_df['Condition'] == 'ClassWeight_Strong']
best_model_name = cond_df.sort_values('Test_AUC', ascending=False).iloc[0]['Model']
valid = ~np.isnan(oof_probs[best_model_name])
preds_best = (oof_probs[best_model_name][valid] >= 0.5).astype(int)
cm = confusion_matrix(y[valid], preds_best)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted: Survived', 'Predicted: Neonatal death'],
            yticklabels=['Actual: Survived', 'Actual: Neonatal death'],
            annot_kws={'size': 16, 'weight': 'bold'}, ax=ax)
ax.set_ylabel('True Label', fontsize=11); ax.set_xlabel('Predicted Label', fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig8_confusion_matrix_revised.png"), dpi=300, bbox_inches='tight'); plt.close()
print(f"Saved: fig8_confusion_matrix_revised.png (best model: {best_model_name})")

# ---- Figure 9: Overfitting check ----
overfit_rows = []
for name in MODEL_NAMES:
    train_auc_mean = np.mean(oof_train_auc[name])
    test_auc_mean = cond_df.loc[cond_df['Model']==name, 'Test_AUC'].values[0]
    overfit_rows.append({'Model': name, 'Train AUC': round(train_auc_mean,3), 'Test AUC': round(test_auc_mean,3)})
overfit_df = pd.DataFrame(overfit_rows)
print("\nOverfitting check (ClassWeight_Strong condition - best performing):")
print(overfit_df.to_string(index=False))
fig, ax = plt.subplots(figsize=(9, 5))
x_pos = np.arange(len(overfit_df)); width = 0.35
ax.bar(x_pos - width/2, overfit_df['Train AUC'], width, label='Train AUC', color='#4575B4')
ax.bar(x_pos + width/2, overfit_df['Test AUC'], width, label='Test AUC', color='#D73027')
ax.set_xticks(x_pos); ax.set_xticklabels(overfit_df['Model'], rotation=20, ha='right')
ax.set_ylabel('AUC', fontsize=11); ax.set_ylim(0.4, 1.0); ax.legend()
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig9_overfitting_check_revised.png"), dpi=300, bbox_inches='tight'); plt.close()
print("Saved: fig9_overfitting_check_revised.png")
overfit_df.to_csv(os.path.join(OUT_DIR, "overfitting_check_revised.csv"), index=False)

print("\n\n" + "="*70)
print("PIPELINE COMPLETE - ALL SECTIONS RAN SUCCESSFULLY")
print("="*70)


# ======================================================================
# NESTED SELECTION OF THE MODELLING CONDITION - MODEL 1
# Paste at the VERY END of neonatal_ml_death_REVISED.py (after the
# calibration block) and run the whole script once.
#
# RULE (fixed before looking at results):
#   In each outer fold, the condition is chosen using ONLY inner-validation
#   AUC from that fold's training data (mean across the five algorithms).
#   The outer test fold is never used to choose. The AUC of the chosen
#   condition on that untouched outer test fold is then recorded.
#
# Two candidate sets are reported:
#   (a) all five conditions
#   (b) the four fixed-grid conditions (excluding Optuna), because Optuna
#       searches 30 settings against 4 for the others, so its inner AUC is
#       inflated by search alone and it would be favoured unfairly.
# ======================================================================

def nested_selection(candidates, label):
    n_folds = len(folds)
    inner_rows, chosen = [], []
    for f in range(n_folds):
        mean_inner = {c: float(np.mean([condition_inner_auc[c][m][f] for m in MODEL_NAMES]))
                      for c in candidates}
        chosen.append(max(mean_inner, key=mean_inner.get))
        inner_rows.append({'Fold': f + 1, **{c: round(v, 3) for c, v in mean_inner.items()},
                           'Chosen': chosen[-1]})
    rows = []
    for m in MODEL_NAMES:
        fold_aucs = [condition_results[chosen[f]][m]['auc'][f] for f in range(n_folds)]
        mean, lo, hi = compute_ci(fold_aucs)
        post_hoc = float(np.mean(condition_results['ClassWeight_Strong'][m]['auc']))
        rows.append({'Candidate_set': label, 'Model': m,
                     'Nested_AUC': round(mean, 3), 'Nested_95CI': f"[{lo:.3f}, {hi:.3f}]",
                     'ClassWeight_Strong_AUC': round(post_hoc, 3),
                     'Difference': round(mean - post_hoc, 3)})
    return pd.DataFrame(inner_rows), pd.DataFrame(rows)

sets = [('all five conditions', list(CONDITIONS)),
        ('four fixed-grid conditions (no Optuna)', [c for c in CONDITIONS if c != 'Optuna_ClassWeight'])]

all_nested = []
for label, cands in sets:
    inner_df, res_df = nested_selection(cands, label)
    print("\n" + "=" * 100)
    print(f"NESTED SELECTION - candidate set: {label}")
    print("=" * 100)
    print("Inner-validation AUC (mean across the five algorithms) and the condition chosen in each outer fold:")
    print(inner_df.to_string(index=False))
    print("\nOuter-fold test AUC of the chosen condition (untouched by the selection):")
    print(res_df.drop(columns='Candidate_set').to_string(index=False))
    lr = res_df.loc[res_df['Model'] == 'Logistic Regression', 'Nested_AUC'].iloc[0]
    best_tree = res_df.loc[res_df['Model'] != 'Logistic Regression'].sort_values('Nested_AUC').iloc[-1]
    print(f"\nLogistic regression {lr:.3f} vs best tree-based algorithm "
          f"({best_tree['Model']}) {best_tree['Nested_AUC']:.3f}: "
          f"tree minus LR = {best_tree['Nested_AUC'] - lr:+.3f}")
    all_nested.append(res_df)

pd.concat(all_nested).to_csv(os.path.join(OUT_DIR, "nested_selection_model1.csv"), index=False)
print("\nSaved: nested_selection_model1.csv")


# ====================================================================
# SECTION 10: SUBGROUP PERFORMANCE BY INFANT SEX AND HOUSEHOLD WEALTH (Model 1)
# Reuses the saved out-of-fold probabilities (oof_probs, ClassWeight_Strong);
# no model is refit. Produces Supplementary Figure S4.
# ====================================================================
from sklearn.metrics import roc_auc_score

y_arr = y.reset_index(drop=True)
sex = df['b4'].reset_index(drop=True)
wealth = df['v190'].reset_index(drop=True)
wealth3 = wealth.map(lambda v: 'Poor' if v in ['poorest', 'poorer']
                      else ('Middle' if v == 'middle' else 'Rich'))

sex_rows, wealth_rows = [], []
for name in MODEL_NAMES:
    probs = oof_probs[name]

    srow = {'Model': name}
    for grp in ['male', 'female']:
        mask = (sex == grp).values
        srow[f'{grp}_n'] = int(mask.sum())
        srow[f'{grp}_events'] = int(y_arr[mask].sum())
        srow[f'{grp}_AUC'] = roc_auc_score(y_arr[mask], probs[mask])
    sex_rows.append(srow)

    wrow = {'Model': name}
    for grp in ['Poor', 'Middle', 'Rich']:
        mask = (wealth3 == grp).values
        wrow[f'{grp}_n'] = int(mask.sum())
        wrow[f'{grp}_events'] = int(y_arr[mask].sum())
        wrow[f'{grp}_AUC'] = roc_auc_score(y_arr[mask], probs[mask])
    wealth_rows.append(wrow)

sex_df = pd.DataFrame(sex_rows)
wealth_df = pd.DataFrame(wealth_rows)

print("\n=== Out-of-fold AUC by sex, ClassWeight_Strong condition, Model 1 ===")
print(sex_df.to_string(index=False))
print("\n=== Out-of-fold AUC by wealth, ClassWeight_Strong condition, Model 1 ===")
print(wealth_df.to_string(index=False))

sex_df.to_csv(os.path.join(OUT_DIR, "fairness_by_sex_model1.csv"), index=False)
wealth_df.to_csv(os.path.join(OUT_DIR, "fairness_by_wealth_model1.csv"), index=False)
print(f"\nSaved: fairness_by_sex_model1.csv")
print(f"Saved: fairness_by_wealth_model1.csv")

# ---- One combined figure, two panels, legends above each panel ----
models_lbl = [m.replace(' ', '\n', 1) if ' ' in m else m for m in MODEL_NAMES]
x = np.arange(len(MODEL_NAMES))

fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), gridspec_kw={'width_ratios': [1, 1.3]})

# Panel A: sex
ax = axes[0]
w = 0.35
b1 = ax.bar(x - w/2, sex_df['male_AUC'], w,
            label=f"Male (n={sex_df['male_n'].iloc[0]:,}, {sex_df['male_events'].iloc[0]} events)",
            color='#4C72B0')
b2 = ax.bar(x + w/2, sex_df['female_AUC'], w,
            label=f"Female (n={sex_df['female_n'].iloc[0]:,}, {sex_df['female_events'].iloc[0]} events)",
            color='#DD8452')
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{b.get_height():.3f}",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(models_lbl, fontsize=11)
ax.set_ylabel('Out-of-fold AUC', fontsize=11)
ax.set_ylim(0.4, 0.82)
ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize=10, frameon=False)
ax.set_title('A. By infant sex', fontsize=12, loc='left', y=1.17)

# Panel B: wealth
ax = axes[1]
w = 0.33
x2 = x * 1.6
cols = {'Poor': '#4C72B0', 'Middle': '#55A868', 'Rich': '#DD8452'}
offsets = {'Poor': 0.01, 'Middle': 0.032, 'Rich': 0.01}
for i, grp in enumerate(['Poor', 'Middle', 'Rich']):
    lbl = f"{grp} (n={wealth_df[f'{grp}_n'].iloc[0]:,}, {wealth_df[f'{grp}_events'].iloc[0]} events)"
    bars = ax.bar(x2 + (i - 1) * w, wealth_df[f'{grp}_AUC'], w, label=lbl, color=cols[grp])
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + offsets[grp], f"{b.get_height():.3f}",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(x2); ax.set_xticklabels(models_lbl, fontsize=11)
ax.set_ylabel('Out-of-fold AUC', fontsize=11)
ax.set_ylim(0.4, 0.82)
ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=9.5, frameon=False, columnspacing=1.2)
ax.set_title('B. By household wealth', fontsize=12, loc='left', y=1.17)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_fairness_combined_model1.png"), dpi=300, bbox_inches='tight')
print("Saved: fig_fairness_combined_model1.png")


# ====================================================================
# SECTION 11: CALIBRATION, OVERALL AND BY SUBGROUP (Model 1)
# Calibration intercept and slope per algorithm (overall) and observed-to-expected
# risk by sex and wealth group. Reuses oof_probs; no model is refit.
# ====================================================================
from scipy.special import expit, logit
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression

y_arr = y.reset_index(drop=True).values
sex_s = df['b4'].reset_index(drop=True)
wealth3_s = df['v190'].reset_index(drop=True).map(
    lambda v: 'Poor' if v in ['poorest', 'poorer'] else ('Middle' if v == 'middle' else 'Rich'))

def calib_intercept_slope(y_true, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    lp = logit(p)
    # C=1e6 = effectively unpenalised; works in every scikit-learn version
    m = LogisticRegression(C=1e6, max_iter=2000).fit(lp.reshape(-1, 1), y_true)
    slope = float(m.coef_[0][0])
    # calibration-in-the-large: intercept with the slope fixed at 1
    try:
        intercept = float(brentq(lambda a: np.sum(y_true - expit(a + lp)), -25, 25))
    except ValueError:
        intercept = float('nan')
    return intercept, slope

rows = []
for name in MODEL_NAMES:
    p = np.asarray(oof_probs[name], dtype=float)
    b0, b1 = calib_intercept_slope(y_arr, p)
    rows.append({'Model': name, 'Observed_rate': round(y_arr.mean(), 4),
                 'Mean_predicted': round(p.mean(), 4),
                 'O_to_E': round(y_arr.mean() / p.mean(), 3),
                 'Calib_intercept': round(b0, 2), 'Calib_slope': round(b1, 2)})
overall = pd.DataFrame(rows)
print("\n=== Calibration, out-of-fold, ClassWeight_Strong, Model 1 (n=%d, %d events) ===" % (len(y_arr), int(y_arr.sum())))
print(overall.to_string(index=False))
overall.to_csv(os.path.join(OUT_DIR, "calibration_overall_model1.csv"), index=False)

sub_rows = []
for dim, series, groups in [('Sex', sex_s, ['male', 'female']),
                            ('Wealth', wealth3_s, ['Poor', 'Middle', 'Rich'])]:
    for name in MODEL_NAMES:
        p = np.asarray(oof_probs[name], dtype=float)
        for g in groups:
            mask = (series == g).values
            obs, exp = y_arr[mask].mean(), p[mask].mean()
            sub_rows.append({'Dimension': dim, 'Group': g, 'Model': name,
                             'n': int(mask.sum()), 'Events': int(y_arr[mask].sum()),
                             'Observed_rate': round(obs, 4), 'Mean_predicted': round(exp, 4),
                             'O_to_E': round(obs / exp, 3)})
sub = pd.DataFrame(sub_rows)
print("\n=== Observed vs predicted risk by subgroup (O:E; ideal = 1) ===")
print(sub.to_string(index=False))
sub.to_csv(os.path.join(OUT_DIR, "calibration_by_subgroup_model1.csv"), index=False)
print("\nSaved: calibration_overall_model1.csv, calibration_by_subgroup_model1.csv")
