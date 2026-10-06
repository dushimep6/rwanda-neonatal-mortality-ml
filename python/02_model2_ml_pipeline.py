# ----------------------------------------------------------------------
# 02_model2_ml_pipeline.py
# Model 2 (sensitivity analysis): the same five algorithms and five conditions with
# three additional care-related predictors (place of delivery, mode of delivery,
# perceived birth size), births in the three years before the survey.
#
# INPUT : data/model2_analytic_data_improved.dta  (written by stata/02_model2_regression_and_export.do)
# OUTPUT: tables and figures in results/model2/
# RUN   : from the repository root, `python python/02_model2_ml_pipeline.py`
# ----------------------------------------------------------------------
# ======================================================================
# MODEL 2: NEONATAL MORTALITY - ML COMPARATORS
# 3-year restricted sample, 11 predictors (incl. facility delivery,
# mode of delivery, birth size) vs Logistic Regression baseline
# (Stata AUC=0.6986, corrected sample)
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
                              confusion_matrix, brier_score_loss)
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
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
import math

optuna.logging.set_verbosity(optuna.logging.WARNING)
RANDOM_STATE = 42
import os
# Folders can be set with environment variables; defaults are relative to the repository root.
DATA_DIR = os.environ.get("DATA_DIR", "data")      # holds the analytic .dta file written by the Stata do-file
OUT_DIR = os.environ.get("OUT_DIR", "results/model2")  # all tables and figures are written here
os.makedirs(OUT_DIR, exist_ok=True)


# ======================================================================
# SECTION 0: LOAD AND VERIFY DATA
# ======================================================================

df = pd.read_stata(os.path.join(DATA_DIR, "model2_analytic_data_improved.dta"))
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

categorical_vars = ['bord_cat', 'multiple_birth', 'v190', 'v106', 'b4',
                     'marital_cat', 'short_interval', 'matage_cat',
                     'facility_delivery', 'm17', 'birth_size_cat']

X_raw = df[categorical_vars].copy()
X_raw_cat = X_raw.copy()
for col in categorical_vars:
    X_raw_cat[col] = X_raw_cat[col].astype(str)

def sanitize_colname(col):
    col = col.replace('<', 'lt').replace('>', 'gt')
    col = re.sub(r'[\[\]]', '', col)
    return col

# ---- FIX: explicit category order (reference category listed FIRST,
# matching the Stata regression's reference/omitted level exactly) -
# used consistently everywhere in this script. Relying on alphabetical
# sorting silently drops the WRONG category
# for variables like multiple_birth ("Multiple birth" < "Single birth"
# alphabetically) and b4 ("female" < "male"), which both reverses the
# encoded direction relative to the Stata model AND can misalign
# feature names with SHAP values if two different orderings are used
# in different parts of the pipeline. ----
CATEGORY_ORDER = {
    'bord_cat': ['1st', '2nd-4th', '5th+'],
    'multiple_birth': ['Single birth', 'Multiple birth'],
    'v190': ['poorest', 'poorer', 'middle', 'richer', 'richest'],
    'v106': ['no education', 'primary', 'secondary', 'higher'],
    'b4': ['male', 'female'],
    'marital_cat': ['Never in union', 'Married', 'Living with partner', 'Formerly partnered'],
    'short_interval': ['Not short/NA', 'Short interval (lt24mo)'],
    'matage_cat': ['<20', '20-34', '35+'],
    'facility_delivery': ['Home', 'Facility'],
    'm17': ['Vaginal', 'Caesarean'],
    'birth_size_cat': ['Small', 'Average', 'Large'],
}

def encode_fixed(cat_df):
    """One-hot encode using a FIXED, explicit category order (reference
    category first) - identical every time this is called, anywhere in
    the script, so column names/order never drift between Section 1's
    initial X and any fold-level encoding done later."""
    df_enc = cat_df.copy()
    for col in categorical_vars:
        use_cats = CATEGORY_ORDER[col]
        if col == 'short_interval':
            # match the sanitized label used downstream ('<' -> 'lt')
            df_enc[col] = df_enc[col].replace({'Short interval (<24mo)': 'Short interval (lt24mo)'})
            use_cats = [c.replace('<', 'lt') for c in use_cats]
        df_enc[col] = pd.Categorical(df_enc[col], categories=use_cats)
    enc = pd.get_dummies(df_enc, columns=categorical_vars, drop_first=True)
    enc.columns = [sanitize_colname(c) for c in enc.columns]
    return enc.astype(int)

# Build X using the SAME encode_fixed function - not a separate
# pd.get_dummies call with different (natural/unsorted) category order
X = encode_fixed(X_raw_cat)

bad_chars = [c for c in X.columns if any(ch in c for ch in ['<', '>', '[', ']'])]
print("\nOK: no problematic characters remain." if not bad_chars else f"WARNING: {bad_chars}")
print("\nEncoded feature matrix shape:", X.shape)
print("Encoded columns:", X.columns.tolist())

assert len(X) == len(y)
assert X.isnull().sum().sum() == 0
print("\nAll checks passed - X and y are aligned and complete.")


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
# SECTION 5: OPTUNA SEARCH SPACES (30 trials/model/fold)
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
    X_train_enc = encode_fixed(Xcat_train)

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
naive_baseline_accuracy = 1 - y.mean()
print(f"\nNAIVE BASELINE ACCURACY (always predict 'no neonatal death'): {naive_baseline_accuracy:.3f}")
print("Accuracy is reported for completeness but is NOT informative given severe class")
print("imbalance (event rate = {:.3f}).".format(y.mean()))
print("\n" + "="*110)
print("FULL 5-CONDITION COMPARISON (all models x all conditions) - MODEL 2")
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

final_condition_df.to_csv(os.path.join(OUT_DIR, "five_condition_comparison_model2_revised.csv"), index=False)
print(f"\nSaved: five_condition_comparison_model2_revised.csv")
print(f"\nReference: Stata logistic regression (in-sample, full data) AUC = 0.6986 [0.635, 0.763]")


# ======================================================================
# SECTION 8: OUT-OF-FOLD SHAP - RANDOM FOREST (ClassWeight_Strong)
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
shap_importance_df.to_csv(os.path.join(OUT_DIR, "shap_importance_rf_oof_model2_revised.csv"), index=False)


# ======================================================================
# SECTION 9: SHAP FIGURES (no titles)
# ======================================================================

LABEL_MAP = {
    'bord_cat_2nd-4th': 'Birth order: 2nd-4th', 'bord_cat_5th+': 'Birth order: 5th+',
    'multiple_birth_Multiple birth': 'Multiple birth',
    'v190_poorer': 'Wealth: Poorer', 'v190_middle': 'Wealth: Middle',
    'v190_richer': 'Wealth: Richer', 'v190_richest': 'Wealth: Richest',
    'v106_primary': "Mother's education: Primary", 'v106_secondary': "Mother's education: Secondary",
    'v106_higher': "Mother's education: Higher", 'b4_female': 'Sex: Female',
    'marital_cat_Married': 'Marital status: Married',
    'marital_cat_Living with partner': 'Marital status: Living with partner',
    'marital_cat_Formerly partnered': 'Marital status: Formerly partnered',
    'short_interval_Short interval (lt24mo)': 'Birth interval: Short (<24mo)',
    'matage_cat_20-34': 'Maternal age: 20-34', 'matage_cat_35+': 'Maternal age: 35+',
    'facility_delivery_Facility': 'Facility delivery',
    'm17_Caesarean': 'Mode of delivery: Caesarean',
    'birth_size_cat_Average': 'Birth size: Average', 'birth_size_cat_Large': 'Birth size: Large',
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
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig_shap_bar_model2_revised.png"), dpi=300, bbox_inches='tight'); plt.close()
print("Saved: fig_shap_bar_model2_revised.png")

X_full_shap_labeled = X_full_shap.copy()
X_full_shap_labeled.columns = relabel(X_full_shap.columns)
plt.figure()
shap.summary_plot(oof_shap_array, X_full_shap_labeled, show=False)
plt.gca().set_title('')
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig_shap_beeswarm_model2_revised.png"), dpi=300, bbox_inches='tight'); plt.close()
print("Saved: fig_shap_beeswarm_model2_revised.png")


# ======================================================================
# SECTION 10: REMAINING FIGURES
# ======================================================================

# ---- Figure 1: Study flow (Model 2 numbers) ----
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
    (8.8,  "Births in 3-year recall period\n(excl. interview month), n = 4,345"),
    (6.2,  "Outcome available\nn = 4,340"),
    (3.6,  "Analytic sample (Model 2)\nn = 4,210\n(72 neonatal deaths)"),
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
    (7.5, "Excluded: unknown\nage at death, n = 5"),
    (4.9, "Excluded: missing\npredictor data, n = 130"),
]
for y_box, text in exclusions:
    draw_box(ax, excl_x, y_box, excl_w, excl_h, text, '#FEE0D2')
    ax.plot([main_x, excl_x - excl_w/2], [y_box, y_box], color='black', lw=1.2)
    ax.annotate('', xy=(excl_x - excl_w/2, y_box), xytext=(main_x + 0.05, y_box),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig1_study_flow_model2_revised.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig1_study_flow_model2_revised.png")

# ---- Figure 2: ROC curves - clean lines, no shading ----
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
plt.savefig(os.path.join(OUT_DIR, "fig2_roc_curves_model2_revised.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig2_roc_curves_model2_revised.png")

# ---- Figure 3: Brier scores ----
brier_rows = []
for name in MODEL_NAMES:
    valid = ~np.isnan(oof_probs[name])
    bs = brier_score_loss(y[valid], oof_probs[name][valid])
    brier_rows.append({'Model': name, 'Brier': bs})
brier_df = pd.DataFrame(brier_rows).sort_values('Brier')

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(brier_df['Model'], brier_df['Brier'],
                color=[MODEL_COLORS[n] for n in brier_df['Model']], edgecolor='black')
for bar, val in zip(bars, brier_df['Brier']):
    ax.annotate(f'{val:.4f}', xy=(val, bar.get_y() + bar.get_height()/2),
                xytext=(4, 0), textcoords='offset points',
                va='center', ha='left', fontsize=9, weight='bold')
ax.set_xlabel('Brier score (out-of-fold, lower = better)', fontsize=11)
ax.set_xlim(0, brier_df['Brier'].max() * 1.18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig3_brier_scores_model2_revised.png"), dpi=300, bbox_inches='tight')
plt.close()
brier_df.to_csv(os.path.join(OUT_DIR, "brier_scores_model2_revised.csv"), index=False)
print("Saved: fig3_brier_scores_model2_revised.png")

# ---- Figure 4: Calibration ----
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
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig4_calibration_model2_revised.png"), dpi=300, bbox_inches='tight'); plt.close()
print("Saved: fig4_calibration_model2_revised.png")

# ---- Figure 5: Confusion matrix ----
import seaborn as sns
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
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig5_confusion_matrix_model2_revised.png"), dpi=300, bbox_inches='tight'); plt.close()
print(f"Saved: fig5_confusion_matrix_model2_revised.png (best model: {best_model_name})")

# ---- Figure 6: Overfitting check ----
overfit_rows = []
for name in MODEL_NAMES:
    train_auc_mean = np.mean(oof_train_auc[name])
    test_auc_mean = cond_df.loc[cond_df['Model']==name, 'Test_AUC'].values[0]
    overfit_rows.append({'Model': name, 'Train AUC': round(train_auc_mean,3), 'Test AUC': round(test_auc_mean,3)})
overfit_df = pd.DataFrame(overfit_rows)
print("\nOverfitting check (ClassWeight_Strong condition):")
print(overfit_df.to_string(index=False))
fig, ax = plt.subplots(figsize=(9, 5))
x_pos = np.arange(len(overfit_df)); width = 0.35
ax.bar(x_pos - width/2, overfit_df['Train AUC'], width, label='Train AUC', color='#4575B4')
ax.bar(x_pos + width/2, overfit_df['Test AUC'], width, label='Test AUC', color='#D73027')
ax.set_xticks(x_pos); ax.set_xticklabels(overfit_df['Model'], rotation=20, ha='right')
ax.set_ylabel('AUC', fontsize=11); ax.set_ylim(0.4, 1.0); ax.legend()
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig6_overfitting_check_model2_revised.png"), dpi=300, bbox_inches='tight'); plt.close()
print("Saved: fig6_overfitting_check_model2_revised.png")
overfit_df.to_csv(os.path.join(OUT_DIR, "overfitting_check_model2_revised.csv"), index=False)

print("\n\n" + "="*70)
print("MODEL 2 PIPELINE COMPLETE - ALL SECTIONS RAN SUCCESSFULLY")
print("="*70)
