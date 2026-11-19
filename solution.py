"""Home Credit Default Risk - LightGBM + CatBoost + XGBoost ensemble (5-fold CV, seed-fixed).
Reproduces output/submission.csv from input/train.csv and input/test.csv only.

Run from the competition folder:  python solution.py   (about 15-20 min; XGBoost and CatBoost use
the GPU: device="cuda" / task_type="GPU". Set them to CPU if no NVIDIA GPU is available).
Requirements: numpy, pandas, scipy, scikit-learn, lightgbm, xgboost, catboost.
Expected: out-of-fold blend AUC about 0.7636; public leaderboard 0.76592.

Rules compliance: no external data, no hand-labelling; every prediction comes from the models.
All random seeds are fixed (SEED=42 plus SEEDS list below). GPU training may cause tiny
run-to-run differences in the last decimals.

Pipeline:
  1. Cleaning: DAYS_EMPLOYED == 365243 and "XNA" are treated as missing.
  2. Feature engineering: ratios (credit/income, annuity/income, ...), EXT_SOURCE aggregates and
     interactions, age/employment ratios, and group aggregates by organisation/occupation/education.
  3. EXT_SOURCE imputation features: each EXT_SOURCE is predicted from the other columns with a
     LightGBM regressor (out-of-fold); prediction, imputed value and residual are added as features.
     Uses features only, no labels. (My own idea, validated by CV: +0.0015 AUC.)
  4. LightGBM (params from an Optuna search, see experiments/tune.py), XGBoost and CatBoost,
     5-fold stratified CV, 3 seeds each; the final prediction is a rank average of the three models.

References (ideas only, no code copied):
  - deepsense.ai, 5th-place write-up of the original Kaggle competition:
    https://deepsense.ai/wait-so-loans-need-to-be-repaid-the-home-credit-risk-prediction-competition-on-kaggle/
  - Public Kaggle write-ups for the usual EXT_SOURCE aggregates and credit/annuity/income ratios,
    e.g. https://medium.com/thecyphy/home-credit-default-risk-part-2-84b58c1ab9d5
  - Original competition: https://www.kaggle.com/c/home-credit-default-risk
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 42
np.random.seed(SEED)

train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")
y = train["TARGET"].values
test_ids = test["SK_ID_CURR"].values

df = pd.concat([train.drop(columns="TARGET"), test], ignore_index=True)
df = df.drop(columns="SK_ID_CURR")

# --- cleaning ---
df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
for c in df.select_dtypes("object").columns:
    df[c] = df[c].replace("XNA", np.nan)

# --- feature engineering ---
ext = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
df["EXT_MEAN"] = df[ext].mean(axis=1)
df["EXT_MIN"] = df[ext].min(axis=1)
df["EXT_MAX"] = df[ext].max(axis=1)
df["EXT_STD"] = df[ext].std(axis=1)
df["EXT_NAN"] = df[ext].isna().sum(axis=1)
df["EXT_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
df["EXT_W"] = (df["EXT_SOURCE_1"] * 2 + df["EXT_SOURCE_2"] * 3 + df["EXT_SOURCE_3"] * 4) / 9
df["EXT_12"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]
df["EXT_13"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]
df["EXT_23"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
df["EXT_DIFF_12"] = df["EXT_SOURCE_1"] - df["EXT_SOURCE_2"]
df["EXT_DIFF_23"] = df["EXT_SOURCE_2"] - df["EXT_SOURCE_3"]
df["EXT_DIFF_13"] = df["EXT_SOURCE_1"] - df["EXT_SOURCE_3"]
df["CREDIT_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
df["ANNUITY_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
df["CREDIT_TERM_INV"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
df["GOODS_CREDIT"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
df["GOODS_INCOME"] = df["AMT_GOODS_PRICE"] / df["AMT_INCOME_TOTAL"]
df["CREDIT_GOODS_DIFF"] = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]
df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
df["INCOME_PER_CHILD"] = df["AMT_INCOME_TOTAL"] / (1 + df["CNT_CHILDREN"])
df["CREDIT_PER_PERSON"] = df["AMT_CREDIT"] / df["CNT_FAM_MEMBERS"]
df["ANNUITY_PER_PERSON"] = df["AMT_ANNUITY"] / df["CNT_FAM_MEMBERS"]
df["CHILD_RATIO"] = df["CNT_CHILDREN"] / df["CNT_FAM_MEMBERS"]
df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365
df["EMPLOYED_BIRTH"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
df["INCOME_EMPLOYED"] = df["AMT_INCOME_TOTAL"] / (-df["DAYS_EMPLOYED"] + 1)
df["CAR_AGE_BIRTH"] = df["OWN_CAR_AGE"] / df["AGE_YEARS"]
df["CAR_EMPLOYED"] = df["OWN_CAR_AGE"] / (EMP := (-df["DAYS_EMPLOYED"] / 365 + 1))
df["PHONE_BIRTH"] = df["DAYS_LAST_PHONE_CHANGE"] / df["DAYS_BIRTH"]
df["PHONE_EMPLOYED"] = df["DAYS_LAST_PHONE_CHANGE"] / df["DAYS_EMPLOYED"]
df["REG_BIRTH"] = df["DAYS_REGISTRATION"] / df["DAYS_BIRTH"]
df["ID_BIRTH"] = df["DAYS_ID_PUBLISH"] / df["DAYS_BIRTH"]
df["ID_REG_DIFF"] = df["DAYS_ID_PUBLISH"] - df["DAYS_REGISTRATION"]
df["CREDIT_TO_AGE"] = df["AMT_CREDIT"] / df["AGE_YEARS"]
df["TERM_X_AGE"] = df["CREDIT_TERM_INV"] + df["AGE_YEARS"]  # approx age at loan end
df["EXT2_CREDIT"] = df["EXT_SOURCE_2"] * df["AMT_CREDIT"]
df["EXT3_ANNUITY_INCOME"] = df["EXT_SOURCE_3"] * df["ANNUITY_INCOME"]
df["EXT_MEAN_AGE"] = df["EXT_MEAN"] * df["AGE_YEARS"]
df["EXT_MEAN_CREDIT_INCOME"] = df["EXT_MEAN"] * df["CREDIT_INCOME"]
df["EXT_MEAN_EMPLOYED"] = df["EXT_MEAN"] * df["EMPLOYED_BIRTH"]
bur = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU")]
df["BUREAU_TOTAL"] = df[bur].sum(axis=1)
df["BUREAU_NAN"] = df[bur].isna().sum(axis=1)
df["FLAGS_SUM"] = df[["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE"]].sum(axis=1)
df["NAN_COUNT"] = df.isna().sum(axis=1)

# --- group aggregates (unsupervised, computed on train+test features only) ---
def add_group(group_cols, name, vals):
    key = df[group_cols].astype(str).agg("_".join, axis=1) if len(group_cols) > 1 else df[group_cols[0]].astype(str)
    for v in vals:
        g = df.groupby(key)[v]
        mean, std = g.transform("mean"), g.transform("std")
        df[f"{name}_{v}_MEAN"] = mean
        df[f"{name}_{v}_DIFF"] = df[v] - mean
        df[f"{name}_{v}_Z"] = (df[v] - mean) / (std + 1e-9)

add_group(["ORGANIZATION_TYPE"], "ORG", ["AMT_INCOME_TOTAL", "EXT_MEAN", "AGE_YEARS"])
add_group(["OCCUPATION_TYPE"], "OCC", ["AMT_INCOME_TOTAL", "EXT_MEAN", "AMT_CREDIT", "EMPLOYED_YEARS"])
add_group(["NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE"], "EDUINC", ["AMT_INCOME_TOTAL", "EXT_MEAN", "AMT_CREDIT", "AGE_YEARS"])
add_group(["REGION_RATING_CLIENT", "NAME_CONTRACT_TYPE"], "REG", ["AMT_CREDIT", "AMT_INCOME_TOTAL", "ANNUITY_INCOME"])

df = df.replace([np.inf, -np.inf], np.nan)

# --- categorical encoding ---
cat_cols = df.select_dtypes("object").columns.tolist()
for c in cat_cols:
    df[c] = df[c].astype("category")
# --- EXT_SOURCE imputation features (unsupervised: no labels, uses train+test features only) ---
# Predict each EXT_SOURCE from the non-EXT columns (out-of-fold for rows where it is known),
# then use the prediction, the imputed value, and the residual as features.
from sklearn.model_selection import KFold
_feat = [c for c in df.columns if "EXT" not in c]
_base = [c for c in _feat if not any(c.startswith(p) for p in ("ORG_", "OCC_", "EDUINC_", "REG_"))]
_R = dict(objective="regression", learning_rate=0.08, n_estimators=400, num_leaves=63, min_child_samples=50,
          colsample_bytree=0.6, subsample=0.8, subsample_freq=1, n_jobs=-1, verbose=-1, random_state=SEED)
_new = pd.DataFrame(index=df.index)
for k in (1, 2, 3):
    col = f"EXT_SOURCE_{k}"
    oth = [c for c in _base if c != col]
    present = df[col].notna().values
    pred_k = np.zeros(len(df))
    idx = np.where(present)[0]
    for a, b in KFold(5, shuffle=True, random_state=SEED).split(idx):
        m = lgb.LGBMRegressor(**_R).fit(df.iloc[idx[a]][oth], df[col].values[idx[a]])
        pred_k[idx[b]] = m.predict(df.iloc[idx[b]][oth])
    m = lgb.LGBMRegressor(**_R).fit(df.iloc[idx][oth], df[col].values[idx])
    miss = np.where(~present)[0]
    if len(miss):
        pred_k[miss] = m.predict(df.iloc[miss][oth])
    _new[f"{col}_PRED"] = pred_k
    _new[f"{col}_IMP"] = np.where(present, df[col].values, pred_k)
    _new[f"{col}_RESID"] = df[col].values - pred_k
_imp = _new[[f"EXT_SOURCE_{k}_IMP" for k in (1, 2, 3)]]
_new["EXTI_MEAN"] = _imp.mean(axis=1)
_new["EXTI_PROD"] = _imp.prod(axis=1)
_new["EXTI_MIN"] = _imp.min(axis=1)
_new["EXTI_STD"] = _imp.std(axis=1)
df = pd.concat([df, _new], axis=1)

df_codes = df.copy()
for c in cat_cols:  # integer codes for xgboost / catboost fallback
    df_codes[c] = df_codes[c].cat.codes.replace(-1, np.nan)

n = len(train)
X, X_test = df.iloc[:n].reset_index(drop=True), df.iloc[n:].reset_index(drop=True)
Xc, Xc_test = df_codes.iloc[:n].reset_index(drop=True), df_codes.iloc[n:].reset_index(drop=True)
# CatBoost: string categoricals
Xcb = df.copy()
for c in cat_cols:
    Xcb[c] = Xcb[c].astype(object).where(Xcb[c].notna(), "NA").astype(str)
Xcb, Xcb_test = Xcb.iloc[:n].reset_index(drop=True), Xcb.iloc[n:].reset_index(drop=True)
print("features:", X.shape[1], flush=True)

# LightGBM params found with Optuna (tune.py, 3-fold CV), refit with a lower learning rate
lgb_params = dict(
    objective="binary", metric="auc", learning_rate=0.015, n_estimators=10000,
    num_leaves=59, max_depth=4, min_child_samples=248, subsample=0.939, subsample_freq=1,
    colsample_bytree=0.196, reg_alpha=6.94, reg_lambda=0.127, min_split_gain=0.37,
    cat_smooth=88.7, min_data_per_group=100, n_jobs=-1, verbose=-1,
)
xgb_params = dict(
    objective="binary:logistic", eval_metric="auc", learning_rate=0.02, n_estimators=8000,
    max_depth=4, min_child_weight=30, subsample=0.8, colsample_bytree=0.3,
    reg_alpha=2.0, reg_lambda=10.0, tree_method="hist", device="cuda",
    early_stopping_rounds=200,
)
cb_params = dict(
    iterations=6000, learning_rate=0.04, depth=6, l2_leaf_reg=10, eval_metric="AUC",
    od_type="Iter", od_wait=300, verbose=0, task_type="GPU", devices="0",
    cat_features=cat_cols,
)
SEEDS = [42, 7, 2024]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
names = ["lgb", "xgb", "cat"]
oof = {k: np.zeros(n) for k in names}
pred = {k: np.zeros(len(X_test)) for k in names}
NS = len(SEEDS)
for fold, (tr, va) in enumerate(skf.split(X, y)):
    for sd in SEEDS:
        m = lgb.LGBMClassifier(**lgb_params, random_state=sd)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
              callbacks=[lgb.early_stopping(300, verbose=False)])
        oof["lgb"][va] += m.predict_proba(X.iloc[va])[:, 1] / NS
        pred["lgb"] += m.predict_proba(X_test)[:, 1] / 5 / NS

        m = xgb.XGBClassifier(**xgb_params, random_state=sd)
        m.fit(Xc.iloc[tr], y[tr], eval_set=[(Xc.iloc[va], y[va])], verbose=False)
        oof["xgb"][va] += m.predict_proba(Xc.iloc[va])[:, 1] / NS
        pred["xgb"] += m.predict_proba(Xc_test)[:, 1] / 5 / NS

        m = CatBoostClassifier(**cb_params, random_seed=sd)
        m.fit(Xcb.iloc[tr], y[tr], eval_set=(Xcb.iloc[va], y[va]))
        oof["cat"][va] += m.predict_proba(Xcb.iloc[va])[:, 1] / NS
        pred["cat"] += m.predict_proba(Xcb_test)[:, 1] / 5 / NS

    print(f"fold {fold}: " + " ".join(f"{k}={roc_auc_score(y[va], oof[k][va]):.5f}" for k in names), flush=True)

for k in names:
    print(f"OOF {k}: {roc_auc_score(y, oof[k]):.5f}")

# rank-average blend
def rk(a):
    return rankdata(a) / len(a)

blend_oof = sum(rk(oof[k]) for k in names) / len(names)
print(f"OOF blend (equal rank avg): {roc_auc_score(y, blend_oof):.5f}", flush=True)
blend_test = sum(rk(pred[k]) for k in names) / len(names)

os.makedirs("output", exist_ok=True)
sub = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": blend_test})
sub.to_csv("output/submission.csv", index=False)
print(sub.shape, sub["TARGET"].describe())
