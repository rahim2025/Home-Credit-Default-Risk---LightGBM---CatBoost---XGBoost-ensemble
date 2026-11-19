import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
src = open("solution.py", encoding="utf8").read().split("# LightGBM params found")[0]
exec(src)
P = dict(objective="binary", metric="auc", learning_rate=0.05, n_estimators=4000, num_leaves=59, max_depth=4,
         min_child_samples=248, subsample=0.939, subsample_freq=1, colsample_bytree=0.196, reg_alpha=6.94,
         reg_lambda=0.127, min_split_gain=0.37, cat_smooth=88.7, min_data_per_group=100, n_jobs=-1, verbose=-1, random_state=SEED)
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
feat = [c for c in df.columns if "EXT" not in c]   # no EXT-derived inputs, no labels
base_cols = [c for c in feat if not any(c.startswith(p) for p in ("ORG_", "OCC_", "EDUINC_", "REG_"))]
R = dict(objective="regression", learning_rate=0.08, n_estimators=400, num_leaves=63, min_child_samples=50,
         colsample_bytree=0.6, subsample=0.8, subsample_freq=1, n_jobs=-1, verbose=-1, random_state=SEED)
new = pd.DataFrame(index=df.index)
for k in (1, 2, 3):
    col = f"EXT_SOURCE_{k}"
    oth = [c for c in base_cols if c != col]
    present = df[col].notna().values
    pred = np.zeros(len(df))
    idx = np.where(present)[0]
    for a, b in KFold(5, shuffle=True, random_state=SEED).split(idx):
        m = lgb.LGBMRegressor(**R).fit(df.iloc[idx[a]][oth], df[col].values[idx[a]])
        pred[idx[b]] = m.predict(df.iloc[idx[b]][oth])
    m = lgb.LGBMRegressor(**R).fit(df.iloc[idx][oth], df[col].values[idx])
    miss = np.where(~present)[0]
    if len(miss): pred[miss] = m.predict(df.iloc[miss][oth])
    r2 = np.corrcoef(pred[idx], df[col].values[idx])[0, 1]**2
    print(col, "missing:", len(miss), "R2 of imputer:", round(r2, 3), flush=True)
    new[f"{col}_PRED"] = pred
    new[f"{col}_IMP"] = np.where(present, df[col].values, pred)
    new[f"{col}_RESID"] = df[col].values - pred
imp = new[[f"EXT_SOURCE_{k}_IMP" for k in (1, 2, 3)]]
new["EXTI_MEAN"] = imp.mean(axis=1); new["EXTI_PROD"] = imp.prod(axis=1); new["EXTI_MIN"] = imp.min(axis=1)
new["EXTI_STD"] = imp.std(axis=1)
X2 = pd.concat([X, new.iloc[:n].reset_index(drop=True)], axis=1)
def cv(Xs, name):
    oof = np.zeros(n)
    for tr, va in skf.split(Xs, y):
        m = lgb.LGBMClassifier(**P).fit(Xs.iloc[tr], y[tr], eval_set=[(Xs.iloc[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(Xs.iloc[va])[:, 1]
    print(name, round(roc_auc_score(y, oof), 5), flush=True)
cv(X, "baseline")
cv(X2, "+EXT imputation feats")
