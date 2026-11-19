import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
src = open("solution.py", encoding="utf8").read().split("# LightGBM params found")[0]
exec(src)
P = dict(objective="binary", metric="auc", learning_rate=0.05, n_estimators=4000, num_leaves=59, max_depth=4,
         min_child_samples=248, subsample=0.939, subsample_freq=1, colsample_bytree=0.196, reg_alpha=6.94,
         reg_lambda=0.127, min_split_gain=0.37, cat_smooth=88.7, min_data_per_group=100, n_jobs=-1, verbose=-1, random_state=SEED)
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
raw = [c for c in train.columns if c not in ("SK_ID_CURR", "TARGET")]
R = dict(objective="regression", learning_rate=0.1, n_estimators=250, num_leaves=63, min_child_samples=50,
         colsample_bytree=0.7, subsample=0.8, subsample_freq=1, n_jobs=-1, verbose=-1, random_state=SEED)
targets = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_GOODS_PRICE", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED",
           "OWN_CAR_AGE", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE", "REGION_POPULATION_RELATIVE"]
logt = {"AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_GOODS_PRICE", "AMT_ANNUITY"}
new = pd.DataFrame(index=df.index)
for col in targets:
    oth = [c for c in raw if c != col]
    v = df[col].astype(float).values
    v = np.log1p(v) if col in logt else v
    present = ~np.isnan(v); idx = np.where(present)[0]
    pred = np.zeros(len(df))
    for a, b in KFold(5, shuffle=True, random_state=SEED).split(idx):
        m = lgb.LGBMRegressor(**R).fit(df.iloc[idx[a]][oth], v[idx[a]]); pred[idx[b]] = m.predict(df.iloc[idx[b]][oth])
    miss = np.where(~present)[0]
    if len(miss):
        m = lgb.LGBMRegressor(**R).fit(df.iloc[idx][oth], v[idx]); pred[miss] = m.predict(df.iloc[miss][oth])
    new[f"{col}_XP"] = pred; new[f"{col}_XR"] = v - pred
    print(col, "R2", round(np.corrcoef(pred[idx], v[idx])[0, 1] ** 2, 3), flush=True)
X2 = pd.concat([X, new.iloc[:n].reset_index(drop=True)], axis=1)
def cv(Xs, name):
    oof = np.zeros(n)
    for tr, va in skf.split(Xs, y):
        m = lgb.LGBMClassifier(**P).fit(Xs.iloc[tr], y[tr], eval_set=[(Xs.iloc[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(Xs.iloc[va])[:, 1]
    print(name, round(roc_auc_score(y, oof), 5), flush=True)
cv(X, "baseline (current features)")
cv(X2, "+residual feats for 11 columns")
cv(pd.concat([X, new.iloc[:n].reset_index(drop=True)[[c for c in new.columns if c.endswith("_XR")]]], axis=1), "+residuals only")
