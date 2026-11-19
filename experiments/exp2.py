import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
src = open("solution.py", encoding="utf8").read().split("# LightGBM params found")[0]
exec(src)
P = dict(objective="binary", metric="auc", learning_rate=0.05, n_estimators=4000, num_leaves=59, max_depth=4,
         min_child_samples=248, subsample=0.939, subsample_freq=1, colsample_bytree=0.196, reg_alpha=6.94,
         reg_lambda=0.127, min_split_gain=0.37, cat_smooth=88.7, min_data_per_group=100, n_jobs=-1, verbose=-1, random_state=SEED)
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)

# 1. adversarial validation: can a model tell train from test?
lab = np.r_[np.zeros(n), np.ones(len(X_test))]
Xall = pd.concat([X, X_test], ignore_index=True)
o = np.zeros(len(lab))
for tr, va in StratifiedKFold(3, shuffle=True, random_state=0).split(Xall, lab):
    m = lgb.LGBMClassifier(**{**P, "n_estimators": 300}).fit(Xall.iloc[tr], lab[tr])
    o[va] = m.predict_proba(Xall.iloc[va])[:, 1]
print("ADV AUC (0.5 = no shift):", round(roc_auc_score(lab, o), 4), flush=True)
imp = pd.Series(m.feature_importances_, Xall.columns).sort_values(ascending=False)
print(imp.head(5).to_dict(), flush=True)

def cv(params, cols=None, name=""):
    Xs = X if cols is None else X[cols]
    oof = np.zeros(n)
    for tr, va in skf.split(Xs, y):
        m = lgb.LGBMClassifier(**params).fit(Xs.iloc[tr], y[tr], eval_set=[(Xs.iloc[va], y[va])],
                                           callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(Xs.iloc[va])[:, 1]
    print(name, round(roc_auc_score(y, oof), 5), flush=True)
    return oof

base = cv(P, name="baseline(121 feats)")
# 2. feature pruning by gain importance (computed on a single fold's train part only)
tr0, va0 = next(iter(skf.split(X, y)))
m0 = lgb.LGBMClassifier(**{**P, "importance_type": "gain"}).fit(X.iloc[tr0], y[tr0], eval_set=[(X.iloc[va0], y[va0])], callbacks=[lgb.early_stopping(100, verbose=False)])
rank = pd.Series(m0.feature_importances_, X.columns).sort_values(ascending=False).index.tolist()
for k in (40, 70, 95):
    cv(P, rank[:k], f"top{k} feats")
# 3. model variants
cv({**P, "extra_trees": True}, name="extra_trees")
cv({**P, "num_leaves": 15, "max_depth": 3, "min_child_samples": 400}, name="shallower d3")
cv({**P, "num_leaves": 120, "max_depth": 6, "min_child_samples": 300, "colsample_bytree": 0.15, "reg_lambda": 20}, name="deeper d6")
pd.Series(base).to_csv("oof_lgb_base.csv", index=False)
