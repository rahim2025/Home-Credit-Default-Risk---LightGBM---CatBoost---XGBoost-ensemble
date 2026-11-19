import numpy as np, pandas as pd, lightgbm as lgb, sys
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
src = open("solution.py", encoding="utf8").read().split("# LightGBM params found")[0]
exec(src)
P = dict(objective="binary", metric="auc", learning_rate=0.05, n_estimators=4000, num_leaves=59, max_depth=4,
         min_child_samples=248, subsample=0.939, subsample_freq=1, colsample_bytree=0.196, reg_alpha=6.94,
         reg_lambda=0.127, min_split_gain=0.37, cat_smooth=88.7, min_data_per_group=100, n_jobs=-1, verbose=-1, random_state=SEED)
knn_cols = ["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3","AGE_YEARS","CREDIT_TERM","ANNUITY_INCOME","GOODS_CREDIT","EMPLOYED_YEARS"]
K = df[knn_cols].astype(float)
K = K.fillna(K.median()); K = ((K - K.mean()) / K.std()).values
Kx, Kt = K[:n], K[n:]
te_cols = ["ORGANIZATION_TYPE","OCCUPATION_TYPE","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE"]
def knn_feat(fit_idx, query, ytr, k=300, self_ex=False):
    nn = NearestNeighbors(n_neighbors=k+int(self_ex), n_jobs=-1).fit(Kx[fit_idx])
    d, ix = nn.kneighbors(query)
    if self_ex: ix = ix[:, 1:]; d = d[:, 1:]
    t = ytr[ix]
    return np.c_[t.mean(1), t[:, :50].mean(1), d.mean(1)]
def te_feat(trX, trY, qX, a=50):
    out = {}
    pm = trY.mean()
    for c in te_cols:
        s = pd.DataFrame({"c": trX[c].astype(str), "y": trY}).groupby("c")["y"].agg(["sum","count"])
        m = (s["sum"] + a*pm) / (s["count"] + a)
        out[c] = qX[c].astype(str).map(m).fillna(pm).values
    return pd.DataFrame(out)
def build(tr, va, Xte=None):
    """features for train part (inner OOF) and for query sets"""
    ytr = y[tr]
    inner = StratifiedKFold(5, shuffle=True, random_state=1)
    ftr = np.zeros((len(tr), 3)); tetr = pd.DataFrame(0.0, index=range(len(tr)), columns=te_cols)
    Xtr = X.iloc[tr].reset_index(drop=True)
    for a, b in inner.split(Xtr, ytr):
        ftr[b] = knn_feat(tr[a], Kx[tr[b]], ytr[a])
        tetr.iloc[b] = te_feat(Xtr.iloc[a], ytr[a], Xtr.iloc[b]).values
    fva = knn_feat(tr, Kx[va], ytr); teva = te_feat(Xtr, ytr, X.iloc[va])
    return ftr, tetr, fva, teva
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
base = np.zeros(n); new = np.zeros(n)
for fold, (tr, va) in enumerate(skf.split(X, y)):
    m = lgb.LGBMClassifier(**P).fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    base[va] = m.predict_proba(X.iloc[va])[:, 1]
    ftr, tetr, fva, teva = build(tr, va)
    def aug(Xp, f, t):
        Xp = Xp.reset_index(drop=True).copy()
        Xp[["KNN_MEAN","KNN_MEAN50","KNN_DIST"]] = f
        for c in te_cols: Xp["TE_"+c] = np.asarray(t[c])
        return Xp
    A, B = aug(X.iloc[tr], ftr, tetr), aug(X.iloc[va], fva, teva)
    m = lgb.LGBMClassifier(**P).fit(A, y[tr], eval_set=[(B, y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    new[va] = m.predict_proba(B)[:, 1]
    print(fold, round(roc_auc_score(y[va], base[va]), 5), round(roc_auc_score(y[va], new[va]), 5), flush=True)
print("BASE", roc_auc_score(y, base), "NEW", roc_auc_score(y, new))
