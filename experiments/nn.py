import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
src = open("solution.py", encoding="utf8").read().split("# LightGBM params found")[0]
exec(src)
torch.manual_seed(SEED)
num = [c for c in df.columns if c not in cat_cols]
D = df[num].astype(float)
nanc = [c for c in num if D[c].isna().any()]
ind = D[nanc].isna().astype(np.float32).add_suffix("_nan")
D = D.fillna(D.median())
Q = QuantileTransformer(n_quantiles=200, output_distribution="normal", subsample=10**9, random_state=SEED).fit_transform(D)
oh = pd.get_dummies(df[cat_cols].astype(object).fillna("NA"), dtype=np.float32)
M = np.hstack([Q, ind.values, oh.values]).astype(np.float32)
Mx, Mt = M[:n], M[n:]
print("matrix", M.shape, flush=True)
dev = "cuda"
def fit_nn(tr, va, seed, epochs=25):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(M.shape[1], 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
                        nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * (len(tr) // 512 + 1))
    Xt = torch.tensor(Mx[tr], device=dev); yt = torch.tensor(y[tr], dtype=torch.float32, device=dev)
    Xv = torch.tensor(Mx[va], device=dev); Xte = torch.tensor(Mt, device=dev)
    best, bv, bt = 0, None, None
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(tr), device=dev)
        for i in range(0, len(tr), 512):
            b = perm[i:i+512]
            if len(b) < 2: continue
            loss = nn.functional.binary_cross_entropy_with_logits(net(Xt[b]).squeeze(1), yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xv).squeeze(1)).cpu().numpy()
            a = roc_auc_score(y[va], pv)
            if a > best: best, bv, bt = a, pv, torch.sigmoid(net(Xte).squeeze(1)).cpu().numpy()
    return bv, bt
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
oof_nn = np.zeros(n); te_nn = np.zeros(len(Mt)); oof_lr = np.zeros(n)
for f, (tr, va) in enumerate(skf.split(Mx, y)):
    for s in (1, 2, 3):
        pv, pt = fit_nn(tr, va, s); oof_nn[va] += pv / 3; te_nn += pt / 15
    lr = LogisticRegression(C=0.02, max_iter=300).fit(Mx[tr], y[tr]); oof_lr[va] = lr.predict_proba(Mx[va])[:, 1]
    print(f, "nn", round(roc_auc_score(y[va], oof_nn[va]), 5), "lr", round(roc_auc_score(y[va], oof_lr[va]), 5), flush=True)
print("OOF nn", roc_auc_score(y, oof_nn), "lr", roc_auc_score(y, oof_lr))
lg = pd.read_csv("oof_lgb_base.csv").iloc[:, 0].values
rk = lambda a: rankdata(a) / len(a)
for w in (0, 0.1, 0.2, 0.3, 0.4, 0.5):
    print("blend lgb+nn w=%.1f" % w, round(roc_auc_score(y, (1 - w) * rk(lg) + w * rk(oof_nn)), 5))
np.save("oof_nn.npy", oof_nn); np.save("te_nn.npy", te_nn)
