import json, optuna, numpy as np, lightgbm as lgb
src = open("solution.py", encoding="utf8").read().split("lgb_params = dict(")[0]
exec(src)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
def obj(t):
    p = dict(objective="binary", metric="auc", learning_rate=0.05, n_estimators=3000,
        num_leaves=t.suggest_int("num_leaves", 8, 64), max_depth=t.suggest_int("max_depth", 3, 8),
        min_child_samples=t.suggest_int("min_child_samples", 30, 400),
        subsample=t.suggest_float("subsample", 0.6, 1.0), subsample_freq=1,
        colsample_bytree=t.suggest_float("colsample_bytree", 0.15, 0.7),
        reg_alpha=t.suggest_float("reg_alpha", 1e-2, 30, log=True),
        reg_lambda=t.suggest_float("reg_lambda", 1e-1, 100, log=True),
        min_split_gain=t.suggest_float("min_split_gain", 0, 0.5),
        cat_smooth=t.suggest_float("cat_smooth", 5, 100), min_data_per_group=100,
        random_state=SEED, n_jobs=-1, verbose=-1)
    o = np.zeros(n)
    for tr, va in skf3.split(X, y):
        m = lgb.LGBMClassifier(**p)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        o[va] = m.predict_proba(X.iloc[va])[:, 1]
    return roc_auc_score(y, o)
optuna.logging.set_verbosity(optuna.logging.WARNING)
st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
st.optimize(obj, n_trials=40, callbacks=[lambda s, t: print(t.number, round(t.value, 5), round(s.best_value, 5), flush=True)])
print("BEST", st.best_value, st.best_params)
json.dump(st.best_params, open("best_lgb.json", "w"))
