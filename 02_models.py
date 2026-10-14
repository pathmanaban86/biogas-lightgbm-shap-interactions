"""Step 2: reproduce original result, then tuned comparison under expanding-window CV."""
import os; os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
import json, time, pickle, warnings
import numpy as np, pandas as pd, optuna
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from common import *

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
N_TRIALS = int(__import__('os').environ.get('N_TRIALS', 30))


def metrics(y, p):
    return dict(R2=r2_score(y, p), RMSE=np.sqrt(mean_squared_error(y, p)),
                MAE=mean_absolute_error(y, p), MAPE=np.mean(np.abs((y - p) / y)) * 100)


# ---------- A. Reproduce the submitted result (original row-based split, original settings) ----------
raw = pd.read_csv('data/biogas_dataset.csv')
s = int(0.8 * len(raw))
orig = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=10, num_leaves=31, min_child_samples=20,
                     subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
orig.fit(raw[FEATS].iloc[:s], raw[TARGET].iloc[:s])
repro = metrics(raw[TARGET].iloc[s:].values, orig.predict(raw[FEATS].iloc[s:]))
print('Reproduction of submitted LightGBM:', {k: round(v, 4) for k, v in repro.items()})

# ---------- B. Revised protocol ----------
df = load()
dev, test = date_split(df)
folds = ts_folds(dev)
Xd, yd, Xt, yt = dev[FEATS], dev[TARGET].values, test[FEATS], test[TARGET].values
print(f'dev {len(dev)} rows ({dev.Date.min().date()}..{dev.Date.max().date()}), '
      f'test {len(test)} rows ({test.Date.min().date()}..{test.Date.max().date()})')


def make(name, p, seed=SEED):
    if name == 'RF':
        return RandomForestRegressor(**p, random_state=seed, n_jobs=-1)
    if name == 'XGBoost':
        return XGBRegressor(**p, random_state=seed, n_jobs=-1, tree_method='hist')
    if name == 'LightGBM':
        return LGBMRegressor(**p, subsample_freq=1, random_state=seed, n_jobs=-1, verbose=-1)
    if name == 'CatBoost':
        return CatBoostRegressor(**p, random_seed=seed, verbose=0, thread_count=-1)
    if name == 'LightGBM-additive':  # depth-1 trees: no feature interactions possible
        return LGBMRegressor(**p, max_depth=1, num_leaves=2, subsample_freq=1, random_state=seed,
                             n_jobs=-1, verbose=-1)
    if name == 'Linear':
        return LinearRegression()


def space(name, t):
    if name == 'RF':
        return dict(n_estimators=t.suggest_int('n_estimators', 100, 500, step=100),
                    max_depth=t.suggest_int('max_depth', 6, 30),
                    min_samples_leaf=t.suggest_int('min_samples_leaf', 1, 10),
                    max_features=t.suggest_float('max_features', 0.3, 1.0))
    if name == 'XGBoost':
        return dict(n_estimators=t.suggest_int('n_estimators', 200, 1500, step=100),
                    learning_rate=t.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    max_depth=t.suggest_int('max_depth', 2, 10),
                    subsample=t.suggest_float('subsample', 0.6, 1.0),
                    colsample_bytree=t.suggest_float('colsample_bytree', 0.6, 1.0),
                    reg_lambda=t.suggest_float('reg_lambda', 1e-3, 10, log=True))
    if name in ('LightGBM', 'LightGBM-additive'):
        p = dict(n_estimators=t.suggest_int('n_estimators', 200, 2000, step=100),
                 learning_rate=t.suggest_float('learning_rate', 0.01, 0.2, log=True),
                 min_child_samples=t.suggest_int('min_child_samples', 5, 100),
                 subsample=t.suggest_float('subsample', 0.6, 1.0),
                 colsample_bytree=t.suggest_float('colsample_bytree', 0.6, 1.0),
                 reg_lambda=t.suggest_float('reg_lambda', 1e-3, 10, log=True))
        if name == 'LightGBM':
            p.update(num_leaves=t.suggest_int('num_leaves', 4, 128, log=True),
                     max_depth=t.suggest_int('max_depth', 2, 12))
        return p
    if name == 'CatBoost':
        return dict(iterations=t.suggest_int('iterations', 300, 1500, step=100),
                    learning_rate=t.suggest_float('learning_rate', 0.02, 0.2, log=True),
                    depth=t.suggest_int('depth', 3, 8),
                    l2_leaf_reg=t.suggest_float('l2_leaf_reg', 1, 10, log=True))


def cv_scores(name, p, seed=SEED):
    out = []
    for tr, va in folds:
        m = make(name, p, seed).fit(Xd.iloc[tr], yd[tr])
        out.append(metrics(yd[va], m.predict(Xd.iloc[va])))
    return pd.DataFrame(out)


results, best_params, test_pred = {}, {}, {}
for name in ['Linear', 'RF', 'XGBoost', 'CatBoost', 'LightGBM-additive', 'LightGBM']:
    t0 = time.time()
    if name == 'Linear':
        bp = {}
    else:
        st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
        st.optimize(lambda t: cv_scores(name, space(name, t)).RMSE.mean(), n_trials=N_TRIALS)
        bp = st.best_params
    best_params[name] = bp
    cv = cv_scores(name, bp)
    # final model on whole development period, evaluated once on hold-out; repeated over 5 seeds
    seeds = [0] if name == 'Linear' else [42, 1, 2, 3, 4]
    tm = []
    for sd in seeds:
        m = make(name, bp, sd).fit(Xd, yd)
        p = m.predict(Xt); tm.append(metrics(yt, p))
        if sd == seeds[0]:
            test_pred[name] = p
            if name == 'LightGBM':
                pickle.dump(m, open('results/lgbm_final.pkl', 'wb'))
    tm = pd.DataFrame(tm)
    tr_m = metrics(yd, make(name, bp).fit(Xd, yd).predict(Xd))
    results[name] = dict(train_R2=tr_m['R2'],
                         **{f'cv_{k}_mean': cv[k].mean() for k in cv}, **{f'cv_{k}_sd': cv[k].std() for k in cv},
                         **{f'test_{k}': tm[k].mean() for k in tm}, **{f'test_{k}_seed_sd': tm[k].std() for k in tm},
                         time_s=time.time() - t0)
    print(f'{name:18s} CV R2 {cv.R2.mean():.4f}±{cv.R2.std():.4f} RMSE {cv.RMSE.mean():.3f} | '
          f'test R2 {tm.R2.mean():.4f} RMSE {tm.RMSE.mean():.3f}  ({time.time()-t0:.0f}s)', flush=True)

# ---------- C. Date-block bootstrap CI on the hold-out set ----------
rng = np.random.default_rng(SEED)
dates = test.Date.values; ud = np.unique(dates)
idx_by_date = {d: np.where(dates == d)[0] for d in ud}
boot = {k: [] for k in test_pred}
for _ in range(2000):
    samp = np.concatenate([idx_by_date[d] for d in rng.choice(ud, len(ud), replace=True)])
    for k, p in test_pred.items():
        boot[k].append([r2_score(yt[samp], p[samp]), np.sqrt(mean_squared_error(yt[samp], p[samp]))])
for k in boot:
    b = np.array(boot[k])
    results[k].update(test_R2_lo=np.percentile(b[:, 0], 2.5), test_R2_hi=np.percentile(b[:, 0], 97.5),
                      test_RMSE_lo=np.percentile(b[:, 1], 2.5), test_RMSE_hi=np.percentile(b[:, 1], 97.5))
# paired bootstrap: is LightGBM better than each other model?
for k in boot:
    if k != 'LightGBM':
        d = np.array(boot[k])[:, 1] - np.array(boot['LightGBM'])[:, 1]
        results[k]['dRMSE_vs_LGBM_CI'] = [np.percentile(d, 2.5), np.percentile(d, 97.5)]

res = pd.DataFrame(results).T
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(res.round(4))
res.to_csv('results/table4_models.csv')
json.dump(dict(reproduction=repro, best_params=best_params), open('results/model_meta.json', 'w'), indent=1, default=float)
pickle.dump(dict(test_pred=test_pred, yt=yt, test_dates=test.Date.values), open('results/test_preds.pkl', 'wb'))
