"""Step 4: SHAP, signed interactions with bootstrap CI and additive null, network, climate, seasons."""
import os; os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
import json, pickle, warnings, itertools
import numpy as np, pandas as pd, shap, networkx as nx
from scipy import stats
import scikit_posthocs as sp
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from common import *
warnings.filterwarnings('ignore')
rng = np.random.default_rng(SEED)

df = load(); dev, test = date_split(df)
Xd, yd, Xt, yt = dev[FEATS], dev[TARGET].values, test[FEATS], test[TARGET].values
meta = json.load(open('results/model_meta.json'))
bp, bpa = meta['best_params']['LightGBM'], meta['best_params']['LightGBM-additive']
mk = lambda p, **kw: LGBMRegressor(**p, subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1, **kw)
model = pickle.load(open('results/lgbm_final.pkl', 'rb'))
pred = model.predict(Xt)
out = {}

# ---------------- SHAP main effects on the full hold-out set ----------------
ex = shap.TreeExplainer(model)
sv = ex.shap_values(Xt)
base = float(np.atleast_1d(ex.expected_value)[0])
imp = pd.Series(np.abs(sv).mean(0), index=FEATS).sort_values(ascending=False)
t5 = pd.DataFrame({'mean_abs_SHAP': imp, 'share_pct': 100 * imp / imp.sum()})
t5['cum_pct'] = t5.share_pct.cumsum()
print(t5.round(3)); t5.round(4).to_csv('results/table5_shap.csv')
grp = {'Feedstock': FEED, 'Operational': OPER, 'Climate': CLIM}
gshare = {g: 100 * imp[f].sum() / imp.sum() for g, f in grp.items()}
print('group shares of total mean|SHAP| (%):', {k: round(v, 2) for k, v in gshare.items()})
out.update(base_value=base, group_share=gshare, top3_share=t5.share_pct.iloc[:3].sum())

# ---------------- SHAP interactions ----------------
n_int = 1500
sub = rng.choice(len(Xt), n_int, replace=False)
Xs = Xt.iloc[sub]; dates_s = test.Date.values[sub]
siv = ex.shap_interaction_values(Xs)          # (n, p, p); off-diagonal split symmetrically
fi = [FEATS.index(f) for f in FEED]
pairs = list(itertools.combinations(range(len(FEED)), 2))
def pair_stats(S):
    # total pairwise interaction effect = phi_ij + phi_ji = 2*phi_ij
    return (np.array([[2 * S[:, fi[a], fi[b]].mean(), 2 * np.abs(S[:, fi[a], fi[b]]).mean()] for a, b in pairs]))
obs = pair_stats(siv)
# overall weight of interactions in the model's predictions
off = siv.sum(axis=2) - np.einsum('nii->ni', siv)        # each feature's interaction part
inter_total = off.sum(1); fx = siv.sum((1, 2)) + base
out['interaction_var_share_pct'] = 100 * np.var(inter_total) / np.var(fx)
out['interaction_abs_share_pct'] = 100 * np.abs(off).sum(1).mean() / np.abs(siv).sum((1, 2)).mean()
print('interaction terms: share of prediction variance %.3f %%, share of total |attribution| %.2f %%'
      % (out['interaction_var_share_pct'], out['interaction_abs_share_pct']))

# bootstrap CI over dates
ud = np.unique(dates_s); idx_by = {d: np.where(dates_s == d)[0] for d in ud}
boot = []
for _ in range(1000):
    s = np.concatenate([idx_by[d] for d in rng.choice(ud, len(ud))])
    boot.append(pair_stats(siv[s]))
boot = np.array(boot)
# null under additivity: target = additive-model prediction + resampled residuals; refit full model
add = mk(bpa, max_depth=1, num_leaves=2).fit(Xd, yd)
fa = add.predict(Xd); resid = yd - fa
null = []
for b in range(100):
    ystar = fa + rng.choice(resid, len(resid), replace=True)
    mb = mk(bp, ).set_params(random_state=b).fit(Xd, ystar)
    null.append(pair_stats(shap.TreeExplainer(mb).shap_interaction_values(Xs))[:, 1])
    print('null', b, flush=True)
null = np.array(null)
P = pd.DataFrame({'i': [SYM[FEED[a]] for a, b in pairs], 'j': [SYM[FEED[b]] for a, b in pairs],
                  'signed_mean': obs[:, 0], 'signed_lo': np.percentile(boot[:, :, 0], 2.5, 0),
                  'signed_hi': np.percentile(boot[:, :, 0], 97.5, 0),
                  'mean_abs': obs[:, 1], 'abs_lo': np.percentile(boot[:, :, 1], 2.5, 0),
                  'abs_hi': np.percentile(boot[:, :, 1], 97.5, 0),
                  'null_mean_abs': null.mean(0), 'null_q95': np.percentile(null, 95, 0)})
P['p_emp'] = (1 + (null >= obs[:, 1]).sum(0)) / (1 + len(null))
P['exceeds_null'] = P.mean_abs > P.null_q95
from statsmodels.stats.multitest import multipletests
P['q_BH'] = multipletests(P.p_emp, method='fdr_bh')[1]
P['sig_BH'] = P.q_BH < 0.05
P['ratio_to_null'] = P.mean_abs / P.null_mean_abs
P['sign_resolved'] = (P.signed_lo > 0) | (P.signed_hi < 0)
P = P.sort_values('mean_abs', ascending=False)
pd.set_option('display.width', 250); print(P.round(4))
P.to_csv('results/table6_interactions.csv', index=False)
# old 70th-percentile rule for comparison
thr70 = np.percentile(P.mean_abs, 70); out['thr70'] = thr70
# Spearman: is interaction magnitude just a restatement of main-effect importance?
prod_imp = [imp[FEED[a]] * imp[FEED[b]] for a, b in pairs]
rho = stats.spearmanr(obs[:, 1], prod_imp); out['rho_int_vs_importance'] = [rho.statistic, rho.pvalue]
print('Spearman |int| vs product of main-effect importances:', rho)

# ---------------- Network from pairs exceeding the additive null ----------------
G = nx.Graph(); G.add_nodes_from([SYM[f] for f in FEED])
for _, r in P[P.sig_BH].iterrows(): G.add_edge(r.i, r.j, weight=r.mean_abs, sign=np.sign(r.signed_mean))
cent = pd.DataFrame({'degree': nx.degree_centrality(G),
                     'strength': dict(G.degree(weight='weight'))}).sort_values('strength', ascending=False)
print(cent.round(3)); cent.to_csv('results/network_centrality.csv')
pickle.dump(G, open('results/network.pkl', 'wb'))

# ---------------- Climate & operational contribution ----------------
base_rmse = np.sqrt(mean_squared_error(yt, pred))
perm = {}
for g, f in grp.items():
    d = []
    for r in range(30):
        Xp = Xt.copy(); Xp[f] = Xp[f].values[rng.permutation(len(Xp))]
        d.append(np.sqrt(mean_squared_error(yt, model.predict(Xp))) - base_rmse)
    perm[g] = (np.mean(d), np.std(d))
print('grouped permutation dRMSE:', perm)
drop = {}
for name, feats in {'All': FEATS, 'Without climate': FEED + OPER, 'Without operational': FEED + CLIM,
                    'Feedstock only': FEED}.items():
    m = mk(bp).fit(Xd[feats], yd); p = m.predict(Xt[feats])
    drop[name] = dict(R2=r2_score(yt, p), RMSE=np.sqrt(mean_squared_error(yt, p)))
print('drop-group retraining:', pd.DataFrame(drop).T.round(4))
out.update(perm=perm, drop=drop, base_rmse=base_rmse)

# ---------------- Seasonal analysis on Indian (IMD) seasons ----------------
def imd(m): return 'Winter' if m in (1, 2) else 'Pre-monsoon' if m in (3, 4, 5) else 'Monsoon' if m in (6, 7, 8, 9) else 'Post-monsoon'
daily = df.groupby('Date').agg(y=(TARGET, 'mean'), Ta=('Temperature (C)', 'mean'), R=('Rainfall (mm)', 'mean')).reset_index()
daily['Season'] = daily.Date.dt.month.map(imd)
seas = {}
for v in ['y', 'Ta', 'R']:
    groups = [g[v].values for _, g in daily.groupby('Season')]
    H = stats.kruskal(*groups); n = len(daily); eps2 = (H.statistic - 4 + 1) / (n - 4)
    seas[v] = dict(H=H.statistic, p=H.pvalue, eps2=eps2,
                   means=daily.groupby('Season')[v].mean().to_dict(), sds=daily.groupby('Season')[v].std().to_dict(),
                   dunn=sp.posthoc_dunn(daily, val_col=v, group_col='Season', p_adjust='holm').round(4).to_dict())
    print('Kruskal-Wallis', v, round(H.statistic, 3), 'p', round(H.pvalue, 4), 'eps2', round(eps2, 5))
out['seasonal'] = seas
daily.to_csv('results/daily_seasonal.csv', index=False)

# ---------------- Local explanation cases (selected on the explained set) ----------------
res = yt - pred
cases = {'Highest predicted yield': int(np.argmax(pred)), 'Lowest predicted yield': int(np.argmin(pred)),
         'Largest absolute error': int(np.argmax(np.abs(res)))}
out['cases'] = {k: dict(row=i, date=str(test.Date.iloc[i].date()), y=float(yt[i]), f=float(pred[i])) for k, i in cases.items()}
print(out['cases'])
pickle.dump(dict(sv=sv, base=base, pred=pred, yt=yt, Xt=Xt, siv_sub=sub, cases=cases, imp=imp), open('results/shap.pkl', 'wb'))
json.dump(out, open('results/analysis.json', 'w'), indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o))
