"""Step 3: how much structure is beyond an additive/linear model?"""
import numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from common import *

df = load(); dev, test = date_split(df)
Xd, yd, Xt, yt = dev[FEATS], dev[TARGET], test[FEATS], test[TARGET]
ols = sm.OLS(yd, sm.add_constant(Xd)).fit()
print(ols.summary().tables[1])
res_d = yd - ols.predict(sm.add_constant(Xd)); res_t = yt - ols.predict(sm.add_constant(Xt))
print('OLS test R2', round(r2_score(yt, ols.predict(sm.add_constant(Xt))),4), 'resid SD', res_t.std().round(3))
# Can a flexible model predict the OLS residuals (non-linearity / interactions left over)?
m = LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, random_state=42, verbose=-1).fit(Xd, res_d)
print('LGBM on OLS residuals: test R2 of residual =', round(r2_score(res_t, m.predict(Xt)),4))
# Pairwise product terms: do any feedstock x feedstock products add explanatory power?
import itertools
rows = []
base = sm.OLS(yd, sm.add_constant(Xd)).fit()
for a, b in itertools.combinations(FEED, 2):
    Z = Xd.copy(); Z['ab'] = Xd[a] * Xd[b]
    f = sm.OLS(yd, sm.add_constant(Z)).fit()
    rows.append((SYM[a], SYM[b], f.params['ab'], f.pvalues['ab']))
r = pd.DataFrame(rows, columns=['a', 'b', 'coef', 'p']).sort_values('p')
r['p_bonf'] = np.minimum(r.p * len(r), 1)
print(r.head(10).round(5)); print('n pairs with Bonferroni p<0.05:', (r.p_bonf < 0.05).sum())
# quadratic terms
rows = []
for a in FEATS:
    Z = Xd.copy(); Z['aa'] = Xd[a] ** 2
    f = sm.OLS(yd, sm.add_constant(Z)).fit(); rows.append((SYM[a], f.params['aa'], f.pvalues['aa']))
q = pd.DataFrame(rows, columns=['f', 'coef', 'p']).sort_values('p'); print(q.head(6))
