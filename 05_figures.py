"""Step 5: all manuscript and supplementary figures, drawn only from saved model outputs."""
import os; os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
import json, pickle, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, networkx as nx, shap
from scipy import stats
from common import *
warnings.filterwarnings('ignore')

BLUE, ORANGE, GREY, INK = '#2a78d6', '#eb6834', '#8a8984', '#0b0b0b'
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
                     'axes.grid': False, 'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 10,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
                     'xtick.direction': 'out', 'ytick.direction': 'out', 'legend.frameon': False,
                     'savefig.dpi': 600, 'savefig.bbox': 'tight'})
S = pickle.load(open('results/shap.pkl', 'rb')); A = json.load(open('results/analysis.json'))
yt, pred, sv, Xt, base = S['yt'], S['pred'], S['sv'], S['Xt'], S['base']
df = load(); dev, test = date_split(df)
lab = [SYM[f] for f in FEATS]
U = 'm$^3$'
def nogrid(fig):
    for ax_ in fig.axes:
        for ln in list(ax_.lines):
            if ln.get_linestyle() in (':', '--') and len(ln.get_xdata())==2 and ln.get_xdata()[0]==ln.get_xdata()[1]==ln.get_xdata()[0] and False: pass
        for ln in list(ax_.lines):
            yd_ = ln.get_ydata()
            if ln.get_linestyle() in (':', '--') and len(yd_) == 2 and yd_[0] == yd_[1]: ln.remove()

def save(fig, name): fig.savefig(f'figures/{name}.png'); fig.savefig(f'figures/{name}.pdf'); plt.close(fig)

# ---------- Fig 2: diagnostics ----------
res = yt - pred
fig, ax = plt.subplots(2, 2, figsize=(7.2, 6.2))
a = ax[0, 0]; a.scatter(yt, pred, s=4, color=BLUE, alpha=0.5, lw=0)
lo, hi = 40, 125; a.plot([lo, hi], [lo, hi], color=INK, lw=0.8, ls='--')
a.set(xlim=(lo, hi), ylim=(lo, hi), xlabel=f'Observed ({U})', ylabel=f'Predicted ({U})', aspect='equal')
a = ax[0, 1]; a.scatter(pred, res, s=4, color=BLUE, alpha=0.5, lw=0)
a.axhline(0, color=INK, lw=0.8, ls='--'); sd = res.std()
for k in (-2, 2): a.axhline(k * sd, color=GREY, lw=0.8, ls=':')
a.set(xlabel=f'Predicted ({U})', ylabel=f'Residual ({U})')
a = ax[1, 0]; a.hist(res, bins=50, color=BLUE, edgecolor='white', lw=0.3, density=True)
xx = np.linspace(res.min(), res.max(), 200); a.plot(xx, stats.norm.pdf(xx, res.mean(), sd), color=ORANGE, lw=1.5)
a.set(xlabel=f'Residual ({U})', ylabel='Density')
a = ax[1, 1]
d = pd.DataFrame({'Date': test.Date, 'y': yt, 'p': pred}).groupby('Date').mean().iloc[:120]
a.plot(d.index, d.y, color=INK, lw=1.0, label='Observed'); a.plot(d.index, d.p, color=ORANGE, lw=1.0, ls='--', label='Predicted')
a.set(ylabel=f'Daily mean ({U})'); a.legend(loc='upper right', fontsize=8)
import matplotlib.dates as mdates; a.xaxis.set_major_locator(mdates.MonthLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y')); a.tick_params(axis='x', labelsize=8)
for i, a in enumerate(ax.flat): a.text(-0.2, 1.02, f'({"abcd"[i]})', transform=a.transAxes, fontsize=10)
fig.tight_layout(); save(fig, 'Figure_2')
print('Fig2 residual mean %.3f sd %.3f skew %.3f' % (res.mean(), sd, stats.skew(res)))

# ---------- Fig 3: SHAP beeswarm ----------
plt.figure()
shap.summary_plot(sv, Xt.values, feature_names=lab, show=False, max_display=18, plot_size=(6, 6.5),
                  color_bar_label='Feature value')
f = plt.gcf(); f.axes[0].set_xlabel(f'SHAP value ({U})'); nogrid(f); save(f, 'Figure_3')

# ---------- Fig 4: interaction matrix and comparison with additive null ----------
P = pd.read_csv('results/table6_interactions.csv')
fs = [SYM[f] for f in FEED]; M = pd.DataFrame(0.0, index=fs, columns=fs); Sg = M.copy()
for _, r in P.iterrows(): M.loc[r.i, r.j] = M.loc[r.j, r.i] = r.mean_abs; Sg.loc[r.i, r.j] = Sg.loc[r.j, r.i] = r.signed_mean
fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw={'width_ratios': [1, 1.15]})
mask = np.triu(np.ones_like(M, dtype=bool))
im = ax[0].imshow(np.ma.masked_array(M.values, mask), cmap='Blues', vmin=0)
ax[0].set_xticks(range(10), fs, rotation=90); ax[0].set_yticks(range(10), fs)
for s in ax[0].spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax[0], shrink=0.75); cb.set_label(r'Mean $|\Phi_{ij}|$ (' + U + ')'); cb.outline.set_visible(False)
top = P.head(15).iloc[::-1]
yy = np.arange(len(top))
ax[1].barh(yy, top.mean_abs, color=BLUE, height=0.6, label='Observed')
ax[1].errorbar(top.mean_abs, yy, xerr=[top.mean_abs - top.abs_lo, top.abs_hi - top.mean_abs], fmt='none', ecolor=INK, lw=0.8)
ax[1].scatter(top.null_q95, yy, marker='|', s=120, color=ORANGE, lw=1.6, zorder=3, label='Additive null, 95th pct.')
ax[1].set_yticks(yy, [f'{a}–{b}' for a, b in zip(top.i, top.j)])
ax[1].set_xlabel(r'Mean $|\Phi_{ij}|$ (' + U + ')'); ax[1].legend(loc='lower right', fontsize=8)
for i, a in enumerate(ax): a.text(-0.18, 1.02, f'({"ab"[i]})', transform=a.transAxes)
fig.tight_layout(); save(fig, 'Figure_5')

# ---------- Fig 6: group contributions ----------
gs = A['group_share']; perm = A['perm']; drop = A['drop']
fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.0))
g = ['Feedstock', 'Operational', 'Climate']; cols = [BLUE, GREY, ORANGE]
ax[0].bar(g, [float(gs[k]) for k in g], color=cols, width=0.6); ax[0].set_ylabel('Share of total mean |SHAP| (%)')
pm = [eval(perm[k]) if isinstance(perm[k], str) else perm[k] for k in g]
ax[1].bar(g, [p[0] for p in pm], yerr=[p[1] for p in pm], color=cols, width=0.6, ecolor=INK)
ax[1].set_ylabel(f'Permutation $\\Delta$RMSE ({U})')
dk = ['All', 'Without climate', 'Without operational', 'Feedstock only']
dd = [eval(drop[k]) if isinstance(drop[k], str) else drop[k] for k in dk]
ax[2].bar(['All', '−Climate', '−Oper.', 'Feed only'], [x['RMSE'] for x in dd], color=[INK, ORANGE, GREY, BLUE], width=0.6)
ax[2].set_ylabel(f'Test RMSE ({U})')
for i, a in enumerate(ax): a.text(-0.25, 1.02, f'({"abc"[i]})', transform=a.transAxes); a.tick_params(axis='x', labelsize=8)
fig.tight_layout(); save(fig, 'Figure_6')

# ---------- Fig 7: SHAP dependence, top six ----------
imp = S['imp']; top6 = list(imp.index[:6])
fig, ax = plt.subplots(2, 3, figsize=(9.0, 5.4), sharey=False)
for k, (f_, a) in enumerate(zip(top6, ax.flat)):
    j = FEATS.index(f_)
    a.scatter(Xt[f_], sv[:, j], s=3, color=BLUE, alpha=0.4, lw=0)
    a.axhline(0, color=GREY, lw=0.6)
    a.set_xlabel(f'{SYM[f_]} (kg)'); a.set_ylabel(f'SHAP ({U})' if k % 3 == 0 else '')
    a.text(0.02, 0.92, f'({"abcdef"[k]})', transform=a.transAxes)
fig.tight_layout(); save(fig, 'Figure_4')

# ---------- Fig 8: waterfall, highest and lowest predicted yield ----------
for tag, key in [('a', 'Highest predicted yield'), ('b', 'Lowest predicted yield')]:
    i = S['cases'][key]
    e = shap.Explanation(values=sv[i], base_values=base, data=Xt.values[i].round(2), feature_names=lab)
    plt.figure(); shap.plots.waterfall(e, max_display=10, show=False)
    f = plt.gcf(); f.set_size_inches(8, 4.6); nogrid(f); save(f, f'Figure_7{tag}')

# ---------- Supplementary ----------
# S1: distributions
cols_ = FEATS + [TARGET]
fig, ax = plt.subplots(4, 5, figsize=(11, 8))
for c, a in zip(cols_, ax.flat):
    a.hist(df[c], bins=40, color=BLUE, edgecolor='white', lw=0.2); a.set_title(SYM.get(c, 'Biogas'), fontsize=9)
    a.tick_params(labelsize=7); a.set_yticks([])
ax.flat[-1].axis('off'); fig.tight_layout(); save(fig, 'Figure_S2')
# S2: records per date and monthly climate means
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
rpd = df.groupby('Date').size().value_counts().sort_index()
ax[0].bar(rpd.index.astype(str), rpd.values, color=BLUE, width=0.6); ax[0].set(xlabel='Records per date', ylabel='Number of dates')
mm = df.groupby('Month')[['Temperature (C)', 'Rainfall (mm)', TARGET]].agg(['mean', 'std'])
ax[1].errorbar(mm.index, mm[('Temperature (C)', 'mean')], yerr=mm[('Temperature (C)', 'std')], color=ORANGE, fmt='o-', ms=3, lw=1)
ax[1].set(xlabel='Month', ylabel='T$_a$ (°C)', xticks=range(1, 13))
ax[2].errorbar(mm.index, mm[(TARGET, 'mean')], yerr=mm[(TARGET, 'std')], color=BLUE, fmt='o-', ms=3, lw=1)
ax[2].set(xlabel='Month', ylabel=f'Biogas ({U})', xticks=range(1, 13))
for i, a in enumerate(ax): a.text(-0.2, 1.02, f'({"abc"[i]})', transform=a.transAxes)
fig.tight_layout(); save(fig, 'Figure_S1')
# S3: seasonal boxplots (IMD seasons)
daily = pd.read_csv('results/daily_seasonal.csv'); order = ['Winter', 'Pre-monsoon', 'Monsoon', 'Post-monsoon']
fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
for a, v, yl in [(ax[0], 'y', f'Daily mean biogas ({U})'), (ax[1], 'Ta', 'Daily mean T$_a$ (°C)')]:
    a.boxplot([daily[daily.Season == s][v] for s in order], tick_labels=order, widths=0.5, showfliers=False,
              medianprops=dict(color=ORANGE), boxprops=dict(color=INK), whiskerprops=dict(color=INK), capprops=dict(color=INK))
    a.set_ylabel(yl); a.tick_params(axis='x', labelsize=8)
for i, a in enumerate(ax): a.text(-0.15, 1.02, f'({"ab"[i]})', transform=a.transAxes)
fig.tight_layout(); save(fig, 'Figure_S3')
# S4: CV fold performance per model
print('figures done')
