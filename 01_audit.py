"""Step 1: Data audit for the biogas dataset (reviewer comments on dates, Table 2)."""
import os; os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
import pandas as pd, numpy as np
from scipy import stats

df = pd.read_csv('data/biogas_dataset.csv')
print('shape', df.shape)
print('dup rows', df.duplicated().sum())
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
print('date min/max', df.Date.min(), df.Date.max())
span = (df.Date.max() - df.Date.min()).days + 1
print('calendar days in span', span, '| unique dates', df.Date.nunique())
rpd = df.groupby('Date').size()
print('rows per date distribution:\n', rpd.value_counts().sort_index())
print('is file sorted by date?', df.Date.is_monotonic_increasing)
print('rows per year:\n', df.groupby('Year').size())

# Descriptive statistics (Table 2)
cols = [c for c in df.columns if c not in ['Year', 'Month', 'Day', 'Date']]
t2 = pd.DataFrame({'mean': df[cols].mean(), 'sd': df[cols].std(), 'min': df[cols].min(),
                   'max': df[cols].max(), 'skew': df[cols].skew(), 'cv': df[cols].std() / df[cols].mean()})
pd.set_option('display.width', 200)
print(t2.round(3))
t2.round(3).to_csv('results/table2_descriptives.csv')

# Train/test split by position (original: first 80% rows)
n = len(df); s = int(0.8 * n)
print('train dates', df.Date.iloc[0].date(), '->', df.Date.iloc[s - 1].date())
print('test dates', df.Date.iloc[s].date(), '->', df.Date.iloc[-1].date())

# Trend check: target mean by year and by row position
print('target mean by year:\n', df.groupby('Year').biogas_production.mean().round(2))
print('target first/last 3060 rows mean', df.biogas_production.iloc[:3060].mean().round(2),
      df.biogas_production.iloc[-3060:].mean().round(2))
# lag-1 autocorrelation of daily-mean target
dm = df.groupby('Date').biogas_production.mean()
print('lag-1 ACF daily mean target', round(dm.autocorr(1), 3), 'lag-7', round(dm.autocorr(7), 3))
# correlations among predictors
X = df[cols[:-1]]
c = X.corr().values.copy(); np.fill_diagonal(c, np.nan)
print('max |r| among predictors', np.nanmax(np.abs(c)).round(3))
print('corr with target:\n', df[cols].corr()['biogas_production'].sort_values().round(3))
print('humidity >100%:', (df['Humidity (%)'] > 100).sum(), '| rainfall==0:', (df['Rainfall (mm)'] < 0.01).sum())
print('C/N at bounds 15:', (df['C/N Ratio'] <= 15.0001).sum(), 'digester at 30/42:', (df['Digester Temp (C)'] <= 30.0001).sum(), (df['Digester Temp (C)'] >= 41.9999).sum())
# ambient temperature by month (does climate vary seasonally?)
print('ambient T by month', df.groupby('Month')['Temperature (C)'].mean().round(2).tolist())
print('rain by month', df.groupby('Month')['Rainfall (mm)'].mean().round(2).tolist())
