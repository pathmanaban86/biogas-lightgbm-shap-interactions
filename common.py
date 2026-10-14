"""Shared data loading and split definitions for the revised biogas analysis."""
import numpy as np, pandas as pd

FEED = ['Pig Manure (kg)', 'Kitchen Food Waste (kg)', 'Chicken Litter (kg)', 'Cassava (kg)',
        'Bagasse Feed (kg)', 'Energy Grass (kg)', 'Banana Shafts (kg)', 'Alcohol Waste (kg)',
        'Municipal Residue (kg)', 'Fish Waste (kg)']
OPER = ['Water (L)', 'Diesel (L)', 'Electricity Use (kWh)', 'C/N Ratio', 'Digester Temp (C)']
CLIM = ['Temperature (C)', 'Humidity (%)', 'Rainfall (mm)']
FEATS = FEED + OPER + CLIM
TARGET = 'biogas_production'
SEED = 42

# Short symbols used consistently in all figures and tables
SYM = {'Pig Manure (kg)': 'PM', 'Kitchen Food Waste (kg)': 'KFW', 'Chicken Litter (kg)': 'CL',
       'Cassava (kg)': 'CAS', 'Bagasse Feed (kg)': 'BF', 'Energy Grass (kg)': 'EG',
       'Banana Shafts (kg)': 'BS', 'Alcohol Waste (kg)': 'AW', 'Municipal Residue (kg)': 'MR',
       'Fish Waste (kg)': 'FW', 'Water (L)': 'W', 'Diesel (L)': 'D', 'Electricity Use (kWh)': 'E',
       'C/N Ratio': 'C/N', 'Digester Temp (C)': 'T$_d$', 'Temperature (C)': 'T$_a$',
       'Humidity (%)': 'RH', 'Rainfall (mm)': 'R'}


def load():
    df = pd.read_csv('data/biogas_dataset.csv')
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    # 97 humidity records exceed the physical limit of 100 %; capped at 100 % (documented in paper)
    df['Humidity (%)'] = df['Humidity (%)'].clip(upper=100.0)
    return df.sort_values('Date', kind='stable').reset_index(drop=True)


def date_split(df, frac=0.8):
    """Chronological split on whole dates so no date appears in both sets."""
    dates = np.sort(df.Date.unique())
    cut = dates[int(frac * len(dates))]
    dev = df[df.Date < cut].reset_index(drop=True)
    test = df[df.Date >= cut].reset_index(drop=True)
    return dev, test


def ts_folds(dev, k=5):
    """Expanding-window folds on whole dates (sklearn TimeSeriesSplit logic on unique dates)."""
    from sklearn.model_selection import TimeSeriesSplit
    dates = np.sort(dev.Date.unique())
    folds = []
    for tr, va in TimeSeriesSplit(n_splits=k).split(dates):
        tri = np.where(dev.Date.isin(dates[tr]))[0]
        vai = np.where(dev.Date.isin(dates[va]))[0]
        folds.append((tri, vai))
    return folds
