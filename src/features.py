import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def create_date_features(df):
    """Generuje cechy kalendarzowe."""
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    # Specyfika Ekwadoru (wypłaty 15-go i ostatniego)
    df['is_wage_day'] = ((df['day'] == 15) | (df['date'].dt.is_month_end)).astype(int)
    return df

def encode_categoricals(df, le_family=None, le_store=None):
    """Label Encoding dla rodziny i sklepu."""
    if le_family is None:
        le_family = LabelEncoder()
        df['family_idx'] = le_family.fit_transform(df['family'])
    else:
        df['family_idx'] = le_family.transform(df['family'])
        
    if le_store is None:
        le_store = LabelEncoder()
        df['store_idx'] = le_store.fit_transform(df['store_nbr'])
    else:
        df['store_idx'] = le_store.transform(df['store_nbr'])
        
    return df, le_family, le_store

def create_lag_features(df):
    """Tworzy opóźnienia i średnie ruchome (Time Series Magic)."""
    df = df.copy()
    # Sortowanie jest kluczowe dla shift()
    df = df.sort_values(['store_nbr', 'family', 'date'])
    
    groupby_obj = df.groupby(['store_nbr', 'family'])['sales_log']
    
    # Lags (bezpieczne dla horyzontu 16 dni)
    df['lag_16'] = groupby_obj.shift(16)
    df['lag_21'] = groupby_obj.shift(21)
    df['lag_365'] = groupby_obj.shift(365)
    
    # Rolling (przesunięte o 16 dni)
    df['rolling_mean_14'] = groupby_obj.transform(lambda x: x.shift(16).rolling(window=14).mean())
    df['rolling_std_14'] = groupby_obj.transform(lambda x: x.shift(16).rolling(window=14).std())
    
    return df