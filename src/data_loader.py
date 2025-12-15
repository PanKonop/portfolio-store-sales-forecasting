import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path

def load_raw_data(data_path: str):
    """Wczytuje surowe pliki CSV."""
    path = Path(data_path)
    # Wczytujemy tylko to, co niezbędne na start
    df_train = pd.read_csv(path / 'train.csv', parse_dates=['date'])
    df_test = pd.read_csv(path / 'test.csv', parse_dates=['date'])
    print("✅ Dane surowe wczytane.")
    return df_train, df_test

def create_training_grid(df_train):
    """Tworzy pełną siatkę dat i uzupełnia braki (np. 25 grudnia)."""
    # 1. Unikalne wartości
    unique_dates = pd.date_range(start=df_train.date.min(), end=df_train.date.max())
    unique_stores = df_train.store_nbr.unique()
    unique_families = df_train.family.unique()
    
    # 2. Iloczyn kartezjański
    cartesian = product(unique_dates, unique_stores, unique_families)
    df_grid = pd.DataFrame(cartesian, columns=['date', 'store_nbr', 'family'])
    
    # 3. Łączenie
    df_full = df_grid.merge(df_train, on=['date', 'store_nbr', 'family'], how='left')
    
    # 4. Wypełnianie zerami
    df_full['sales'] = df_full['sales'].fillna(0)
    df_full['onpromotion'] = df_full['onpromotion'].fillna(0)
    
    # Nie potrzebujemy id w treningu
    if 'id' in df_full.columns:
        df_full = df_full.drop(columns=['id'])
        
    print(f"✅ Grid stworzony. Rozmiar: {df_full.shape}")
    return df_full