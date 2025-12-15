import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from src.data_loader import load_raw_data, create_training_grid
from src.features import create_date_features, encode_categoricals, create_lag_features


# Dodaj bieżący katalog do ścieżki systemowej Pythona
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Dopiero teraz Twoje importy
from src.data_loader import load_raw_data, create_training_grid
# ... reszta importów

# Konfiguracja
DATA_PATH = r"data" # Ścieżka relatywna
MODEL_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.02,
    'max_depth': 12,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'n_jobs': -1,
    'random_state': 42,
    # 'early_stopping_rounds': 50
}

def run_pipeline():
    print("🚀 Rozpoczynam proces treningu...")
    
    # 1. Load Data
    df_train_raw, df_test_raw = load_raw_data(DATA_PATH)
    
    # 2. Prepare Train Grid
    print("🛠️ Przygotowanie danych treningowych...")
    df_train = create_training_grid(df_train_raw)
    
    # 3. Concatenate for Feature Engineering
    # Przygotowanie testu do połączenia
    df_test_prep = df_test_raw[['id', 'date', 'store_nbr', 'family', 'onpromotion']].copy()
    df_test_prep['sales'] = np.nan
    
    # Train nie ma ID, bierzemy kolumny biznesowe
    df_train_prep = df_train[['date', 'store_nbr', 'family', 'sales', 'onpromotion']].copy()
    
    df_concat = pd.concat([df_train_prep, df_test_prep], axis=0).reset_index(drop=True)
    df_concat['sales'] = df_concat['sales'].fillna(0)
    
    # 4. Feature Engineering
    print("🧠 Inżynieria cech...")
    df_concat = create_date_features(df_concat)
    df_concat, le_fam, le_store = encode_categoricals(df_concat) # Fit encoders on full data
    df_concat['sales_log'] = np.log1p(df_concat['sales'])
    
    df_features = create_lag_features(df_concat)
    
    # 5. Split back
    test_start_date = '2017-08-16'
    features_list = [
        'onpromotion', 'day_of_week', 'is_weekend', 'is_wage_day', 'day_of_year',
        'family_idx', 'store_idx',
        'lag_16', 'lag_21', 'lag_365', 'rolling_mean_14', 'rolling_std_14'
    ]
    
    train_final = df_features[df_features['date'] < test_start_date].dropna(subset=features_list)
    test_final = df_features[df_features['date'] >= test_start_date]
    
    # 6. Train Model
    print("🏋️ Trenowanie modelu XGBoost...")
    X_train = train_final[features_list]
    y_train = train_final['sales_log']
    
    model = xgb.XGBRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train, verbose=False) # Bez walidacji tutaj, trenujemy na wszystkim
    
    # 7. Predict & Submission
    print("🔮 Predykcja...")
    preds_log = model.predict(test_final[features_list])
    preds_log = np.maximum(preds_log, 0)
    preds_real = np.expm1(preds_log)
    
    submission = pd.DataFrame({
        'id': test_final['id'].astype(int),
        'sales': preds_real
    })
    
    submission.to_csv(os.path.join(DATA_PATH, 'submission_pipeline.csv'), index=False)
    print("✅ Gotowe! Plik zapisany jako submission_pipeline.csv")

if __name__ == "__main__":
    run_pipeline()