
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import xgboost as xgb
from feature_engine import build_ml_features
import json

def main():
    print("Loading data...")
    panel = pd.read_parquet(".quantbot_data/ashare_daily.parquet")
    panel = panel.rename(columns={'date': 'date', 'code': 'code', 'open': 'open', 'close': 'close', 'high': 'high', 'low': 'low', 'volume': 'vol'})
    panel['date'] = pd.to_datetime(panel['date'])
    for col in ['open', 'high', 'low', 'close', 'vol']:
        panel[col] = pd.to_numeric(panel[col], errors='coerce')
    panel['prev_close'] = panel.groupby('code')['close'].shift(1)
    panel['fwd_ret_t1'] = (panel.groupby('code')['open'].shift(-2) / panel.groupby('code')['open'].shift(-1) - 1) * 100
    panel['fwd_ret_t5'] = (panel.groupby('code')['open'].shift(-6) / panel.groupby('code')['open'].shift(-1) - 1) * 100
    panel['fwd_ret_t10'] = (panel.groupby('code')['open'].shift(-11) / panel.groupby('code')['open'].shift(-1) - 1) * 100
    panel['fwd_ret_t20'] = (panel.groupby('code')['open'].shift(-21) / panel.groupby('code')['open'].shift(-1) - 1) * 100
    
    panel = build_ml_features(panel)
    
    with open('.quantbot_data/horizon_features.json', 'r') as f:
        horizon_features = json.load(f)
        
    dates = sorted(panel['date'].dropna().unique())
    train_dates = dates[:500] 
    df_train = panel[panel['date'].isin(train_dates)].copy()
    
    results = {}
    for h in [1, 5, 10, 20]:
        target_col = f'fwd_ret_t{h}'
        features = horizon_features[f"T+{h}"]
        df_sub = df_train.dropna(subset=features + [target_col])
        X = df_sub[features].values
        y = df_sub[target_col].values
        model = xgb.XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, tree_method='hist', importance_type='gain', random_state=42, n_jobs=-1)
        model.fit(X, y)
        gains = model.feature_importances_
        gains = gains / gains.sum()
        res = []
        for f, g in zip(features, gains):
            res.append((f, g))
        res.sort(key=lambda x: x[1], reverse=True)
        results[f"T+{h}"] = res
        
    for h, feats in results.items():
        print(f"\n=== {h} Horizon Factor Contribution Rate ===")
        for f, g in feats:
            print(f"{f}: {g*100:.2f}%")

if __name__ == "__main__":
    main()
