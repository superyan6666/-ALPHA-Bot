import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
import logging
from feature_engine import build_ml_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def load_and_prepare_data():
    parquet_path = ".quantbot_data/ashare_daily.parquet"
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Data lake not found at {parquet_path}")
    
    log.info("Loading data lake...")
    panel = pd.read_parquet(parquet_path)
    panel = panel.rename(columns={'date': 'date', 'code': 'code', 'open': 'open', 'close': 'close', 'high': 'high', 'low': 'low', 'volume': 'vol'})
    panel['date'] = pd.to_datetime(panel['date'])
    panel['code'] = panel['code'].str.replace('sh.', '', regex=False).str.replace('sz.', '', regex=False).str.replace('bj.', '', regex=False)
    panel = panel.sort_values(['code', 'date']).reset_index(drop=True)
    
    # Preprocess
    for col in ['open', 'high', 'low', 'close', 'vol']:
        panel[col] = pd.to_numeric(panel[col], errors='coerce')
    panel['prev_close'] = panel.groupby('code')['close'].shift(1)
    panel['pct_chg'] = (panel['close'] / panel['prev_close'] - 1) * 100
    panel['is_limit'] = (panel['pct_chg'].abs() >= 9.5) & (panel['high'] == panel['low'])
    
    # Targets
    panel['fwd_ret_t1'] = (panel.groupby('code')['open'].shift(-2) / panel.groupby('code')['open'].shift(-1) - 1) * 100
    panel['fwd_ret_t5'] = (panel.groupby('code')['open'].shift(-6) / panel.groupby('code')['open'].shift(-1) - 1) * 100
    panel['fwd_ret_t10'] = (panel.groupby('code')['open'].shift(-11) / panel.groupby('code')['open'].shift(-1) - 1) * 100
    panel['fwd_ret_t20'] = (panel.groupby('code')['open'].shift(-21) / panel.groupby('code')['open'].shift(-1) - 1) * 100
    panel['fwd_ret_t40'] = (panel.groupby('code')['open'].shift(-41) / panel.groupby('code')['open'].shift(-1) - 1) * 100
    panel['fwd_ret_t60'] = (panel.groupby('code')['open'].shift(-61) / panel.groupby('code')['open'].shift(-1) - 1) * 100
    panel['fwd_ret_t120'] = (panel.groupby('code')['open'].shift(-121) / panel.groupby('code')['open'].shift(-1) - 1) * 100
    
    # Extract features
    panel = build_ml_features(panel)
    return panel

def select_features_for_horizon(df, target_col, all_features, train_dates, optimal_horizon_map=None):
    log.info(f"Selecting features for {target_col}...")
    
    # Apply optimal horizon mapping filtering if provided
    if optimal_horizon_map is not None:
        horizon_name = target_col.replace('fwd_ret_t', 'T+')
        valid_features = [f for f in all_features if optimal_horizon_map.get(f) == horizon_name]
        log.info(f"Pre-filtered from {len(all_features)} down to {len(valid_features)} features assigned to {horizon_name}.")
        if len(valid_features) == 0:
            log.warning(f"No features optimally assigned to {horizon_name}! Falling back to all features.")
            valid_features = all_features
    else:
        valid_features = all_features
    
    # 1. Filter to first WFO training window to prevent lookahead bias
    df_train = df[df['date'].isin(train_dates)].copy()
    
    # Pre-clean: Drop any features that are completely NaN in this subset (e.g. deprecated macro features)
    empty_features = [f for f in valid_features if df_train[f].isna().all()]
    if empty_features:
        log.warning(f"Dropping entirely NaN features: {empty_features}")
        valid_features = [f for f in valid_features if f not in empty_features]
        
    # 2. Delayed Dropna: only dropna for the specific target and features
    df_train = df_train.dropna(subset=valid_features + [target_col])
    
    if len(df_train) == 0:
        log.error(f"No training data for {target_col} after dropna!")
        return valid_features
        
    X = df_train[valid_features].values
    y = df_train[target_col].values
    
    # 3. Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        tree_method='hist',
        importance_type='gain',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    # 4. Extract Importances
    gains = model.feature_importances_
    feature_gains = pd.DataFrame({
        'feature': valid_features,
        'gain': gains
    }).sort_values('gain', ascending=False).reset_index(drop=True)
    
    # 5. Correlation Filter (> 0.9)
    log.info("Applying correlation filter...")
    corr_matrix = df_train[feature_gains['feature']].corr().abs()
    
    selected_features = []
    dropped_features = []
    
    for f in feature_gains['feature']:
        if f in dropped_features:
            continue
        selected_features.append(f)
        # Find highly correlated features and drop them
        high_corr = corr_matrix.index[corr_matrix[f] > 0.9].tolist()
        for hc in high_corr:
            if hc != f and hc not in dropped_features and hc not in selected_features:
                dropped_features.append(hc)
                log.info(f"  Dropped {hc} due to high correlation with {f}")
                
    # 6. Cumulative Gain Truncation (85%) - Downgraded to Warning Only (B8 Policy)
    filtered_gains = feature_gains[feature_gains['feature'].isin(selected_features)].copy()
    filtered_gains['gain_pct'] = filtered_gains['gain'] / filtered_gains['gain'].sum()
    filtered_gains['cum_gain'] = filtered_gains['gain_pct'].cumsum()
    
    # Select features that contribute to the top 85% of cumulative gain, but at least 5 features
    top_features = filtered_gains[filtered_gains['cum_gain'] <= 0.85]['feature'].tolist()
    if len(top_features) < 5:
        top_features = filtered_gains['feature'].head(5).tolist()
        
    dropped_by_gain = [f for f in selected_features if f not in top_features]
    if dropped_by_gain:
        log.warning(f"Adaptive Warning: The following {len(dropped_by_gain)} features fall outside the Top 85% gain: {dropped_by_gain}. Keeping them anyway per Systematization Policy.")
        
    # Return all selected_features (after correlation filter) instead of truncating!
    log.info(f"Selected {len(selected_features)} features for {target_col} (after correlation filter): {selected_features}")
    return selected_features

def main():
    panel = load_and_prepare_data()
    
    optimal_horizon_path = '.quantbot_data/factor_optimal_horizons.csv'
    optimal_horizon_map = None
    # 🚨 Rollback: Disabled Strict Horizon Isolation due to B11.3 Performance Regression 🚨
    # if os.path.exists(optimal_horizon_path):
    #     opt_df = pd.read_csv(optimal_horizon_path)
    #     optimal_horizon_map = dict(zip(opt_df['feature'], opt_df['best_horizon']))
    #     log.info(f"Loaded optimal horizon mapping for {len(optimal_horizon_map)} features.")
        
    # Get all columns starting with F_
    all_features = [c for c in panel.columns if c.startswith('F_')]
    log.info(f"Found {len(all_features)} total features: {all_features}")
    
    dates = sorted(panel['date'].dropna().unique())
    # First 500 dates corresponds to the first 2-year window in WFO
    train_dates = dates[:500] 
    
    horizons = [1, 5, 10, 20, 40, 60, 120]
    horizon_features = {}
    
    for h in horizons:
        target_col = f'fwd_ret_t{h}'
        selected = select_features_for_horizon(panel, target_col, all_features, train_dates, optimal_horizon_map)
        horizon_features[f"T+{h}"] = selected
        
    # Save to JSON
    os.makedirs('.quantbot_data', exist_ok=True)
    out_path = '.quantbot_data/horizon_features.json'
    with open(out_path, 'w') as f:
        json.dump(horizon_features, f, indent=4)
    log.info(f"Saved feature dictionary to {out_path}")

if __name__ == "__main__":
    main()
