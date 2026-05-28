import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np

from ml_engine import XGBoostLTR, apply_liquidity_gate
from feature_engine import build_ml_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def train_production_model():
    log.info("Starting Production Model Training Pipeline...")
    
    ashare_path = '.quantbot_data/ashare_daily.parquet'
    if not os.path.exists(ashare_path):
        log.error(f"Cannot find {ashare_path}")
        return
        
    log.info(f"Loading data lake: {ashare_path}")
    panel = pd.read_parquet(ashare_path)
    
    if 'volume' in panel.columns and 'vol' not in panel.columns:
        panel.rename(columns={'volume': 'vol'}, inplace=True)
    
    # 1. Prepare targets for multiple horizons
    log.info("Preparing multi-horizon targets...")
    panel['fwd_ret_t1'] = panel.groupby('code')['close'].shift(-1) / (panel['close'] + 1e-5) - 1
    panel['fwd_ret_t5'] = panel.groupby('code')['close'].shift(-5) / (panel['close'] + 1e-5) - 1
    panel['fwd_ret_t10'] = panel.groupby('code')['close'].shift(-10) / (panel['close'] + 1e-5) - 1
    
    # 2. Build Features using DRY engine
    panel = build_ml_features(panel)
    
    # Apply Liquidity Gate
    panel = apply_liquidity_gate(panel, amihud_col='amihud_20', threshold_pct=0.90)
    panel = panel[~panel['is_limit']].copy() if 'is_limit' in panel.columns else panel
    
    feature_cols = ['sm_corr', 'clv', 'volatility_5d', 'vol_ratio', 'alpha_reversal_5d', 'alpha_024_approx',
                    'market_ret_20d', 'market_ret_60d', 'market_vol_20d', 'cn_10y_trend']
                    
    horizons = [1, 5, 10]
    for h in horizons:
        target_col = f'fwd_ret_t{h}'
        log.info(f"\n{'='*40}\nTraining model for Horizon T+{h}\n{'='*40}")
        
        # Drop NaNs for this specific horizon target
        ml_df = panel.dropna(subset=feature_cols + [target_col, 'date']).copy()
        
        dates = sorted(ml_df['date'].unique())
        train_window = 500
        if len(dates) < train_window:
            log.warning(f"Not enough dates to train. Expected {train_window}, got {len(dates)}")
            train_window = len(dates)
            
        train_dates = dates[-train_window:]
        log.info(f"[T+{h}] Training on rolling window: {train_dates[0].strftime('%Y-%m-%d')} to {train_dates[-1].strftime('%Y-%m-%d')} ({len(train_dates)} days)")
        
        train_df = ml_df[ml_df['date'].isin(train_dates)].copy()
        
        ltr = XGBoostLTR()
        ltr.train(train_df, feature_cols, target_col=target_col, group_col='date')
        
        # Print minimal evaluation summary (Top 5 features)
        imp_df = ltr.get_feature_importance(feature_cols)
        log.info(f"[T+{h}] Top 5 Feature Importances (Gain):")
        for _, row in imp_df.head(5).iterrows():
            log.info(f"  {row['feature']}: {row['importance']:.4f}")
            
        mean_ic = np.nan
        # Predict on the last 30 days to compute Rank IC proxy
        if len(dates) > 30:
            val_dates = dates[-30:]
            val_df = ml_df[ml_df['date'].isin(val_dates)].copy()
            val_df['xgb_score'] = ltr.predict(val_df, feature_cols)
            
            # Rank IC calculation
            def calc_ic(g):
                return g['xgb_score'].corr(g[target_col], method='spearman')
                
            ic_series = val_df.groupby('date').apply(calc_ic)
            mean_ic = ic_series.mean()
            log.info(f"[T+{h}] Recent 30-day Mean Rank IC proxy: {mean_ic:.4f}")
            
        os.makedirs('.quantbot_data', exist_ok=True)
        model_path = f'.quantbot_data/prod_xgb_model_t{h}.json'
        meta_path = f'.quantbot_data/prod_xgb_meta_t{h}.json'
        
        ltr.save_model(model_path)
        
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'train_start_date': train_dates[0].strftime('%Y-%m-%d'),
            'train_end_date': train_dates[-1].strftime('%Y-%m-%d'),
            'num_days': len(train_dates),
            'num_samples': len(train_df),
            'features': feature_cols,
            'horizon': h,
            'mean_ic_30d': float(mean_ic) if pd.notna(mean_ic) else None
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        log.info(f"Production model and metadata for T+{h} saved successfully.")

if __name__ == '__main__':
    train_production_model()
