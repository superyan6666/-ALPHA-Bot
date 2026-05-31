import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import torch.optim as optim

from ml_engine import PyTorchDLModel, apply_liquidity_gate
from feature_engine import build_ml_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def train_production_model():
    log.info("Starting Production PyTorch Model Training Pipeline...")
    
    with open('.quantbot_data/default_genes.json') as f:
        genes = json.load(f)
        
    feature_cols = genes['features']
    epochs = genes['model']['epochs']
    lr = genes['model']['learning_rate']
    batch_size = genes['model'].get('batch_size', 2048)
    
    log.info(f"Loaded Genes -> Features: {len(feature_cols)}, Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")
    
    ashare_path = '.quantbot_data/ashare_daily.parquet'
    if not os.path.exists(ashare_path):
        log.error(f"Cannot find {ashare_path}")
        return
        
    log.info(f"Loading data lake: {ashare_path}")
    panel = pd.read_parquet(ashare_path)
    
    if 'volume' in panel.columns and 'vol' not in panel.columns:
        panel.rename(columns={'volume': 'vol'}, inplace=True)
        
    # Standardize code
    panel['code'] = panel['code'].str.replace('sh.', '', regex=False)\
                                 .str.replace('sz.', '', regex=False)\
                                 .str.replace('bj.', '', regex=False)
    
    # 1. Prepare targets for multiple horizons
    log.info("Preparing multi-horizon targets...")
    panel = panel.sort_values(['code', 'date'])
    panel['fwd_ret_t1'] = panel.groupby('code')['close'].shift(-1) / (panel.groupby('code')['open'].shift(-1) + 1e-5) - 1
    panel['fwd_ret_t5'] = panel.groupby('code')['close'].shift(-5) / (panel.groupby('code')['open'].shift(-1) + 1e-5) - 1
    panel['fwd_ret_t10'] = panel.groupby('code')['close'].shift(-10) / (panel.groupby('code')['open'].shift(-1) + 1e-5) - 1
    
    # 2. Build Features using DRY engine
    panel = build_ml_features(panel)
    
    # Apply Liquidity Gate
    panel = apply_liquidity_gate(panel, amihud_col='amihud_20', threshold_pct=0.90)
    panel = panel[~panel['is_limit']].copy() if 'is_limit' in panel.columns else panel
    
    horizons = [1, 5, 10]
    for h in horizons:
        target_col = f'fwd_ret_t{h}'
        log.info(f"\n{'='*40}\nTraining PyTorch Model for Horizon T+{h}\n{'='*40}")
        
        # Drop NaNs for this specific horizon target
        ml_df = panel.dropna(subset=feature_cols + [target_col, 'date']).copy()
        
        dates = sorted(ml_df['date'].unique())
        # Production model uses ALL available data (no test split)
        train_dates = dates
        log.info(f"[T+{h}] Training on full data: {train_dates[0].strftime('%Y-%m-%d')} to {train_dates[-1].strftime('%Y-%m-%d')} ({len(train_dates)} days)")
        
        train_df = ml_df[ml_df['date'].isin(train_dates)].copy()
        
        dl_model = PyTorchDLModel(input_dim=len(feature_cols))
        dl_model.optimizer = optim.Adam(dl_model.model.parameters(), lr=lr)
        dl_model.train(train_df, feature_cols, target_col=target_col, group_col='date', epochs=epochs, batch_size=batch_size)
            
        mean_ic = np.nan
        # Predict on the last 30 days to compute Rank IC proxy
        if len(dates) > 30:
            val_dates = dates[-30:]
            val_df = ml_df[ml_df['date'].isin(val_dates)].copy()
            val_df['pt_score'] = dl_model.predict(val_df, feature_cols)
            
            # Rank IC calculation
            def calc_ic(g):
                if len(g) > 1:
                    return g['pt_score'].corr(g[target_col], method='spearman')
                return np.nan
                
            ic_series = val_df.groupby('date').apply(calc_ic)
            mean_ic = ic_series.mean()
            log.info(f"[T+{h}] Recent 30-day Mean Rank IC proxy: {mean_ic:.4f}")
            
        os.makedirs('.quantbot_data', exist_ok=True)
        model_path = f'.quantbot_data/prod_pt_model_t{h}.pth'
        meta_path = f'.quantbot_data/prod_pt_meta_t{h}.json'
        
        dl_model.save_model(model_path)
        
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'train_start_date': train_dates[0].strftime('%Y-%m-%d'),
            'train_end_date': train_dates[-1].strftime('%Y-%m-%d'),
            'num_days': len(train_dates),
            'num_samples': len(train_df),
            'features': feature_cols,
            'horizon': h,
            'mean_ic_30d': float(mean_ic) if pd.notna(mean_ic) else None,
            'architecture': 'PyTorchDLModel',
            'epochs': epochs,
            'lr': lr
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        log.info(f"Production model and metadata for T+{h} saved successfully.")

if __name__ == '__main__':
    train_production_model()
