
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

from ml_engine import PyTorchClassifier
from signal_tracker import SignalTracker
from feature_engine import build_ml_features

def main():
    log.info("Starting Feedback Meta-Critic Model training...")
    
    # Load feedback data
    tracker = SignalTracker()
    feedback_df = tracker.get_training_feedback(min_age_days=0)
    
    if feedback_df.empty:
        log.warning("No complete signals found in tracker. Exiting.")
        return
        
    # Map target classes using Excess Return (Alpha / Target B)
    # 2: excess_ret_t20 > 5% or hit_target
    # 1: excess_ret_t20 > 0
    # 0: else
    log.info("Mapping target classes using Excess Return (Target B)...")
    feedback_df['target_class'] = 0
    
    if 'excess_ret_t20' in feedback_df.columns:
        excess = feedback_df['excess_ret_t20'].fillna(-1.0)
        ret_mask = excess > 0
        super_mask = excess > 0.05
    else:
        # Fallback if excess not calculated
        ret_mask = feedback_df.get('actual_ret_t20', pd.Series(0, index=feedback_df.index)) > 0
        super_mask = feedback_df.get('actual_ret_t20', pd.Series(0, index=feedback_df.index)) > 0.05
        
    hit_mask = feedback_df.get('hit_target', pd.Series(False, index=feedback_df.index)) == True
    
    feedback_df.loc[ret_mask, 'target_class'] = 1
    feedback_df.loc[hit_mask | super_mask, 'target_class'] = 2
    
    log.info(f"Target class distribution:\n{feedback_df['target_class'].value_counts()}")
    
    # Load daily data
    parquet_path = '.quantbot_data/ashare_daily.parquet'
    if not os.path.exists(parquet_path):
        log.error(f"Cannot find {parquet_path}. Exiting.")
        return
        
    log.info("Loading ashare_daily.parquet...")
    panel = pd.read_parquet(parquet_path)
    
    # Build features
    log.info("Building ML features...")
    panel = build_ml_features(panel)
    
    # Ensure signal_date format matches date format for merging
    feedback_df['signal_date'] = pd.to_datetime(feedback_df['signal_date']).dt.normalize()
    if not np.issubdtype(panel['date'].dtype, np.datetime64):
        panel['date'] = pd.to_datetime(panel['date']).dt.normalize()
        
    # Merge features with feedback
    log.info("Merging features with feedback signals...")
    merged_df = pd.merge(
        feedback_df, 
        panel, 
        left_on=['code', 'signal_date'], 
        right_on=['code', 'date'], 
        how='inner'
    )
    
    if len(merged_df) < 10:
        log.warning(f"Only {len(merged_df)} rows after merge (need at least 10). Wait for more data. Exiting gracefully.")
        return
        
    log.info(f"Training on {len(merged_df)} merged samples.")
    
    # Load feature columns for T+20
    features_json = '.quantbot_data/horizon_features.json'
    if os.path.exists(features_json):
        with open(features_json, 'r') as f:
            all_features = json.load(f)
            feature_cols = all_features.get('T+20', [])
    else:
        # Fallback to all F_ columns if JSON not found
        feature_cols = [c for c in panel.columns if c.startswith('F_')]
        
    # Make sure we only use columns that exist in merged_df
    feature_cols = [c for c in feature_cols if c in merged_df.columns]
    
    # Train
    model = PyTorchClassifier(input_dim=len(feature_cols), num_classes=3)
    model.train(merged_df, feature_cols=feature_cols, target_col='target_class', epochs=10, batch_size=256)
    
    # Save model
    out_pth = '.quantbot_data/prod_meta_critic.pth'
    out_meta = '.quantbot_data/prod_meta_critic_meta.json'
    
    model.save_model(out_pth)
    with open(out_meta, 'w') as f:
        json.dump({'feature_cols': feature_cols, 'num_classes': 3}, f, indent=4)
        
    log.info(f"Feedback Meta-Critic trained and saved to {out_pth}")

if __name__ == '__main__':
    main()
