
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
import logging
import os
import itertools

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

def evaluate_weights(df, w1, w5, w10, w20):
    # Cross-sectional percentile ranking per day for T+5, T+10, T+20
    # T+1 is explicitly NOT ranked, maintaining raw scores
    df['rank_t5'] = df.groupby('date')['xgb_score_t5'].rank(pct=True) * 100
    df['rank_t10'] = df.groupby('date')['xgb_score_t10'].rank(pct=True) * 100
    df['rank_t20'] = df.groupby('date')['xgb_score_t20'].rank(pct=True) * 100
    
    # Calculate Ensemble Score using percentile ranks (T+1 uses raw, though w1=0)
    df['ensemble_score'] = (
        df['xgb_score_t1'] * w1 +
        df['rank_t5'] * w5 +
        df['rank_t10'] * w10 +
        df['rank_t20'] * w20
    )
    
    # Group into quintiles per day based on ensemble score
    def _group_pred(group):
        if len(group) < 5: return pd.Series(index=group.index, dtype=float)
        try:
            return pd.qcut(group['ensemble_score'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        except:
            return pd.Series(index=group.index, dtype=float)
            
    df['quantile'] = df.groupby('date').apply(_group_pred).reset_index(0, drop=True)
    
    # Calculate Q5 - Q1 spread
    valid_df = df.dropna(subset=['quantile', 'fwd_ret_real'])
    if valid_df.empty: return 0
    
    group_returns = valid_df.groupby('quantile')['fwd_ret_real'].mean() * 10000 # in bps
    spread = group_returns.get(5, 0) - group_returns.get(1, 0)
    return spread

def main():
    oos_path = '.quantbot_data/oos_preds.csv'
    if not os.path.exists(oos_path):
        log.error(f"Cannot find {oos_path}. Please run agent_auto_eval.py first.")
        return

    log.info("Loading OOS Predictions...")
    df = pd.read_csv(oos_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Clean data
    df = df.dropna(subset=['xgb_score_t1', 'xgb_score_t5', 'xgb_score_t10', 'xgb_score_t20', 'fwd_ret_real']).copy()
    log.info(f"Loaded {len(df)} valid OOS rows.")
    
    log.info("Starting Grid Search for Optimal Ensemble Weights...")
    
    # Grid search from 0.0 to 1.0 with step 0.1
    steps = [x / 10.0 for x in range(11)]
    best_spread = -9999
    best_weights = (0.25, 0.25, 0.25, 0.25)
    
    combinations = 0
    for w5, w10, w20 in itertools.product(steps, repeat=3):
        w1 = 0.0  # 强制剥离 T+1 权重
        if abs(w1 + w5 + w10 + w20 - 1.0) > 1e-5:
            continue
            
        combinations += 1
        spread = evaluate_weights(df, w1, w5, w10, w20)
        
        if spread > best_spread:
            best_spread = spread
            best_weights = (w1, w5, w10, w20)
            log.info(f"New Best! w1={w1:.1f}, w5={w5:.1f}, w10={w10:.1f}, w20={w20:.1f} | Spread: {spread:.2f} bps/day")

    log.info(f"Grid Search Finished ({combinations} combinations).")
    log.info(f"🏆 Final Best Weights -> T+1: {best_weights[0]}, T+5: {best_weights[1]}, T+10: {best_weights[2]}, T+20: {best_weights[3]}")
    log.info(f"🏆 Maximum Long-Short Spread: {best_spread:.2f} bps/day")

    # Save weights
    weights_dict = {
        'T+1': best_weights[0],
        'T+5': best_weights[1],
        'T+10': best_weights[2],
        'T+20': best_weights[3],
        'wfo_spread_bps': best_spread
    }
    
    out_path = '.quantbot_data/ensemble_weights.json'
    with open(out_path, 'w') as f:
        json.dump(weights_dict, f, indent=4)
    log.info(f"Optimal weights saved to {out_path}")

if __name__ == '__main__':
    main()
