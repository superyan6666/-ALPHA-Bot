import os
import json
import logging
import numpy as np
import pandas as pd
import akshare as ak
import torch

import ml_engine
import feature_engine

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_csi300():
    try:
        df_index = ak.index_zh_a_hist(symbol="000300", period="daily", start_date="20231201", end_date="20240630")
        df_index = df_index[['日期', '收盘']].rename(columns={'日期': 'date', '收盘': 'close'})
        df_index['date'] = pd.to_datetime(df_index['date'])
        df_index = df_index.set_index('date').sort_index()
    except Exception as e:
        logging.warning(f"Failed to fetch CSI300: {e}. Trying alternative...")
        try:
            df_index = ak.stock_zh_index_daily(symbol="sh000300")
            df_index = df_index.reset_index()
            if 'date' in df_index.columns:
                date_col = 'date'
            else:
                date_col = 'index'
            df_index = df_index[[date_col, 'close']].rename(columns={date_col: 'date', 'close': 'close'})
            df_index['date'] = pd.to_datetime(df_index['date'])
            df_index = df_index[(df_index['date'] >= '2023-12-01') & (df_index['date'] <= '2024-06-30')]
            df_index = df_index.set_index('date').sort_index()
        except Exception as e2:
            logging.warning(f"Alternative also failed: {e2}. Using cross-sectional mean as fallback.")
            return None
    return df_index

def main():
    logging.info("1. Loading data")
    df = pd.read_parquet('.quantbot_data/ashare_daily.parquet')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date'])
    
    logging.info("Calculating forward returns and max drawdown...")
    df['close_t20'] = df.groupby('code')['close'].shift(-20)
    df['actual_ret_t20'] = (df['close_t20'] - df['close']) / df['close']
    
    # Calculate Max Drawdown
    low_rolling = df.groupby('code')['low'].rolling(20, min_periods=1).min().reset_index(0, drop=True)
    df['low_rolling'] = low_rolling
    df['lowest_in_future_20d'] = df.groupby('code')['low_rolling'].shift(-20)
    df['max_drawdown_t20'] = (df['lowest_in_future_20d'] - df['close']) / df['close']
    
    logging.info("Fetching CSI300 Benchmark...")
    csi300 = get_csi300()
    if csi300 is not None and not csi300.empty:
        csi300['mkt_close_t20'] = csi300['close'].shift(-20)
        csi300['mkt_ret_t20'] = (csi300['mkt_close_t20'] - csi300['close']) / csi300['close']
        df = pd.merge(df, csi300[['mkt_ret_t20']], left_on='date', right_index=True, how='left')
    else:
        df['mkt_ret_t20'] = df.groupby('date')['actual_ret_t20'].transform('mean')
        
    if 'volume' in df.columns:
        df = df.rename(columns={'volume': 'vol'})
        
    logging.info("Building features (this might take a minute)...")
    df = feature_engine.build_ml_features(df)
    
    with open('.quantbot_data/prod_pt_meta_t20.json', 'r') as f:
        meta = json.load(f)
    features = meta['features']
    
    logging.info("Loading PyTorchDLModel...")
    input_dim = len(features)
    model = ml_engine.PyTorchDLModel(input_dim)
    model.load_model('.quantbot_data/prod_pt_model_t20.pth')
    
    # We need predictions from Jan to May 2024
    df_eval = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2024-05-31')].copy()
    
    logging.info("Predicting base scores...")
    df_eval['base_score'] = model.predict(df_eval, features)
    df_eval = df_eval.dropna(subset=['base_score'])
    
    logging.info("Selecting Top 20 candidates per day...")
    df_eval['rank'] = df_eval.groupby('date')['base_score'].rank(ascending=False, method='first')
    top20 = df_eval[df_eval['rank'] <= 20].copy()
    
    # We only have complete forward return data up to around May 10th if dataset ends on June 9th.
    # To get meaningful evaluation, we drop NA targets.
    top20 = top20.dropna(subset=['actual_ret_t20', 'mkt_ret_t20', 'max_drawdown_t20'])
    
    logging.info("Creating Target Labels...")
    top20['Target_A'] = np.where(top20['actual_ret_t20'] > 0.05, 2, np.where(top20['actual_ret_t20'] > 0, 1, 0))
    top20['Target_B'] = np.where((top20['actual_ret_t20'] - top20['mkt_ret_t20']) > 0.05, 2, np.where((top20['actual_ret_t20'] - top20['mkt_ret_t20']) > 0, 1, 0))
    
    safe_mdd = np.abs(top20['max_drawdown_t20'].replace(0, -0.0001))
    ratio = top20['actual_ret_t20'] / safe_mdd
    top20['Target_C'] = np.where(ratio > 2.0, 2, np.where(ratio > 0.5, 1, 0))
    
    # Ensure there are no NaNs in features
    top20 = top20.dropna(subset=features)
    
    train_mask = top20['date'] < '2024-05-01'
    test_mask = (top20['date'] >= '2024-05-01') & (top20['date'] < '2026-05-01')
    df_train = top20[train_mask].copy()
    df_test = top20[test_mask].copy()
    
    logging.info(f"Train samples (Jan-Apr): {len(df_train)}, Test samples (May): {len(df_test)}")
    
    if len(df_test) == 0:
        logging.error("No test samples available! (Data cutoff might not cover T+20 for May)")
        return
        
    results_text = []
    results_text.append("# Meta-Critic Ablation Study Results\n")
    results_text.append("## Target Definitions")
    results_text.append("- **Target A**: Absolute Return (2: >5%, 1: >0, 0: else)")
    results_text.append("- **Target B**: Excess Return vs CSI300 (2: >5%, 1: >0, 0: else)")
    results_text.append("- **Target C**: Return/MaxDD Ratio (2: >2.0, 1: >0.5, 0: else)\n")
    
    df['mkt_ret_t20'] = df.groupby('date')['actual_ret_t20'].transform(lambda x: x.mean(skipna=True))
    
    # ... inside main() we need to fix the mkt_ret_t20 calc which was around line 61
    # But since we are replacing lines 122-163, let's just do it here:
    
    from sklearn.ensemble import RandomForestClassifier
    
    for target in ['Target_A', 'Target_B', 'Target_C']:
        logging.info(f"Training Meta-Critic for {target}...")
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        
        X_train = df_train[features].fillna(0)
        y_train = df_train[target]
        clf.fit(X_train, y_train)
        
        X_test = df_test[features].fillna(0)
        probs = clf.predict_proba(X_test)
        
        # In sklearn, predict_proba returns columns in order of sorted classes (0, 1, 2)
        # We need to make sure class 0 is actually at index 0.
        classes = list(clf.classes_)
        if 0 in classes:
            idx_0 = classes.index(0)
            prob_0 = probs[:, idx_0]
        else:
            prob_0 = np.zeros(len(X_test))
            
        df_test_clf = df_test.copy()
        # threshold 0.4 for veto
        df_test_clf['veto'] = prob_0 > 0.4
        surviving = df_test_clf[~df_test_clf['veto']]
        
        n_surviving = len(surviving)
        total_test = len(df_test_clf)
        
        if n_surviving > 0:
            avg_ret = surviving['actual_ret_t20'].mean() * 100
            avg_exc_ret = (surviving['actual_ret_t20'] - surviving['mkt_ret_t20']).mean() * 100
            avg_mdd = surviving['max_drawdown_t20'].mean() * 100
        else:
            avg_ret = avg_exc_ret = avg_mdd = np.nan
            
        res_line = (f"### Model {target}\n"
                    f"- Surviving / Total: {n_surviving} / {total_test} ({n_surviving/total_test*100:.1f}% retention)\n"
                    f"- Average Actual Return: {avg_ret:.2f}%\n"
                    f"- Average Excess Return: {avg_exc_ret:.2f}%\n"
                    f"- Average Max Drawdown: {avg_mdd:.2f}%\n")
        logging.info(res_line)
        results_text.append(res_line)
        
    # Baseline
    avg_ret_base = df_test['actual_ret_t20'].mean() * 100
    avg_exc_ret_base = (df_test['actual_ret_t20'] - df_test['mkt_ret_t20']).mean() * 100
    avg_mdd_base = df_test['max_drawdown_t20'].mean() * 100
    
    res_base = (f"### Baseline (No Veto)\n"
                f"- Total: {len(df_test)}\n"
                f"- Average Actual Return: {avg_ret_base:.2f}%\n"
                f"- Average Excess Return: {avg_exc_ret_base:.2f}%\n"
                f"- Average Max Drawdown: {avg_mdd_base:.2f}%\n")
    logging.info(res_base)
    results_text.append(res_base)
    
    with open('ablation_results_meta.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(results_text))
    logging.info("Done! Results saved to ablation_results_meta.md")

if __name__ == '__main__':
    main()
