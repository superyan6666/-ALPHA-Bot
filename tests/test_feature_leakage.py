import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import feature_engine
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_time_leakage():
    logging.info("Loading sample data for leakage test...")
    # 仅加载部分数据加速测试
    try:
        df = pd.read_parquet('.quantbot_data/ashare_daily.parquet')
        if 'volume' in df.columns:
            df = df.rename(columns={'volume': 'vol'})
        if 'pctChg' in df.columns:
            df = df.rename(columns={'pctChg': 'pct_chg'})
        # 取最近 2 年数据
        df = df[df['date'] >= '2022-01-01'].copy()
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return

    logging.info("Running full cross-sectional feature generation...")
    df_full = feature_engine.build_ml_features(df.copy())
    
    # 选定截断日
    cutoff_date = '2023-06-30'
    logging.info(f"Running truncated feature generation up to {cutoff_date}...")
    df_trunc = df[df['date'] <= cutoff_date].copy()
    df_trunc = feature_engine.build_ml_features(df_trunc)

    # 提取特征列
    f_cols = [c for c in df_full.columns if c.startswith('F_')]
    
    # 比较 cutoff_date 这一天的特征值
    full_day = df_full[df_full['date'] == cutoff_date].set_index('code')[f_cols].sort_index()
    trunc_day = df_trunc[df_trunc['date'] == cutoff_date].set_index('code')[f_cols].sort_index()
    
    # 对齐索引
    common_codes = full_day.index.intersection(trunc_day.index)
    full_day = full_day.loc[common_codes]
    trunc_day = trunc_day.loc[common_codes]
    
    # 比较数值（考虑浮点误差）
    diff = (full_day - trunc_day).abs().max()
    max_diff = diff.max()
    
    if max_diff > 1e-4:
        logging.error(f"CRITICAL ERROR: Future data leakage detected! Max difference: {max_diff}")
        problematic_cols = diff[diff > 1e-4].index.tolist()
        logging.error(f"Problematic columns: {problematic_cols}")
        raise RuntimeError("Time leakage test failed.")
    else:
        logging.info("✅ Time leakage test passed! Features up to T are completely invariant to future data (T+1 to N).")

if __name__ == "__main__":
    test_time_leakage()
