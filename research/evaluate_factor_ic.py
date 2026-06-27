import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
from scipy.stats import spearmanr
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_ic(df: pd.DataFrame, factor_cols: List[str], horizons: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """
    计算给定因子的 IC 和 IR。
    df: 包含因子列和 'code', 'date', 'close' 的 DataFrame
    """
    log = logging.getLogger(__name__)
    
    # Ensure sorting
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    g = df.groupby('code')
    
    # Pre-compute forward returns for all horizons
    ret_cols = []
    for h in horizons:
        col = f'fwd_ret_{h}d'
        df[col] = g['close'].shift(-h) / df['close'] - 1.0
        ret_cols.append(col)
        
    results = []
    
    # Calculate daily IC
    for f in factor_cols:
        if f not in df.columns:
            continue
            
        log.info(f"Evaluating Factor: {f}")
        ic_dict = {'Factor': f}
        
        for h in horizons:
            ret_col = f'fwd_ret_{h}d'
            
            # Drop NAs for this specific pair
            valid_df = df[['date', f, ret_col]].dropna()
            
            if valid_df.empty:
                ic_dict[f'IC_{h}d'] = np.nan
                ic_dict[f'IR_{h}d'] = np.nan
                continue
                
            # Calculate cross-sectional spearman rank correlation daily
            daily_ic = valid_df.groupby('date').apply(
                lambda x: spearmanr(x[f], x[ret_col])[0] if len(x) > 30 else np.nan
            ).dropna()
            
            if daily_ic.empty:
                ic_dict[f'IC_{h}d'] = np.nan
                ic_dict[f'IR_{h}d'] = np.nan
            else:
                mean_ic = daily_ic.mean()
                std_ic = daily_ic.std()
                ir = mean_ic / std_ic if std_ic != 0 else np.nan
                
                ic_dict[f'IC_{h}d'] = mean_ic
                ic_dict[f'IR_{h}d'] = ir
                
        results.append(ic_dict)
        
    res_df = pd.DataFrame(results).set_index('Factor')
    return res_df

if __name__ == "__main__":
    # Example usage
    data_path = '.quantbot_data/ashare_daily.parquet'
    if not os.path.exists(data_path):
        data_path = '../.quantbot_data/ashare_daily.parquet'
        
    if os.path.exists(data_path):
        logging.info("Loading data for IC evaluation...")
        df = pd.read_parquet(data_path)
        
        # We need factors generated first. 
        # This is just a demonstration framework. 
        # In a real workflow, we would load the parquet containing factors.
        # df, factors = calculate_factors(df)
        
        logging.info("Factor IC Evaluation framework is ready.")
    else:
        logging.error("Data lake missing. Cannot test IC.")
