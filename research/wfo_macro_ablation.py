
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import DataProxy, C
from ml_engine import PyTorchDLModel, apply_liquidity_gate
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class MacroAblationEvaluator:
    def __init__(self):
        pass
        
    def load_data(self):
        """Fetch panel data from local parquet data lake and merge macro."""
        parquet_path = ".quantbot_data/ashare_daily.parquet"
        macro_path = ".quantbot_data/macro_daily.parquet"
        
        if not os.path.exists(parquet_path) or not os.path.exists(macro_path):
            log.error("Data lakes not found. Build them first.")
            return None
            
        log.info("Loading A-share data...")
        panel = pd.read_parquet(parquet_path)
        panel = panel.rename(columns={'date': 'date', 'code': 'code', 'open': 'open', 'close': 'close', 'high': 'high', 'low': 'low', 'volume': 'vol'})
        panel['date'] = pd.to_datetime(panel['date'])
        
        # We start from 2022 to give enough time for WFO train (500)
        start_dt = pd.to_datetime("2020-01-01") 
        panel = panel[panel['date'] >= start_dt].copy()
        panel['code'] = panel['code'].str.replace('sh.', '', regex=False).str.replace('sz.', '', regex=False).str.replace('bj.', '', regex=False)
        
        log.info("Preprocessing...")
        for col in ['open', 'high', 'low', 'close', 'vol']:
            panel[col] = pd.to_numeric(panel[col], errors='coerce')
        
        panel['prev_close'] = panel.groupby('code')['close'].shift(1)
        panel['pct_chg'] = (panel['close'] / panel['prev_close'] - 1) * 100
        panel['is_limit'] = (panel['pct_chg'].abs() >= 9.5) & (panel['high'] == panel['low'])
        
        panel['next_open'] = panel.groupby('code')['open'].shift(-1)
        panel['next_close'] = panel.groupby('code')['close'].shift(-1)
        panel['fwd_ret_real'] = panel['next_close'] / (panel['next_open'] + 1e-5) - 1
        panel = panel.dropna(subset=['fwd_ret_real'])
        
        # Basic Factors (retired sm_corr and vol_ratio)
        log.info("Computing Basic Factors...")
        amihud_raw = panel['pct_chg'].abs() / (panel['vol'] * panel['close'] + 1e-5) * 1e6
        panel['amihud'] = np.where(panel.get('is_limit', False), 99999.0, amihud_raw)
        panel['amihud_20'] = panel.groupby('code')['amihud'].transform(lambda x: x.rolling(20).mean())
        
        panel['clv'] = (panel['close'] - panel['low']) / (panel['high'] - panel['low'] + 1e-8)
        panel['volatility_5d'] = panel.groupby('code')['pct_chg'].transform(lambda x: x.rolling(5).std())
        panel['alpha_reversal_5d'] = - (panel['close'] / panel.groupby('code')['close'].shift(5) - 1)
        panel['alpha_024_approx'] = panel.groupby('code')['close'].transform(lambda x: x.rolling(20).mean()) / (panel['close'] + 1e-5) - 1
        
        # Market Regime
        market_daily = panel.groupby('date')['pct_chg'].mean().reset_index()
        market_daily.rename(columns={'pct_chg': 'market_ret'}, inplace=True)
        market_daily['market_ret_20d'] = market_daily['market_ret'].rolling(20, min_periods=5).mean()
        market_daily['market_ret_60d'] = market_daily['market_ret'].rolling(60, min_periods=20).mean()
        market_daily['market_vol_20d'] = market_daily['market_ret'].rolling(20, min_periods=5).std()
        
        panel = pd.merge(panel, market_daily[['date', 'market_ret_20d', 'market_ret_60d', 'market_vol_20d']], on='date', how='left')
        
        # Load and merge macro
        log.info("Merging Macro Data...")
        macro_df = pd.read_parquet(macro_path)
        macro_df.index = pd.to_datetime(macro_df.index)
        
        panel = pd.merge(panel, macro_df, left_on='date', right_index=True, how='left')
        return panel

    def run_wfo(self, ml_df, feature_cols, config_name):
        log.info(f"========== Running WFO for {config_name} ==========")
        dates = sorted(ml_df['date'].unique())
        train_window = 500
        step = 125
        
        all_test_preds = []
        feature_importances = []
        
        for idx in range(train_window, len(dates), step):
            train_dates = dates[max(0, idx - train_window):idx]
            test_dates = dates[idx:min(len(dates), idx + step)]
            if len(test_dates) == 0: break
                
            train_df = ml_df[ml_df['date'].isin(train_dates)].copy()
            test_df = ml_df[ml_df['date'].isin(test_dates)].copy()
                
            model = PyTorchDLModel(input_dim=len(feature_cols))
            model.train(train_df, feature_cols, target_col='fwd_ret_real', group_col='date')
            
            # Since PyTorchDLModel doesn't have get_feature_importance naturally, we omit or mock it
            # We will just use an empty series for importance to not break the pipeline
            feature_importances.append(pd.Series(0, index=feature_cols, name='importance'))
            
            test_df['xgb_score'] = model.predict(test_df, feature_cols)
            all_test_preds.append(test_df[['date', 'code', 'xgb_score', 'fwd_ret_real']])

            
        if not all_test_preds:
            log.error("No WFO predictions generated.")
            return None, None
            
        oos_df = pd.concat(all_test_preds, ignore_index=True)
        agg_imp = pd.concat(feature_importances, axis=1).mean(axis=1).sort_values(ascending=False)
        
        def _group_pred(group):
            if len(group) < 5: return pd.Series(index=group.index, dtype=float)
            try:
                return pd.qcut(group['xgb_score'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            except:
                return pd.Series(index=group.index, dtype=float)
                
        oos_df['xgb_quantile'] = oos_df.groupby('date').apply(_group_pred).reset_index(0, drop=True)
        oos_df = oos_df.dropna(subset=['xgb_quantile'])
        group_returns = oos_df.groupby('xgb_quantile')['fwd_ret_real'].mean() * 10000 
        ls_ret = group_returns.get(5, 0) - group_returns.get(1, 0)
        
        log.info(f"[{config_name}] WFO OOS Spread (Q5-Q1): {ls_ret:.2f} bps/day")
        
        # [CRUCIBLE PROTOCOL] Calculate Sharpe Ratio to enforce DSR validation
        daily_returns = oos_df.groupby('date').apply(
            lambda x: x[x['xgb_quantile'] == 5]['fwd_ret_real'].mean() - x[x['xgb_quantile'] == 1]['fwd_ret_real'].mean()
        ).fillna(0)
        
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0
            
        log.info(f"[{config_name}] WFO OOS Sharpe Ratio: {sharpe:.2f}")
        
        # Log top 3 features (currently mocked as 0 for PyTorch)
        log.info(f"[{config_name}] Top 3 Features: {agg_imp.index.tolist()[:3]}")
        return ls_ret, agg_imp, sharpe

if __name__ == "__main__":
    evaluator = MacroAblationEvaluator()
    panel = evaluator.load_data()
    
    if panel is not None:
        panel = apply_liquidity_gate(panel, amihud_col='amihud_20', threshold_pct=0.90)
        panel = panel[~panel['is_limit']].copy()
        
        # Drop naive NaN rows
        panel = panel.dropna(subset=['fwd_ret_real', 'date'])
        
        base_features = ['clv', 'volatility_5d', 'alpha_reversal_5d', 'alpha_024_approx', 'market_ret_20d', 'market_ret_60d', 'market_vol_20d', 'macro_staleness_days']
        
        configs = {
            "Fold 0: Baseline": base_features,
            "Fold 1: Baseline + Spread": base_features + ['us_cn_spread'],
            "Fold 2: Baseline + US Curve": base_features + ['us_yield_curve_spread', 'us_yield_curve_inversion'],
            "Fold 3: Baseline + CN Trend": base_features + ['cn_10y_trend'],
            "Fold 4: Baseline + All Macro": base_features + ['us_cn_spread', 'us_yield_curve_spread', 'us_yield_curve_inversion', 'cn_10y_trend']
        }
        
        results = {}
        for config_name, f_cols in configs.items():
            ml_df = panel.dropna(subset=f_cols).copy()
            if len(ml_df) < 10000:
                log.warning(f"Not enough data for {config_name}. Skipping.")
                continue
            ls_ret, agg_imp, sharpe = evaluator.run_wfo(ml_df, f_cols, config_name)
            results[config_name] = {'ls_ret': ls_ret, 'sharpe': sharpe}
            
        log.info("========== ABLATION RESULTS SUMMARY ==========")
        for config_name, res in results.items():
            log.info(f"{config_name}: Spread {res['ls_ret']:.2f} bps/day | Sharpe {res['sharpe']:.2f}")
