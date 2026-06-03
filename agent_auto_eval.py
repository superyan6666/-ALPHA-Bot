import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Append current directory to import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import DataProxy, C
from ml_engine import XGBoostLTR, apply_liquidity_gate
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class SandboxEvaluator:
    def __init__(self, start_date="20230101", end_date=None):
        self.dp = DataProxy()
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y%m%d")
        
    def fetch_data(self, stock_list=None):
        """Fetch panel data from local parquet data lake or fallback to network."""
        parquet_path = ".quantbot_data/ashare_daily.parquet"
        if os.path.exists(parquet_path):
            log.info(f"Found offline data lake at {parquet_path}. Loading into memory (this may take a few seconds)...")
            try:
                panel = pd.read_parquet(parquet_path)
                
                # Normalize column names
                panel = panel.rename(columns={'date': 'date', 'code': 'code', 'open': 'open', 'close': 'close', 'high': 'high', 'low': 'low', 'volume': 'vol'})
                
                # Filter by date
                start_dt = pd.to_datetime(self.start_date)
                end_dt = pd.to_datetime(self.end_date)
                panel['date'] = pd.to_datetime(panel['date'])
                panel = panel[(panel['date'] >= start_dt) & (panel['date'] <= end_dt)].copy()
                
                # Normalize code format from sh.600000 to 600000 if needed
                panel['code'] = panel['code'].str.replace('sh.', '', regex=False).str.replace('sz.', '', regex=False).str.replace('bj.', '', regex=False)
                
                if stock_list and stock_list != "ALL":
                    panel = panel[panel['code'].isin(stock_list)]
                    
                panel = panel.sort_values(['code', 'date']).reset_index(drop=True)
                log.info(f"Loaded {len(panel)} rows from data lake.")
                return panel
            except Exception as e:
                log.error(f"Failed to load from parquet: {e}. Falling back to network fetch.")
                
        # Fallback to network fetching
        if stock_list is None or stock_list == "ALL":
            log.error("Network fetching for ALL stocks is blocked to prevent IP ban. Please build the data lake first.")
            return None
            
        log.info(f"Fetching data for {len(stock_list)} stocks from {self.start_date} to {self.end_date} via network...")
        all_dfs = []
        for code in stock_list:
            df = self.dp.get_hist(code, self.start_date, self.end_date)
            if df is not None and not df.empty and len(df) > 60:
                df = df.copy()
                if df.index.name == 'date' or type(df.index) == pd.DatetimeIndex:
                    df = df.reset_index()
                # Ensure columns are english
                df = df.rename(columns={
                    C.H_DATE: 'date',
                    C.H_OPEN: 'open',
                    C.H_CLOSE: 'close',
                    C.H_HIGH: 'high',
                    C.H_LOW: 'low',
                    C.H_VOL: 'vol'
                })
                df['code'] = code
                all_dfs.append(df)
        
        if not all_dfs:
            log.error("No data fetched.")
            return None
        
        panel = pd.concat(all_dfs, ignore_index=True)
        # Sort by stock and date
        panel = panel.sort_values(['code', 'date']).reset_index(drop=True)
        return panel

    def preprocess(self, panel):
        """Clean data, filter limit up/down, compute forward returns."""
        # Clean up NaNs and numeric
        for col in ['open', 'high', 'low', 'close', 'vol']:
            panel[col] = pd.to_numeric(panel[col], errors='coerce')
        
        panel['prev_close'] = panel.groupby('code')['close'].shift(1)
        panel['pct_chg'] = (panel['close'] / panel['prev_close'] - 1) * 100
        
        # Mark limit up/down (approximate 9.5% for A-shares)
        panel['is_limit'] = (panel['pct_chg'].abs() >= 9.5) & (panel['high'] == panel['low'])
        
        # Calculate realistic forward returns (T+1 to T+2)
        # Buy at T+1 Open, Sell at T+2 Open (Or T+1 Close to T+2 Close)
        # Wait, if we generate signals at T Close, we execute at T+1 Open.
        # realistic return: (Open(T+2) / Open(T+1)) - 1
        # Let's use Close(T+1) / Open(T+1) - 1 to represent intraday return on T+1
        panel['next_open'] = panel.groupby('code')['open'].shift(-1)
        panel['next_close'] = panel.groupby('code')['close'].shift(-1)
        panel['fwd_ret_1d'] = panel['next_close'] / panel['close'] - 1  # For linear eval compatibility
        panel['fwd_ret_real'] = panel['next_close'] / (panel['next_open'] + 1e-5) - 1 # Realizable friction-aware return
        
        # Build fwd_ret for all horizons
        panel['fwd_ret_t1'] = panel.groupby('code')['close'].shift(-1) / (panel.groupby('code')['open'].shift(-1) + 1e-5) - 1
        panel['fwd_ret_t5'] = panel.groupby('code')['close'].shift(-5) / (panel.groupby('code')['open'].shift(-1) + 1e-5) - 1
        panel['fwd_ret_t10'] = panel.groupby('code')['close'].shift(-10) / (panel.groupby('code')['open'].shift(-1) + 1e-5) - 1
        panel['fwd_ret_t20'] = panel.groupby('code')['close'].shift(-20) / (panel.groupby('code')['open'].shift(-1) + 1e-5) - 1
        panel['fwd_ret_t40'] = panel.groupby('code')['close'].shift(-40) / (panel.groupby('code')['open'].shift(-1) + 1e-5) - 1
        panel['fwd_ret_t60'] = panel.groupby('code')['close'].shift(-60) / (panel.groupby('code')['open'].shift(-1) + 1e-5) - 1
        panel['fwd_ret_t120'] = panel.groupby('code')['close'].shift(-120) / (panel.groupby('code')['open'].shift(-1) + 1e-5) - 1
        
        panel = panel.dropna(subset=['fwd_ret_1d', 'fwd_ret_real'])
        return panel

    def calc_factors(self, panel):
        """Calculate A-share specific trial factors."""
        log.info("Calculating factors...")
        
        # Calculate percentage change if not already present
        if 'pct_chg' not in panel.columns:
            panel['prev_close'] = panel.groupby('code')['close'].shift(1)
            panel['pct_chg'] = (panel['close'] / panel['prev_close'] - 1) * 100
            
        from feature_engine import build_ml_features
        log.info("Building features via feature_engine.py to maintain DRY...")
        panel = build_ml_features(panel)
        return panel
        
    def evaluate_ml_model(self, panel):
        """Train and evaluate XGBoost LTR model."""
        log.info("--- Starting Machine Learning Evaluation ---")
        
        # Apply Liquidity Gate
        amihud_cols = [c for c in panel.columns if c.startswith('F_amihud_')]
        amihud_col = amihud_cols[0] if amihud_cols else 'F_amihud_20'
        panel = apply_liquidity_gate(panel, amihud_col=amihud_col, threshold_pct=0.90)
        
        # Filter limits
        panel = panel[~panel['is_limit']].copy()
        
        # Load horizon specific features
        import json
        horizon_features_path = '.quantbot_data/horizon_features.json'
        if not os.path.exists(horizon_features_path):
            log.error("horizon_features.json not found! Run select_features.py first.")
            return
        with open(horizon_features_path) as f:
            horizon_features = json.load(f)
        
        train_window = 500
        step = 125
        
        horizons = [10, 20]
        all_oos_dfs = []
        
        log.info(f"Starting Walk-Forward Optimization for {len(horizons)} horizons...")
        
        for h in horizons:
            horizon_key = f"T+{h}"
            feature_cols = horizon_features.get(horizon_key, [])
            if not feature_cols:
                log.error(f"No features found for {horizon_key}")
                continue
                
            log.info(f"\n{'='*40}\nRunning WFO for Horizon T+{h} with {len(feature_cols)} features\n{'='*40}")
            target_col = f'fwd_ret_t{h}'
            
            # Delayed Dropna specific to this horizon
            ml_df = panel.dropna(subset=feature_cols + ['date', target_col]).copy()
            
            dates = sorted(ml_df['date'].unique())
            if len(dates) < 500:
                log.warning(f"Not enough dates for WFO on T+{h} (need > 500), got {len(dates)}.")
                continue
            all_test_preds = []
            
            for idx in range(train_window, len(dates), step):
                train_dates = dates[max(0, idx - train_window):idx]
                test_dates = dates[idx:min(len(dates), idx + step)]
                if len(test_dates) == 0: break
                
                train_df = ml_df[ml_df['date'].isin(train_dates)].copy()
                test_df = ml_df[ml_df['date'].isin(test_dates)].copy()
                
                from ml_engine import PyTorchDLModel
                model = PyTorchDLModel(input_dim=len(feature_cols))
                # WFO is slow, use fewer epochs (e.g., 3) for evaluation
                model.train(train_df, feature_cols, target_col=target_col, group_col='date', epochs=3)
                
                preds = model.predict(test_df, feature_cols)
                test_df[f'xgb_score_t{h}'] = preds
                all_test_preds.append(test_df[['date', 'code', f'xgb_score_t{h}', 'fwd_ret_real', 'fwd_ret_1d', target_col]])
                
                # Permutation Importance (Scorer / 打分器) on the last fold test set to see what drives the model
                if idx + step >= len(dates): 
                    log.info(f"--- Permutation Importance (打分器) for Horizon T+{h} ---")
                    import torch
                    from sklearn.metrics import mean_squared_error
                    baseline_mse = mean_squared_error(test_df[target_col].fillna(0), preds)
                    importances = []
                    test_vals = test_df[feature_cols].fillna(0).values
                    for i, col in enumerate(feature_cols):
                        # Permute the i-th column
                        perm_vals = test_vals.copy()
                        np.random.shuffle(perm_vals[:, i])
                        
                        model.model.eval()
                        with torch.no_grad():
                            perm_X = torch.tensor(perm_vals, dtype=torch.float32).to(model.device)
                            perm_preds = model.model(perm_X).cpu().numpy().flatten()
                            
                        perm_mse = mean_squared_error(test_df[target_col].fillna(0), perm_preds)
                        importance = perm_mse - baseline_mse
                        importances.append((col, importance))
                        
                    importances.sort(key=lambda x: x[1], reverse=True)
                    for col, imp in importances[:10]: # Print top 10
                        log.info(f"  {col}: {imp:.6f}")
                        
            oos_h = pd.concat(all_test_preds, ignore_index=True)
            all_oos_dfs.append(oos_h)
            
        # Merge all horizon OOS predictions
        oos_df = all_oos_dfs[0]
        for i in range(1, len(all_oos_dfs)):
            oos_df = oos_df.merge(all_oos_dfs[i][['date', 'code', f'xgb_score_t{horizons[i]}', f'fwd_ret_t{horizons[i]}']], on=['date', 'code'], how='left')
            
        # Save OOS predictions for weight optimization
        os.makedirs('.quantbot_data', exist_ok=True)
        oos_df.to_csv('.quantbot_data/oos_preds.csv', index=False)
        log.info("Saved OOS predictions to .quantbot_data/oos_preds.csv")
        
        log.info(f"WFO Completed. Total OOS predictions: {len(oos_df)}")
        
        # Group by prediction deciles across the concatenated OOS dataframe
        def _group_pred_col(group, score_col):
            if len(group) < 5: return pd.Series(index=group.index, dtype=float)
            try:
                return pd.qcut(group[score_col], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            except:
                return pd.Series(index=group.index, dtype=float)

        best_ls_spread_bps = -9999
        for h in horizons:
            score_col = f'xgb_score_t{h}'
            target_col = f'fwd_ret_t{h}'
            
            if score_col not in oos_df.columns or target_col not in oos_df.columns:
                continue
                
            oos_df[f'quantile_t{h}'] = oos_df.groupby('date').apply(lambda g: _group_pred_col(g, score_col)).reset_index(0, drop=True)
            
            # Evaluate using the proper horizon target
            group_returns = oos_df.groupby(f'quantile_t{h}')[target_col].mean() * 10000 # bps
            
            log.info(f"\n--- WFO Out-of-Sample Mean Return by Quantile for Horizon T+{h} (bps) ---")
            for q in [1, 2, 3, 4, 5]:
                ret = group_returns.get(q, 0)
                log.info(f"  Q{q}: {ret:.2f} bps")
                
            ls_ret = group_returns.get(5, 0) - group_returns.get(1, 0)
            
            # Calculate annualized spread
            # Since the return is over H days, we divide by H to get daily average spread
            daily_ls_ret = ls_ret / h
            ann_ls_ret = daily_ls_ret * 252
            
            log.info(f"  Long-Short Spread (Q5-Q1): {ls_ret:.2f} bps per holding period")
            log.info(f"  Daily Avg L-S Spread: {daily_ls_ret:.2f} bps/day")
            log.info(f"  Annualized L-S Spread: {ann_ls_ret / 100:.2f}%")
            
            if h == 20: # Use T+20 as the anchor metric because short-term models are noisy
                best_ls_spread_bps = daily_ls_ret
        
        # Save metrics for Auto-Gate
        metrics = {
            "wfo_ls_spread_bps": float(best_ls_spread_bps),
            "timestamp": datetime.now().isoformat()
        }
        with open('.quantbot_data/eval_metrics.json', 'w') as f:
            import json
            json.dump(metrics, f)
            
    def evaluate(self, panel, factor_name):
        """Evaluate a specific factor."""
        log.info(f"--- Evaluating Factor: {factor_name} ---")
        
        # Filter out limit days (cannot buy/sell)
        valid_mask = (~panel['is_limit']) & panel[factor_name].notna() & panel['fwd_ret_1d'].notna()
        eval_df = panel[valid_mask].copy()
        
        if eval_df.empty:
            log.warning("No valid data for evaluation.")
            return
            
        # 1. Rank IC
        def _rank_ic(group):
            if len(group) < 10: return np.nan
            return group[factor_name].rank().corr(group['fwd_ret_1d'].rank(), method='pearson')
            
        ic_series = eval_df.groupby('date').apply(_rank_ic).dropna()
        
        mean_ic = ic_series.mean()
        ic_std = ic_series.std()
        icir = mean_ic / ic_std * np.sqrt(252) if ic_std != 0 else 0
        
        # Calculate Rolling 252-day IC (1 year) to observe decay
        rolling_mean_ic = ic_series.rolling(252, min_periods=60).mean()
        rolling_ic_std = ic_series.rolling(252, min_periods=60).std()
        rolling_icir = (rolling_mean_ic / rolling_ic_std) * np.sqrt(252)
        
        log.info(f"Rank IC Mean: {mean_ic:.4f}")
        log.info(f"Rank IC Std:  {ic_std:.4f}")
        log.info(f"Annual ICIR:  {icir:.4f}")
        if not rolling_icir.dropna().empty:
            log.info(f"Rolling 1Y ICIR (Latest): {rolling_icir.dropna().iloc[-1]:.4f}")
            log.info(f"Rolling 1Y IC Mean (Latest): {rolling_mean_ic.dropna().iloc[-1]:.4f}")
        
        # 2. Quantile Grouping
        # To avoid Lookahead bias, group by cross-sectional factor values each day
        def _group_factor(group):
            if len(group) < 10: return pd.Series(index=group.index, dtype=float)
            try:
                # Rank 1 to 5, where 5 is the highest factor value
                return pd.qcut(group[factor_name], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            except:
                return pd.Series(index=group.index, dtype=float)
                
        eval_df['quantile'] = eval_df.groupby('date').apply(_group_factor, include_groups=False).reset_index('date', drop=True)
        
        eval_df = eval_df.dropna(subset=['quantile'])
        
        group_returns = eval_df.groupby('quantile')['fwd_ret_1d'].mean() * 10000 # in bps
        log.info("Daily Mean Return by Quantile (bps):")
        for q, ret in group_returns.items():
            log.info(f"  Q{q}: {ret:.2f} bps")
            
        # Long-Short Return (Q5 - Q1)
        ls_ret = group_returns.get(5, 0) - group_returns.get(1, 0)
        log.info(f"Long-Short Spread (Q5-Q1): {ls_ret:.2f} bps/day")

if __name__ == "__main__":
    evaluator = SandboxEvaluator(start_date="20220101")  # Increased window to allow 252d rolling
    
    # ⚠️ [WARNING] 幸存者偏差提示 (Survivorship Bias Warning)
    # 当前沙盒使用静态的股票池进行快速验证。在严格的学术与实盘定型中，
    # 必须在每一期调仓日获取当时的成分股快照（包含后来退市的股票），
    # 否则在下行周期（如2022-2024）中，基于当下存活股票的历史测算会导致收益率虚高。
    # 后续演进路线将引入 Tushare/Baostock 的历史成分股动态切片功能。
    
    # test_pool = ["600519", "601318", "600036", "000858", "002594", 
    #              "000333", "600276", "601012", "300750", "002415",
    #              "601899", "601888", "603288", "000001", "000002"]
    test_pool = "ALL"
                 
    panel = evaluator.fetch_data(test_pool)
    if panel is not None:
        panel = evaluator.preprocess(panel)
        panel = evaluator.calc_factors(panel)
        
        # ML Evaluation out of sample
        evaluator.evaluate_ml_model(panel)
