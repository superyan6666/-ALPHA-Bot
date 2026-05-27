import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Append current directory to import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import DataProxy, C
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class SandboxEvaluator:
    def __init__(self, start_date="20230101", end_date=None):
        self.dp = DataProxy()
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y%m%d")
        
    def fetch_data(self, stock_list):
        """Fetch panel data for the given stocks."""
        log.info(f"Fetching data for {len(stock_list)} stocks from {self.start_date} to {self.end_date}...")
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
        # Calculate limit up/down based on previous close
        panel['close'] = pd.to_numeric(panel['close'])
        panel['high'] = pd.to_numeric(panel['high'])
        panel['low'] = pd.to_numeric(panel['low'])
        
        panel['prev_close'] = panel.groupby('code')['close'].shift(1)
        panel['pct_chg'] = (panel['close'] / panel['prev_close'] - 1) * 100
        
        # Mark limit up/down (approximate 9.5% for A-shares)
        panel['is_limit'] = (panel['pct_chg'].abs() >= 9.5) & (panel['high'] == panel['low'])
        
        # Calculate forward returns (T+1 to T+2)
        # If we buy at T+1 open, we want the return from T+1 Open to T+N Close
        # But for simplicity in alpha research, we often use Close to Close of T+1, or Open(T+1) to Open(T+2)
        # Let's use Close-to-Close forward 1 day: (Close(T+1) / Close(T)) - 1
        panel['fwd_ret_1d'] = panel.groupby('code')['close'].shift(-1) / panel['close'] - 1
        
        # Clean up NaNs
        panel = panel.dropna(subset=['fwd_ret_1d'])
        return panel

    def calc_factors(self, panel):
        """Calculate GTJA 191 trial factors."""
        log.info("Calculating factors...")
        
        # Factor 1: 5-day reversal (Dummy Alpha)
        panel['alpha_reversal_5d'] = - (panel['close'] / panel.groupby('code')['close'].shift(5) - 1)
        
        # Factor 2: GTJA Alpha 024 approximation
        # logic: SMA(CLOSE, 20) / CLOSE - 1 (simple distance to 20d MA, smaller means stronger momentum?)
        panel['alpha_024_approx'] = panel.groupby('code')['close'].transform(lambda x: x.rolling(20).mean()) / panel['close'] - 1
        
        return panel

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
                
        eval_df['quantile'] = eval_df.groupby('date').apply(_group_factor).reset_index(level=0, drop=True)
        
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
    
    # Use a static subset of CSI 300 for quick testing
    test_pool = ["600519", "601318", "600036", "000858", "002594", 
                 "000333", "600276", "601012", "300750", "002415",
                 "601899", "601888", "603288", "000001", "000002"]
                 
    panel = evaluator.fetch_data(test_pool)
    if panel is not None:
        panel = evaluator.preprocess(panel)
        panel = evaluator.calc_factors(panel)
        
        evaluator.evaluate(panel, 'alpha_reversal_5d')
        evaluator.evaluate(panel, 'alpha_024_approx')
