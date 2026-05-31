import pandas as pd
import numpy as np
import os
import logging

log = logging.getLogger(__name__)

def build_ml_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Unified ML Feature Engineering for both offline WFO and live production.
    Input: DataFrame with ['date', 'code', 'open', 'high', 'low', 'close', 'vol']
           Must be sorted by date!
    Output: DataFrame with appended feature columns.
    
    CRITICAL RULE (B2): "所有基于价格的指标必须使用 T-1 日已闭合的 K 线数据，并显式用 shift(1) 验证。"
    """
    log.info("Building unified ML features...")
    
    # 0. Base transformations
    if 'pct_chg' not in panel.columns:
        panel['prev_close'] = panel.groupby('code')['close'].shift(1)
        panel['pct_chg'] = (panel['close'] / panel['prev_close'] - 1) * 100
        
    # User Request: "所有基于价格的指标必须使用 T-1 日已闭合的 K 线数据，并显式用 shift(1) 验证。"
    # Create T-1 shifted series for computing lag features
    shifted_close = panel.groupby('code')['close'].shift(1)
    shifted_vol = panel.groupby('code')['vol'].shift(1)
    shifted_pct_chg = panel.groupby('code')['pct_chg'].shift(1)
    
    # 1. Smart Money Correlation (sm_corr) - using T-1
    # Note: kept for legacy compatibility, not actively used by best PyTorch model
    panel['sm_corr'] = panel.groupby('code').apply(
        lambda x: x['pct_chg'].shift(1).rolling(20).corr(x['vol'].shift(1))
    ).reset_index(0, drop=True)
    
    # 2. Amihud Illiquidity (amihud_20) - using T-1
    # Note: kept for legacy compatibility
    amihud_raw = shifted_pct_chg.abs() / (shifted_vol * shifted_close + 1e-5) * 1e6
    is_limit_shifted = panel.groupby('code')['is_limit'].shift(1) if 'is_limit' in panel.columns else False
    panel['amihud'] = np.where(is_limit_shifted, 99999.0, amihud_raw)
    panel['amihud_20'] = panel.groupby('code')['amihud'].transform(lambda x: x.rolling(20).mean())
    
    # 3. Close Location Value (CLV) - using T-1
    shifted_high = panel.groupby('code')['high'].shift(1)
    shifted_low = panel.groupby('code')['low'].shift(1)
    panel['clv'] = (shifted_close - shifted_low) / (shifted_high - shifted_low + 1e-8)
    
    # 4. Volatility and Volume - using T-1
    panel['volatility_5d'] = panel.groupby('code')['pct_chg'].transform(lambda x: x.shift(1).rolling(5).std())
    vol_mean_5d = panel.groupby('code')['vol'].transform(lambda x: x.shift(1).rolling(5).mean())
    panel['vol_ratio'] = shifted_vol / (vol_mean_5d + 1e-5)
    
    # 5. Momentum - using T-1
    panel['alpha_reversal_5d'] = - (shifted_close / panel.groupby('code')['close'].shift(6) - 1)
    close_ma_20_shifted = panel.groupby('code')['close'].transform(lambda x: x.shift(1).rolling(20).mean())
    panel['alpha_024_approx'] = close_ma_20_shifted / (shifted_close + 1e-5) - 1
    
    # 6. Market Regime Proxy (Global Broadcast) - using T-1
    market_daily = panel.groupby('date')['pct_chg'].mean().reset_index()
    market_daily.rename(columns={'pct_chg': 'market_ret'}, inplace=True)
    
    # Shift market returns by 1 to prevent leakage of Day T's market state
    market_daily['market_ret_shifted'] = market_daily['market_ret'].shift(1)
    market_daily['market_ret_20d'] = market_daily['market_ret_shifted'].rolling(20, min_periods=5).mean()
    market_daily['market_ret_60d'] = market_daily['market_ret_shifted'].rolling(60, min_periods=20).mean()
    market_daily['market_vol_20d'] = market_daily['market_ret_shifted'].rolling(20, min_periods=5).std()
    
    panel = pd.merge(panel, market_daily[['date', 'market_ret_20d', 'market_ret_60d', 'market_vol_20d']], on='date', how='left')
    
    # 7. Placeholder for Macro features as we deprecated them
    panel['cn_10y_trend'] = np.nan
    panel['macro_staleness_days'] = 0.0  # Added for compatibility with genes
        
    # [CRUCIBLE PROTOCOL] Downcast float64 to float32 for memory efficiency
    float_cols = panel.select_dtypes(include=['float64']).columns
    panel[float_cols] = panel[float_cols].astype(np.float32)
    
    return panel
