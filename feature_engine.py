import pandas as pd
import numpy as np
import os
import logging

log = logging.getLogger(__name__)

def load_optimal_horizons():
    """动态读取因子周期，附带优雅降级机制 (B3.6)"""
    default_horizons = {
        'rsi': 60,
        'mom_long': 120,
        'mom_mid': 20,
        'bias_long': 120,
        'bias_mid': 60,
        'amihud': 20,
        'drawdown': 120,
        'runup': 120,
        'atr': 60,
        'pv_corr': 20
    }
    
    filepath = '.quantbot_data/factor_optimal_horizons.csv'
    if not os.path.exists(filepath):
        log.info("ℹ️ 未检测到 factor_optimal_horizons.csv, [系统降级] 使用默认静态周期。")
        return default_horizons
        
    try:
        mapping_df = pd.read_csv(filepath)
        dynamic_dict = dict(zip(mapping_df['factor'], mapping_df['anchor_window']))
        
        if 'rsi' in dynamic_dict: default_horizons['rsi'] = dynamic_dict['rsi']
        if 'amihud' in dynamic_dict: default_horizons['amihud'] = dynamic_dict['amihud']
        if 'drawdown' in dynamic_dict: default_horizons['drawdown'] = dynamic_dict['drawdown']
        if 'runup' in dynamic_dict: default_horizons['runup'] = dynamic_dict['runup']
        if 'atr' in dynamic_dict: default_horizons['atr'] = dynamic_dict['atr']
        if 'pv_corr' in dynamic_dict: default_horizons['pv_corr'] = dynamic_dict['pv_corr']
        if 'mom' in dynamic_dict: default_horizons['mom_long'] = dynamic_dict['mom']
        if 'bias' in dynamic_dict: default_horizons['bias_long'] = dynamic_dict['bias']
            
        log.info(f"💡 [FeatureEngine] 已成功挂载动态因子最优周期表: {dynamic_dict}")
    except Exception as e:
        log.warning(f"⚠️ 解析动态周期表失败 ({e}), [系统降级] 强制使用默认静态周期。")
        
    return default_horizons

def build_ml_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Unified ML Feature Engineering for both offline WFO and live production.
    Input: DataFrame with ['date', 'code', 'open', 'high', 'low', 'close', 'vol']
           Must be sorted by date!
    Output: DataFrame with appended feature columns.
    
    CRITICAL RULE (B2): "所有基于价格的指标必须使用 T-1 日已闭合的 K 线数据，并显式用 shift(1) 验证。"
    """
    log.info("Building unified ML features...")
    horizons = load_optimal_horizons()
    
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
    w_amihud = int(horizons['amihud'])
    amihud_raw = shifted_pct_chg.abs() / (shifted_vol * shifted_close + 1e-5) * 1e6
    is_limit_shifted = panel.groupby('code')['is_limit'].shift(1) if 'is_limit' in panel.columns else False
    panel['amihud'] = np.where(is_limit_shifted, 99999.0, amihud_raw)
    panel[f'F_amihud_{w_amihud}'] = panel.groupby('code')['amihud'].transform(lambda x: x.rolling(w_amihud).mean())
    
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
    
    # 7. Long-Term Trend & Momentum Features (Level 2 Upgrade)
    # [重要加固] 极值前置处理 (Winsorize): 保护长线指标不受异常暴涨暴跌污染
    # We clip daily returns to [-20%, 20%] to prevent data errors from compounding into long-term MAs
    pct_chg_win = shifted_pct_chg.clip(lower=-20.0, upper=20.0)
    
    # 7.1 RSI
    w_rsi = int(horizons['rsi'])
    gain = pct_chg_win.where(pct_chg_win > 0, 0.0)
    loss = -pct_chg_win.where(pct_chg_win < 0, 0.0)
    avg_gain = panel.groupby('code')['pct_chg'].transform(lambda x: x.shift(1).clip(lower=-20.0, upper=20.0).where(x.shift(1) > 0, 0.0).rolling(w_rsi, min_periods=w_rsi//3).mean())
    avg_loss = panel.groupby('code')['pct_chg'].transform(lambda x: -x.shift(1).clip(lower=-20.0, upper=20.0).where(x.shift(1) < 0, 0.0).rolling(w_rsi, min_periods=w_rsi//3).mean())
    rs = avg_gain / (avg_loss + 1e-5)
    panel[f'F_rsi_{w_rsi}'] = 100 - (100 / (1 + rs))

    # 7.2 MACD (12, 26, 9) - Normalized by price for cross-sectional consistency
    ema12 = panel.groupby('code')['close'].transform(lambda x: x.shift(1).ewm(span=12, adjust=False).mean())
    ema26 = panel.groupby('code')['close'].transform(lambda x: x.shift(1).ewm(span=26, adjust=False).mean())
    panel['F_macd'] = (ema12 - ema26) / (ema26 + 1e-5) * 100  # Percentage
    panel['F_macd_signal'] = panel.groupby('code')['F_macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    panel['F_macd_hist'] = panel['F_macd'] - panel['F_macd_signal']

    # 7.3 Bias
    w_bias_mid = int(horizons['bias_mid'])
    w_bias_long = int(horizons['bias_long'])
    ma_mid = panel.groupby('code')['close'].transform(lambda x: x.shift(1).rolling(w_bias_mid, min_periods=w_bias_mid//3).mean())
    ma_long = panel.groupby('code')['close'].transform(lambda x: x.shift(1).rolling(w_bias_long, min_periods=w_bias_long//3).mean())
    panel[f'F_bias_{w_bias_mid}'] = (shifted_close / (ma_mid + 1e-5)) - 1
    panel[f'F_bias_{w_bias_long}'] = (shifted_close / (ma_long + 1e-5)) - 1

    # 7.4 Momentum
    w_mom_long = int(horizons['mom_long'])
    panel[f'F_mom_{w_mom_long}'] = (shifted_close / panel.groupby('code')['close'].shift(w_mom_long + 1)) - 1
    
    # 7.5 Drawdown / Runup
    w_drawdown = int(horizons['drawdown'])
    w_runup = int(horizons['runup'])
    roll_max = panel.groupby('code')['high'].transform(lambda x: x.shift(1).rolling(w_drawdown, min_periods=w_drawdown//3).max())
    roll_min = panel.groupby('code')['low'].transform(lambda x: x.shift(1).rolling(w_runup, min_periods=w_runup//3).min())
    panel[f'F_drawdown_{w_drawdown}'] = (shifted_close / (roll_max + 1e-5)) - 1
    panel[f'F_runup_{w_runup}'] = (shifted_close / (roll_min + 1e-5)) - 1
    
    # 7.6 Volume Trend (量能中枢突破)
    vol_ma20 = panel.groupby('code')['vol'].transform(lambda x: x.shift(1).rolling(20, min_periods=10).mean())
    vol_ma120 = panel.groupby('code')['vol'].transform(lambda x: x.shift(1).rolling(120, min_periods=40).mean())
    panel['F_vol_trend_20_120'] = vol_ma20 / (vol_ma120 + 1e-5)
    
    # 7.7 Short-term Pullback Momentum
    w_mom_mid = int(horizons['mom_mid'])
    panel[f'F_mom_{w_mom_mid}'] = (shifted_close / panel.groupby('code')['close'].shift(w_mom_mid + 1)) - 1
    
    # 7.8 Normalized ATR (真实波动率收敛 VCP)
    w_atr = int(horizons['atr'])
    eps = 1e-8
    prev_shifted_close = panel.groupby('code')['close'].shift(2)
    shifted_high = panel.groupby('code')['high'].shift(1)
    shifted_low = panel.groupby('code')['low'].shift(1)
    
    tr1 = shifted_high - shifted_low
    tr2 = (shifted_high - prev_shifted_close).abs()
    tr3 = (shifted_low - prev_shifted_close).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    panel[f'F_atr_{w_atr}_norm'] = true_range.groupby(panel['code']).transform(
        lambda x: x.rolling(w_atr, min_periods=w_atr//2).mean()
    ) / (shifted_close + eps)
    
    # 7.9 Price-Volume Divergence (量价背离系数)
    w_pv = int(horizons['pv_corr'])
    def calc_pv_corr(g):
        return g['close'].shift(1).rolling(window=w_pv, min_periods=w_pv//2).corr(g['vol'].shift(1))

    panel[f'F_pv_corr_{w_pv}'] = panel.groupby('code', group_keys=False).apply(calc_pv_corr)
    panel[f'F_pv_corr_{w_pv}'] = panel[f'F_pv_corr_{w_pv}'].fillna(0.0)
    
    # [重要加固] 截面去极值 (Cross-sectional Winsorization)
    # 防止极端的特征破坏神经网络的 batch normalization
    lt_features = [
        f'F_rsi_{w_rsi}', 'F_macd', 'F_macd_signal', 'F_macd_hist', 
        f'F_bias_{w_bias_mid}', f'F_bias_{w_bias_long}', f'F_mom_{w_mom_long}', f'F_drawdown_{w_drawdown}', f'F_runup_{w_runup}',
        'F_vol_trend_20_120', f'F_mom_{w_mom_mid}', f'F_atr_{w_atr}_norm', f'F_pv_corr_{w_pv}'
    ]
    for f in lt_features:
        # Winsorize at 1% and 99% quantiles per day
        lower = panel.groupby('date')[f].transform(lambda x: x.quantile(0.01))
        upper = panel.groupby('date')[f].transform(lambda x: x.quantile(0.99))
        panel[f] = panel[f].clip(lower=lower, upper=upper)

    # Note: We rename existing features to prefix with F_ for unified filtering
    panel.rename(columns={
        'sm_corr': 'F_sm_corr',
        'clv': 'F_clv',
        'volatility_5d': 'F_volatility_5d',
        'vol_ratio': 'F_vol_ratio',
        'alpha_reversal_5d': 'F_alpha_reversal_5d',
        'alpha_024_approx': 'F_alpha_024_approx',
        'market_ret_20d': 'F_market_ret_20d',
        'market_ret_60d': 'F_market_ret_60d',
        'market_vol_20d': 'F_market_vol_20d'
    }, inplace=True)
    
    # Deprecated Macro
    panel['F_cn_10y_trend'] = np.nan

        
    # [CRUCIBLE PROTOCOL] Downcast float64 to float32 for memory efficiency
    float_cols = panel.select_dtypes(include=['float64']).columns
    panel[float_cols] = panel[float_cols].astype(np.float32)
    
    return panel
