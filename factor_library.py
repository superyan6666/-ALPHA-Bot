import pandas as pd
import numpy as np
import logging
from typing import Callable, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class FactorRegistry:
    def __init__(self):
        self.factors: Dict[str, dict] = {}
        
    def register(self, name: str, category: str = "price_momentum"):
        def decorator(func: Callable):
            self.factors[name] = {"func": func, "category": category}
            return func
        return decorator
        
    def get_all_factors(self) -> Dict[str, dict]:
        return self.factors

registry = FactorRegistry()

def mad_outlier(series: pd.Series, n=3.0) -> pd.Series:
    median = series.median()
    mad = (series - median).abs().median()
    lower_bound = median - n * 1.4826 * mad
    upper_bound = median + n * 1.4826 * mad
    return series.clip(lower=lower_bound, upper=upper_bound)

def zscore_standardize(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / (series.std() + 1e-8)

def neutralize(series: pd.Series, mcap_series: pd.Series) -> pd.Series:
    """Neutralize a factor series against market cap."""
    df_temp = pd.DataFrame({'factor': series, 'mcap': np.log(mcap_series + 1e-8)}).dropna()
    if len(df_temp) < 30:
        return series
        
    X = np.vstack([df_temp['mcap'], np.ones(len(df_temp))]).T
    y = df_temp['factor'].values
    
    try:
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X.dot(coef)
        neutral_series = series.copy()
        neutral_series.loc[df_temp.index] = resid
        return neutral_series
    except Exception:
        return series

def _group_neutralize(g, col_name):
    g[col_name] = neutralize(g[col_name], g['F_float_cap'])
    return g

def apply_factor(df: pd.DataFrame, factor_func: Callable, col_name: str, category: str = "price_momentum", apply_mad=True, apply_zscore=True) -> pd.DataFrame:
    res = factor_func(df)
    if isinstance(res, pd.Series):
        df[col_name] = res
    else:
        log.error(f"Factor {col_name} did not return a Series!")
        return df

    is_cross_section = df['code'].nunique() > 1
    
    if is_cross_section and category in ['value', 'quality', 'fundamental'] and 'F_float_cap' in df.columns:
        df = df.groupby('date', group_keys=False).apply(lambda g: _group_neutralize(g, col_name))

    if is_cross_section:
        # Normalization per cross-section
        if apply_mad:
            df[col_name] = df.groupby('date')[col_name].transform(mad_outlier)
        if apply_zscore:
            df[col_name] = df.groupby('date')[col_name].transform(zscore_standardize)
    else:
        # Skip cross-sectional normalization and apply rolling 252-day z-score normalization
        series = df[col_name]
        df[col_name] = (series - series.rolling(252).mean()) / (series.rolling(252).std() + 1e-8)
    return df

# 自动化批量注册函数 (利用闭包与参数绑定)
def register_window_factors():
    windows = [5, 10, 20]
    
    for w in windows:
        # MOM (动量)
        @registry.register(f"MOM_{w}", category="price_momentum")
        def _f(df, w=w): return df.groupby('code')['close'].pct_change(w)
        
        # VOL (波动率)
        @registry.register(f"VOL_{w}", category="volume")
        def _f(df, w=w): 
            return df.groupby('code')['close'].pct_change().groupby(df['code']).transform(lambda x: x.rolling(w).std())

        # BIAS (乖离率)
        @registry.register(f"BIAS_{w}", category="price_momentum")
        def _f(df, w=w):
            ma = df.groupby('code')['close'].transform(lambda x: x.rolling(w).mean())
            return df['close'] / (ma + 1e-8) - 1.0
            
    # ------ 以下为新增的高阶与另类因子池 ------
    
    for w in [20]:
        # SKEW (偏度 - 捕捉收益不对称性)
        @registry.register(f"SKEW_{w}", category="volume")
        def _f(df, w=w):
            ret = df.groupby('code')['close'].pct_change()
            return ret.groupby(df['code']).transform(lambda x: x.rolling(w).skew())

        # KURT (峰度 - 捕捉尾部风险)
        @registry.register(f"KURT_{w}", category="volume")
        def _f(df, w=w):
            ret = df.groupby('code')['close'].pct_change()
            return ret.groupby(df['code']).transform(lambda x: x.rolling(w).kurt())
            
    for w in [5, 20]:
        # VP_REV (量价背离反转)
        @registry.register(f"VP_REV_{w}", category="vol")
        def _f(df, w=w):
            ret = df.groupby('code')['close'].pct_change(w)
            vol_ma = df.groupby('code')['vol'].transform(lambda x: x.rolling(w).mean())
            return -ret * (df['vol'] / (vol_ma + 1e-8))

register_window_factors()

@registry.register("MACD_DIFF", category="price_momentum")
def calc_macd_diff(df: pd.DataFrame) -> pd.Series:
    ema12 = df.groupby('code')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = df.groupby('code')['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    return ema12 - ema26

@registry.register("BOLL_POS_20", category="price_momentum")
def calc_boll_pos(df: pd.DataFrame) -> pd.Series:
    ma = df.groupby('code')['close'].transform(lambda x: x.rolling(20).mean())
    std = df.groupby('code')['close'].transform(lambda x: x.rolling(20).std())
    upper = ma + 2 * std
    lower = ma - 2 * std
    return (df['close'] - lower) / (upper - lower + 1e-8)

@registry.register("ERP", category="macro")
def calc_erp(df: pd.DataFrame) -> pd.Series:
    return (100.0 / df['pe']) - df['yield_10y']

@registry.register("VIX_TS", category="macro")
def calc_vix_ts(df: pd.DataFrame) -> pd.Series:
    return df['vix'] / df['vix3m'] - 1.0

@registry.register("PE_Zscore_252", category="value")
def calc_pe_zscore_252(df: pd.DataFrame) -> pd.Series:
    pe_mean = df.groupby('code')['pe'].transform(lambda x: x.rolling(252).mean())
    pe_std = df.groupby('code')['pe'].transform(lambda x: x.rolling(252).std())
    return (df['pe'] - pe_mean) / (pe_std + 1e-8)

@registry.register("Yield_Momentum_20", category="macro")
def calc_yield_momentum_20(df: pd.DataFrame) -> pd.Series:
    yield_shift = df.groupby('code')['yield_10y'].shift(20)
    return df['yield_10y'] - yield_shift

@registry.register("PE_Yield_Ratio", category="value")
def calc_pe_yield_ratio(df: pd.DataFrame) -> pd.Series:
    return df['pe'] * df['yield_10y']

@registry.register("VIX_Momentum_10", category="macro")
def calc_vix_momentum_10(df: pd.DataFrame) -> pd.Series:
    vix_shift = df.groupby('code')['vix'].shift(10)
    return df['vix'] - vix_shift

@registry.register("VIX_Volatility_20", category="macro")
def calc_vix_volatility_20(df: pd.DataFrame) -> pd.Series:
    return df.groupby('code')['vix'].transform(lambda x: x.rolling(20).std())

@registry.register("PE_Gap_120", category="value")
def calc_pe_gap_120(df: pd.DataFrame) -> pd.Series:
    pe_mean = df.groupby('code')['pe'].transform(lambda x: x.rolling(120).mean())
    return df['pe'] / pe_mean - 1.0

def calculate_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Generate all registered factors."""
    log.info(f"Starting factor generation on {len(df)} rows using FactorRegistry...")
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    factors = registry.get_all_factors()
    log.info(f"Discovered {len(factors)} registered factors. Applying pipeline...")
    
    for name, factor_info in factors.items():
        col_name = f'F_{name}'
        func = factor_info['func']
        category = factor_info['category']
        df = apply_factor(df, func, col_name, category)
        
    log.info("Factor generation complete.")
    return df
