import os
import numpy as np
import pandas as pd
import logging
from scipy import stats
import xgboost as xgb
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# 1. Data Loader & Merger
def load_and_merge_data():
    data_dir = "../../research/data"
    
    # Load CSI 300 Price
    price_df = pd.read_csv(os.path.join(data_dir, "csi300_price.csv"))
    price_df.rename(columns={'Date': 'date'}, inplace=True)
    price_df['date'] = pd.to_datetime(price_df['date'])
    
    # Load CSI 300 PE
    pe_df = pd.read_csv(os.path.join(data_dir, "csi300_pe.csv"))
    pe_df.rename(columns={'Date': 'date'}, inplace=True)
    pe_df['date'] = pd.to_datetime(pe_df['date'])
    
    # Load China 10Y Yield
    yield_df = pd.read_csv(os.path.join(data_dir, "cn_10y_yield.csv"))
    yield_df.rename(columns={'Date': 'date'}, inplace=True)
    yield_df['date'] = pd.to_datetime(yield_df['date'])
    
    # Load VIX
    vix_df = pd.read_csv(os.path.join(data_dir, "vix_data.csv"))
    vix_df.rename(columns={'Date': 'date'}, inplace=True)
    vix_df['date'] = pd.to_datetime(vix_df['date'])
    
    # Merge on date
    df = price_df.set_index('date')
    df = df.join(pe_df.set_index('date'), how='left')
    df = df.join(yield_df.set_index('date'), how='left')
    df = df.join(vix_df.set_index('date'), how='left')
    
    # Sort and Forward Fill
    df = df.sort_index()
    df[['pe', 'yield_10y', 'vix', 'vix3m']] = df[['pe', 'yield_10y', 'vix', 'vix3m']].ffill()
    df = df.dropna(subset=['close']).reset_index()
    
    # Add Mock/Fallback Columns for compatibility with original framework
    df['code'] = '000300.SH'
    df['open'] = df['close']
    df['volume'] = 1e6
    
    log.info(f"Loaded and merged data: {len(df)} rows, columns: {list(df.columns)}")
    return df

# 2. Factor Calculation
def calculate_all_factors(df):
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    
    # Existing factors adapted for single asset (no groupby date z-score)
    # MOM
    df['F_MOM_5'] = df['close'].pct_change(5)
    df['F_MOM_10'] = df['close'].pct_change(10)
    df['F_MOM_20'] = df['close'].pct_change(20)
    
    # VOL
    df['F_VOL_5'] = df['close'].pct_change().rolling(5).std()
    df['F_VOL_10'] = df['close'].pct_change().rolling(10).std()
    df['F_VOL_20'] = df['close'].pct_change().rolling(20).std()
    
    # BIAS
    df['F_BIAS_5'] = df['close'] / df['close'].rolling(5).mean() - 1.0
    df['F_BIAS_10'] = df['close'] / df['close'].rolling(10).mean() - 1.0
    df['F_BIAS_20'] = df['close'] / df['close'].rolling(20).mean() - 1.0
    
    # SKEW
    df['F_SKEW_20'] = df['close'].pct_change().rolling(20).skew()
    # KURT
    df['F_KURT_20'] = df['close'].pct_change().rolling(20).kurt()
    
    # VP_REV
    df['F_VP_REV_5'] = -df['close'].pct_change(5)
    df['F_VP_REV_20'] = -df['close'].pct_change(20)
    
    # MACD_DIFF
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['F_MACD_DIFF'] = ema12 - ema26
    
    # BOLL_POS_20
    ma = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    df['F_BOLL_POS_20'] = (df['close'] - lower) / (upper - lower + 1e-8)
    
    # New Macro Factors
    # 1. ERP
    df['F_ERP'] = (100.0 / df['pe']) - df['yield_10y']
    
    # 2. VIX Term Structure
    df['F_VIX_TS'] = df['vix'] / df['vix3m'] - 1.0
    
    # 3. PE Valuation Z-score
    df['F_PE_Zscore_252'] = (df['pe'] - df['pe'].rolling(252).mean()) / (df['pe'].rolling(252).std() + 1e-8)
    
    # 4. Yield Momentum
    df['F_Yield_Momentum_20'] = df['yield_10y'] - df['yield_10y'].shift(20)
    
    # 5. PE Yield Ratio
    df['F_PE_Yield_Ratio'] = df['pe'] * df['yield_10y']
    
    # 6. VIX Momentum
    df['F_VIX_Momentum_10'] = df['vix'] - df['vix'].shift(10)
    
    # 7. VIX Volatility
    df['F_VIX_Volatility_20'] = df['vix'].rolling(20).std()
    
    # 8. PE Gap
    df['F_PE_Gap_120'] = df['pe'] / df['pe'].rolling(120).mean() - 1.0
    
    # We apply time-series standardisation (rolling z-score) for all factors to make them comparable
    factor_cols = [c for c in df.columns if c.startswith('F_')]
    for col in factor_cols:
        # Time-series zscore over rolling 252 days
        df[col] = (df[col] - df[col].rolling(252).mean()) / (df[col].rolling(252).std() + 1e-8)
        
    return df

# 3. Time Series Initial Screener
class TimeSeriesInitialScreener:
    def __init__(self, target_ret_window=20):
        self.target_ret_window = target_ret_window
        
    def screen(self, df, factor_cols):
        # Calculate target return
        df['next_open'] = df['open'].shift(-1)
        df['close_tn'] = df['close'].shift(-self.target_ret_window)
        df['fwd_ret'] = df['close_tn'] / (df['next_open'] + 1e-8) - 1.0
        # Since we have only one asset, target return is simply the forward return (or relative to a running mean)
        df['target'] = df['fwd_ret']
        
        # Dropna
        screened_df = df.dropna(subset=['target'] + factor_cols).copy()
        log.info(f"Rows after dropna: {len(screened_df)} / {len(df)}")
        
        # Calculate Time-Series IC (using monthly chunks for IR/t-stat)
        ic_results = []
        screened_df['Month'] = pd.to_datetime(screened_df['date']).dt.to_period('M')
        
        for f in factor_cols:
            # Overall Spearman Correlation
            overall_ic = stats.spearmanr(screened_df[f], screened_df['target'])[0]
            
            # Monthly Spearman Correlation
            monthly_ic = screened_df.groupby('Month').apply(
                lambda x: stats.spearmanr(x[f], x['target'])[0] if len(x) > 5 else np.nan
            ).dropna()
            
            if len(monthly_ic) < 10:
                continue
                
            mean_ic = monthly_ic.mean()
            std_ic = monthly_ic.std()
            ir = mean_ic / (std_ic + 1e-8)
            t_stat = mean_ic / (std_ic / np.sqrt(len(monthly_ic)) + 1e-8)
            
            log.info(f"Factor {f}: Overall IC = {overall_ic:.4f}, Mean Monthly IC = {mean_ic:.4f}, t-stat = {t_stat:.4f}")
            
            # We filter based on Time-Series Mean Monthly IC and t-stat
            # Using same thresholds: mean_ic > 0.015, t_stat > 1.5
            if abs(mean_ic) > 0.015 and abs(t_stat) > 1.5:
                ic_results.append({
                    'factor': f,
                    'mean_ic': mean_ic,
                    'ir': ir,
                    't_stat': t_stat
                })
                
        passed_df = pd.DataFrame(ic_results)
        if passed_df.empty:
            log.warning("No factors passed the time-series IC screen!")
            return screened_df, []
            
        passed_df = passed_df.sort_values(by='mean_ic', key=abs, ascending=False)
        log.info(f"Time-Series Initial Screen Passed: {len(passed_df)} factors out of {len(factor_cols)}")
        print(passed_df.to_string(index=False))
        return screened_df, passed_df['factor'].tolist()

# 4. Fine Screener
class FineScreener:
    def __init__(self, corr_threshold=0.7):
        self.corr_threshold = corr_threshold
        
    def screen(self, df, candidate_factors):
        log.info(f"Starting Fine Screening on {len(candidate_factors)} candidates...")
        dates = sorted(df['date'].unique())
        split_idx = int(len(dates) * 0.8)
        train_dates = dates[:split_idx]
        
        train_df = df[df['date'].isin(train_dates)]
        X_train = train_df[candidate_factors]
        y_train = train_df['target']
        
        log.info("Training XGBoost Regressor to extract feature importance...")
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05, 
            subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=42
        )
        model.fit(X_train, y_train)
        
        importance = model.feature_importances_
        fi_df = pd.DataFrame({'factor': candidate_factors, 'gain': importance})
        fi_df = fi_df.sort_values(by='gain', ascending=False)
        
        log.info(f"Removing collinear factors (Pearson > {self.corr_threshold})...")
        corr_matrix = train_df[fi_df['factor']].corr(method='pearson').abs()
        
        selected_factors = []
        for factor in fi_df['factor']:
            if len(selected_factors) == 0:
                selected_factors.append(factor)
                continue
                
            is_collinear = False
            for sel in selected_factors:
                if corr_matrix.loc[factor, sel] > self.corr_threshold:
                    is_collinear = True
                    break
                    
            if not is_collinear:
                selected_factors.append(factor)
                
        log.info(f"Fine Screen Passed: {len(selected_factors)} factors remaining.")
        final_importance = fi_df[fi_df['factor'].isin(selected_factors)]
        print(final_importance.to_string(index=False))
        
        return selected_factors

# Run test
if __name__ == "__main__":
    df = load_and_merge_data()
    df = calculate_all_factors(df)
    
    factor_cols = [c for c in df.columns if c.startswith('F_')]
    
    screener1 = TimeSeriesInitialScreener(target_ret_window=20)
    screened_df, passed_initial = screener1.screen(df, factor_cols)
    
    if passed_initial:
        screener2 = FineScreener(corr_threshold=0.7)
        final_factors = screener2.screen(screened_df, passed_initial)
        print(f"\nFinal Promoted Factors: {final_factors}")
    else:
        print("\nNo factors promoted.")
