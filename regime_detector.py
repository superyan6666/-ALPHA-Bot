import os
import joblib
import pandas as pd
import numpy as np
import logging
from hmmlearn.hmm import GaussianHMM
from datetime import datetime

log = logging.getLogger(__name__)

class RegimeDetector:
    def __init__(self, model_dir='.quantbot_data'):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, 'regime_hmm.pkl')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract daily market features from a panel DataFrame.
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # We need a cross-sectional daily aggregation
        pivot_close = df.pivot_table(index='date', columns='code', values='close')
        log_ret = np.log(pivot_close).diff()
        
        market_ret = log_ret.mean(axis=1)
        smoothed_ret = market_ret.rolling(window=10).mean()
        
        if len(log_ret.columns) > 1:
            market_vol = log_ret.std(axis=1).rolling(window=10).mean()
        else:
            # Fallback to rolling std for single index data
            market_vol = market_ret.rolling(window=10).std()
        
        if 'amount' in df.columns:
            pivot_amt = df.pivot_table(index='date', columns='code', values='amount')
            market_amt = (pivot_amt.median(axis=1) / 1e8).rolling(window=10).mean()
        else:
            market_amt = pd.Series(1.0, index=market_ret.index)
            
        features = pd.DataFrame({
            'ret': smoothed_ret,
            'vol': market_vol,
            'amt': market_amt
        }).dropna()
        
        return features

    def train(self, features: pd.DataFrame):
        """Train HMM Model"""
        if len(features) < 100:
            log.warning("[Regime] Not enough data to train HMM (<100 days).")
            return False
            
        X = features.values
        # Normalize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-5
        X_scaled = (X - self.mean_) / self.std_
        
        hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
        hmm.fit(X_scaled)
        
        joblib.dump({
            'model': hmm,
            'mean': self.mean_,
            'std': self.std_
        }, self.model_path)
        log.info(f"[Regime] HMM model trained and saved to {self.model_path}")
        return True

    def get_current_regime(self, df: pd.DataFrame) -> dict:
        """
        Returns a dict: {'state': int, 'label': str, 'probs': array}
        state: 0=震荡, 1=趋势, 2=危机
        """
        features = self.extract_features(df)
        if len(features) < 10:
            return {"state": 0, "label": "震荡期(缺数据)", "probs": [1.0, 0.0, 0.0]}
            
        if not os.path.exists(self.model_path):
            success = self.train(features)
            if not success:
                return {"state": 0, "label": "震荡期(无模型)", "probs": [1.0, 0.0, 0.0]}
            
        data = joblib.load(self.model_path)
        hmm = data['model']
        mean = data['mean']
        std = data['std']
        
        X = features.values
        X_scaled = (X - mean) / std
        
        probs = hmm.predict_proba(X_scaled)
        last_prob = probs[-1]
        raw_state = int(np.argmax(last_prob))
        
        means = hmm.means_ 
        vol_means = means[:, 1]
        ret_means = means[:, 0]
        
        # 稳健的状态映射逻辑：
        # 1. 波动率最低的状态 -> 震荡期 (Oscillating)
        # 2. 剩余两个状态中，收益率较高的 -> 趋势期 (Trend/Bull)
        # 3. 剩余两个状态中，收益率较低的 -> 危机期 (Crisis/Bear)
        
        sorted_by_vol = np.argsort(vol_means)
        osc_state = sorted_by_vol[0] # Lowest volatility
        
        rem = [sorted_by_vol[1], sorted_by_vol[2]]
        if ret_means[rem[0]] > ret_means[rem[1]]:
            trend_state = rem[0]
            crisis_state = rem[1]
        else:
            trend_state = rem[1]
            crisis_state = rem[0]
            
        mapping = {
            osc_state: (0, "震荡期(平稳)"),
            trend_state: (1, "趋势期(单边)"),
            crisis_state: (2, "危机期(高波)")
        }
        
        final_state, label = mapping[raw_state]
        
        mapped_probs = np.zeros(3)
        mapped_probs[0] = last_prob[osc_state]
        mapped_probs[1] = last_prob[trend_state]
        mapped_probs[2] = last_prob[crisis_state]
        
        return {
            "state": final_state,
            "label": label,
            "probs": mapped_probs.tolist()
        }
