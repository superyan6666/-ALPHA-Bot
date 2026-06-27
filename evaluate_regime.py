import os
import pandas as pd
import numpy as np
import random
import sys
from regime_detector import RegimeDetector
import joblib

def main():
    from pipeline_manager import PipelineManager
    pm = PipelineManager('research/data/csi300_price.csv')
    df = pm.load_data()
    # Mock amount since csi300_price.csv doesn't have it
    if 'amount' not in df.columns:
        df['amount'] = df['volume'] * df['close']
    
    detector = RegimeDetector()
    features = detector.extract_features(df)
    
    detector.train(features)
    
    dates = features.index.tolist()
    
    random.seed(42)
    # Pick 20 points
    sample_indices = random.sample(range(20, len(dates)), 20)
    sample_indices.sort()
    
    print("="*80)
    print("HMM Regime Evaluation (20 Random Historical Points)")
    print("="*80)
    
    for i in sample_indices:
        d = dates[i]
        
        # Give detector data up to i
        sub_df = df[df['date'] <= d].copy()
        regime = detector.get_current_regime(sub_df)
        label = regime["label"]
        probs = regime["probs"]
        
        # Context: past 5 days return and vol
        past_5_ret = features['ret'].iloc[i-5:i+1].sum() * 100
        past_5_vol = features['vol'].iloc[i-5:i+1].mean() * 100
        
        prob_str = f"P(Osc)={probs[0]:.2f}, P(Trd)={probs[1]:.2f}, P(Cri)={probs[2]:.2f}"
        
        print(f"Date: {d.strftime('%Y-%m-%d')} | Label: {label:<10} | 5d Ret: {past_5_ret:6.2f}% | 5d Vol: {past_5_vol:5.2f}% | {prob_str}")

if __name__ == "__main__":
    main()
