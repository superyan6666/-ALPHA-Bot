
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

def calculate_metrics(returns):
    mean_ret = returns.mean() * 10000  # bps
    annual_ret = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
    return mean_ret, annual_ret, sharpe

print("Loading OOS predictions...")
df = pd.read_csv(".quantbot_data/oos_preds.csv")

horizons = ['xgb_score_t1', 'xgb_score_t5', 'xgb_score_t10', 'xgb_score_t20']
df = df.dropna(subset=horizons + ['fwd_ret_1d']).copy()

# Deduct cross-sectional mean to get pure Alpha
daily_mean = df.groupby('date')['fwd_ret_1d'].transform('mean')
df['alpha_1d'] = df['fwd_ret_1d'] - daily_mean

for col in horizons:
    df[f'{col}_rank'] = df.groupby('date')[col].rank(pct=True, ascending=True)

df['consensus_score'] = 0
for col in horizons:
    df['consensus_score'] += (df[f'{col}_rank'] >= 0.80).astype(int)

df['avg_rank'] = df[[f'{col}_rank' for col in horizons]].mean(axis=1)
df['composite_score'] = df['consensus_score'] + df['avg_rank']
df['ensemble_rank'] = df.groupby('date')['composite_score'].rank(pct=True, ascending=True)

df['is_top_ensemble'] = df['ensemble_rank'] >= 0.90
ensemble_alpha = df[df['is_top_ensemble']].groupby('date')['alpha_1d'].mean()

print("\n=== Pure Alpha by Consensus Score (Top 20% overlap) ===")
consensus_alpha = df.groupby(['date', 'consensus_score'])['alpha_1d'].mean().unstack()
counts = df['consensus_score'].value_counts(normalize=True) * 100

for score in sorted(df['consensus_score'].unique()):
    m_bps, a_ret, sh = calculate_metrics(consensus_alpha[score].dropna())
    print(f"Consensus={score} (Count: {counts[score]:.1f}%): Alpha {m_bps:+.2f} bps/day | Sharpe {sh:.2f}")

print("\n=== Pure Alpha for Top 10% Decile ===")
for col in horizons:
    df[f'is_top_{col}'] = df[f'{col}_rank'] >= 0.90
    port_alpha = df[df[f'is_top_{col}']].groupby('date')['alpha_1d'].mean()
    m_bps, _, sh = calculate_metrics(port_alpha)
    print(f"{col}: Alpha {m_bps:+.2f} bps/day | Sharpe {sh:.2f}")

m_bps, _, sh = calculate_metrics(ensemble_alpha)
print(f"Ensemble Rank: Alpha {m_bps:+.2f} bps/day | Sharpe {sh:.2f}")
