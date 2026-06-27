import pandas as pd
import sys
sys.path.append('.')
from factors_config import get_factors_config

df = pd.read_parquet('.quantbot_data/ashare_daily.parquet')
latest_date = df['date'].max()
df_latest = df[df['date'] == latest_date].copy()
print(f'Latest Date in Cache: {latest_date}, Total Stocks: {len(df_latest)}')

df_latest = df_latest[df_latest['pe'] > 0]
df_latest = df_latest[(df_latest['pb'] >= 0.1) & (df_latest['pb'] <= 20.0)]

factors = get_factors_config(f_val=1.0, f_mom=1.0, f_rev=1.0, f_risk=1.0, tw=1.0, rw=1.0, m_regime='BULL', in_danger=False, danger_label='')

results = []
for idx, row in df_latest.iterrows():
    data = row.to_dict()
    score = 0
    log = []
    for factor in factors:
        try:
            if factor.condition(data):
                score += factor.points
                if factor.template:
                    log.append(factor.template)
        except:
            pass
    results.append({'code': data['code'], 'score': score, 'log': log})

res_df = pd.DataFrame(results)
res_df = res_df.sort_values('score', ascending=False)
print('\n=== Top 5 Signals (Phase 2) ===')
for i, row in res_df.head(5).iterrows():
    print(f"{row['code']} - Score: {row['score']}")
    for l in row['log']:
        if '共振' in l or '天量' in l or '飞刀' in l or '过热' in l:
            print('  >>> ' + l)
        else:
            print('  ' + l)
