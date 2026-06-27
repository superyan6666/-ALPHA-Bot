import baostock as bs
import pandas as pd
import json

bs.login()

with open('advisory_tracker.json.bak', 'r', encoding='utf-8') as f:
    tracker = json.load(f)

print('Stock Performance from 2026-06-01 to 2026-06-10:\n')

for code, info in tracker.items():
    full_code = f'sh.{code}' if code.startswith('6') else f'sz.{code}'
    rs = bs.query_history_k_data_plus(full_code, 'date,open,high,low,close,pctChg', start_date='2026-06-01', end_date='2026-06-10', frequency='d', adjustflag='2')
    
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    
    if not data_list:
        print(f'{info["name"]} ({full_code}): No data found.')
        continue
        
    df = pd.DataFrame(data_list, columns=rs.fields)
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['open'] = df['open'].astype(float)
    
    entry_price = df.iloc[0]['open']  # assuming bought at next day open (June 2 or June 1 open)
    latest_close = df.iloc[-1]['close']
    max_high = df['high'].max()
    min_low = df['low'].min()
    
    total_return = (latest_close / entry_price - 1) * 100
    max_gain = (max_high / entry_price - 1) * 100
    max_drawdown = (min_low / entry_price - 1) * 100
    
    status = 'ACTIVE'
    if min_low <= info['stop']:
        status = 'STOP_LOSS_TRIGGERED'
    
    print(f'{info["name"]} ({full_code})')
    print(f'  Entry Price: {entry_price:.2f}')
    print(f'  Latest Close: {latest_close:.2f} ({total_return:+.2f}%)')
    print(f'  Max Gain: {max_gain:+.2f}%')
    print(f'  Max Drawdown: {max_drawdown:+.2f}%')
    print(f'  Stop Loss Threshold: {info["stop"]} -> Status: {status}\n')

bs.logout()
