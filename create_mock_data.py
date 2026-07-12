import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

cache_dir = '/workspace/hist_cache'
os.makedirs(cache_dir, exist_ok=True)

today = datetime.now()
dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(120)]

spot_data = []
stocks = [
    ('600519', '贵州茅台', 1680.0, 2.5),
    ('601318', '中国平安', 48.5, 1.2),
    ('600036', '招商银行', 35.8, 0.8),
    ('000858', '五粮液', 145.0, 1.5),
    ('002594', '比亚迪', 258.0, 3.2),
    ('300750', '宁德时代', 185.0, 2.8),
    ('601012', '隆基绿能', 28.5, 1.8),
    ('600276', '恒瑞医药', 45.0, 0.5),
    ('510300', '沪深300ETF', 4.85, 0.3),
    ('510500', '中证500ETF', 7.28, 0.5),
    ('512760', '半导体ETF', 1.25, 2.1),
    ('512010', '医药ETF', 0.85, 0.3),
]

for code, name, price, pct in stocks:
    spot_data.append({
        '代码': code,
        '名称': name,
        '最新价': price,
        '涨跌幅': pct,
        '今开': price * 0.995,
        '最高': price * 1.015,
        '最低': price * 0.985,
        '成交量': 10000000,
        '成交额': price * 10000000 * 100,
        '换手率': 2.5,
        '市盈率-动态': 25.0,
        '市净率': 3.5,
        '量比': 1.2,
        '流通市值': 50000000000,
    })

spot_df = pd.DataFrame(spot_data)
with open(os.path.join(cache_dir, 'spot.pkl'), 'wb') as f:
    pickle.dump({
        'created_at': int(datetime.now().timestamp()),
        'data': spot_df
    }, f)
print("Created spot.pkl")

index_data = []
close = 3250.0
for date in dates:
    change = np.random.normal(0, 8)
    close = max(3000, min(3500, close + change))
    index_data.append({
        'date': date,
        'open': close - np.random.uniform(0, 5),
        'close': close,
        'high': close + np.random.uniform(0, 6),
        'low': close - np.random.uniform(0, 6),
        'volume': 30000000000,
    })

index_df = pd.DataFrame(index_data)
with open(os.path.join(cache_dir, 'index_sh000001.pkl'), 'wb') as f:
    pickle.dump({
        'created_at': int(datetime.now().timestamp()),
        'data': index_df
    }, f)
print("Created index_sh000001.pkl")

close = 4850.0
index_data = []
for date in dates:
    change = np.random.normal(0, 10)
    close = max(4000, min(5500, close + change))
    index_data.append({
        'date': date,
        'open': close - np.random.uniform(0, 6),
        'close': close,
        'high': close + np.random.uniform(0, 7),
        'low': close - np.random.uniform(0, 7),
        'volume': 20000000000,
    })
index_df = pd.DataFrame(index_data)
with open(os.path.join(cache_dir, 'index_sh000300.pkl'), 'wb') as f:
    pickle.dump({
        'created_at': int(datetime.now().timestamp()),
        'data': index_df
    }, f)
print("Created index_sh000300.pkl")

close = 7200.0
index_data = []
for date in dates:
    change = np.random.normal(0, 15)
    close = max(6000, min(8500, close + change))
    index_data.append({
        'date': date,
        'open': close - np.random.uniform(0, 8),
        'close': close,
        'high': close + np.random.uniform(0, 9),
        'low': close - np.random.uniform(0, 9),
        'volume': 15000000000,
    })
index_df = pd.DataFrame(index_data)
with open(os.path.join(cache_dir, 'index_sh000905.pkl'), 'wb') as f:
    pickle.dump({
        'created_at': int(datetime.now().timestamp()),
        'data': index_df
    }, f)
print("Created index_sh000905.pkl")

close = 2100.0
index_data = []
for date in dates:
    change = np.random.normal(0, 12)
    close = max(1700, min(2500, close + change))
    index_data.append({
        'date': date,
        'open': close - np.random.uniform(0, 6),
        'close': close,
        'high': close + np.random.uniform(0, 7),
        'low': close - np.random.uniform(0, 7),
        'volume': 18000000000,
    })
index_df = pd.DataFrame(index_data)
with open(os.path.join(cache_dir, 'index_sz399006.pkl'), 'wb') as f:
    pickle.dump({
        'created_at': int(datetime.now().timestamp()),
        'data': index_df
    }, f)
print("Created index_sz399006.pkl")

for code, name, price, _ in stocks[:12]:
    hist_data = []
    close = price * 0.8
    for date in dates:
        change = np.random.normal(0, price * 0.01)
        close = max(price * 0.5, min(price * 1.5, close + change))
        hist_data.append({
            '日期': date,
            '开盘': close - np.random.uniform(0, price * 0.005),
            '收盘': close,
            '最高': close + np.random.uniform(0, price * 0.008),
            '最低': close - np.random.uniform(0, price * 0.008),
            '成交量': 5000000,
        })
    
    hist_df = pd.DataFrame(hist_data)
    end_date = dates[0].replace('-', '')
    with open(os.path.join(cache_dir, f'hist_{code}_{end_date}.pkl'), 'wb') as f:
        pickle.dump({
            'created_at': int(datetime.now().timestamp()),
            'data': hist_df
        }, f)
    print(f"Created hist_{code}_{end_date}.pkl")

core_pool = {
    '600519', '601318', '600036', '000858', '002594',
    '300750', '601012', '600276', '510300', '510500',
}
with open(os.path.join(cache_dir, 'core_pool.pkl'), 'wb') as f:
    pickle.dump({
        'created_at': int(datetime.now().timestamp()),
        'data': core_pool
    }, f)
print("Created core_pool.pkl")

hot_sectors = {
    '600519': '白酒',
    '000858': '白酒',
    '300750': '新能源',
    '002594': '新能源',
    '512760': '半导体',
}
with open(os.path.join(cache_dir, 'hot_sectors.pkl'), 'wb') as f:
    pickle.dump({
        'created_at': int(datetime.now().timestamp()),
        'data': hot_sectors
    }, f)
print("Created hot_sectors.pkl")

northbound = (35.5, "\n- 🌊 **聪明钱流向**：北水大举流入 **+35亿**")
with open(os.path.join(cache_dir, 'northbound.pkl'), 'wb') as f:
    pickle.dump({
        'created_at': int(datetime.now().timestamp()),
        'data': northbound
    }, f)
print("Created northbound.pkl")

print("\n✅ 所有模拟数据创建完成！")