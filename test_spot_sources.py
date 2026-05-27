import requests
import json
import logging
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO)

def fetch_tencent(codes):
    """
    Tencent API parsing based on explicit mapping to avoid index shifting problems.
    Tencent indices: 1=Name, 2=Code, 3=Price, 4=PrevClose, 5=Open, 32=Pct, 33=High, 34=Low, 36=Vol, 37=Amt, 38=Turn, 39=PE, 46=PB, 49=VR
    """
    # 格式化代码，加前缀
    formatted_codes = []
    for c in codes:
        if c.startswith('6'):
            formatted_codes.append(f'sh{c}')
        elif c.startswith('0') or c.startswith('3'):
            formatted_codes.append(f'sz{c}')
        elif c.startswith('8') or c.startswith('4'):
            formatted_codes.append(f'bj{c}')
            
    url = f"http://qt.gtimg.cn/q={','.join(formatted_codes)}"
    try:
        resp = requests.get(url, timeout=5)
        resp.encoding = 'gbk'
        lines = resp.text.strip().split('\n')
        
        results = []
        for line in lines:
            if '=' not in line: continue
            var, data = line.split('=', 1)
            parts = data.replace('"', '').replace(';', '').split('~')
            if len(parts) < 45:
                continue # 数据无效或停牌
                
            try:
                # 使用字典映射而不是单纯数组切片，处理可能为空的字符串
                parsed = {
                    'name': parts[1],
                    'code': parts[2],
                    'price': float(parts[3]) if parts[3] else None,
                    'pct_chg': float(parts[32]) if parts[32] else None,
                    'open': float(parts[5]) if parts[5] else None,
                    'high': float(parts[33]) if parts[33] else None,
                    'low': float(parts[34]) if parts[34] else None,
                    'volume': float(parts[36]) if parts[36] else None,
                    'amount': float(parts[37]) * 10000 if parts[37] else None,
                    'turnover': float(parts[38]) if parts[38] else 2.0,
                    'pe': float(parts[39]) if parts[39] else -1.0,
                    'pb': float(parts[46]) if len(parts)>46 and parts[46] else 2.0,
                    'vr': float(parts[49]) if len(parts)>49 and parts[49] else 1.0,
                    'source_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                results.append(parsed)
            except Exception as e:
                logging.warning(f"Error parsing tencent data for {parts[2]}: {e}")
                
        df = pd.DataFrame(results)
        return df
    except Exception as e:
        logging.error(f"Tencent API error: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    test_codes = ['600519', '000001', '300750']
    df = fetch_tencent(test_codes)
    print("Tencent Spot Data:")
    print(df)
