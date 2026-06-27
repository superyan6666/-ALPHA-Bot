
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import time
import logging
import subprocess
import pandas as pd
import baostock as bs
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

os.environ['NO_PROXY'] = '*' # [B13 Fix] Bypass proxy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_all_a_share_codes():
    codes = set()
    bs.login()
    logging.info("Querying CSI 300 constituents from baostock...")
    rs = bs.query_hs300_stocks()
    while (rs.error_code == '0') and rs.next():
        codes.add(rs.get_row_data()[1])
        
    logging.info("Querying CSI 500 constituents from baostock...")
    rs = bs.query_zz500_stocks()
    while (rs.error_code == '0') and rs.next():
        codes.add(rs.get_row_data()[1])
        
    logging.info("Querying STAR 200 constituents from akshare...")
    try:
        import akshare as ak
        df_kc200 = ak.index_stock_cons(symbol='000699')
        for code in df_kc200['品种代码']:
            codes.add(f"sh.{code}")
    except Exception as e:
        logging.warning(f"Failed to fetch STAR 200 from akshare: {e}")
        
    bs.logout()
    return list(codes)

def fetch_stock_isolated(code):
    chunk_dir = ".quantbot_data/chunks"
    chunk_file = os.path.join(chunk_dir, f"{code}.parquet")
    if os.path.exists(chunk_file):
        return chunk_file
        
    max_retries = 3
    for attempt in range(max_retries):
        try:
            subprocess.run(
                ['python', 'fetch_one.py', code],
                timeout=45,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            if os.path.exists(chunk_file):
                logging.info(f"[{code}] Downloaded successfully.")
                return chunk_file
        except subprocess.TimeoutExpired:
            logging.warning(f"[{code}] Timeout on attempt {attempt+1}")
        except subprocess.CalledProcessError:
            logging.warning(f"[{code}] Failed on attempt {attempt+1}")
            
        time.sleep(2)
    logging.error(f"[{code}] Exceeded max retries.")
    return None

def build_data_lake(output_dir=".quantbot_data", max_stocks=None):
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "ashare_daily.parquet")
    
    codes = get_all_a_share_codes()
    logging.info(f"共获取到 {len(codes)} 只A股核心标的 (CSI 300 + 500)")
    
    if max_stocks:
        codes = codes[:max_stocks]
    
    workers = 6 # 线程池调用独立的子进程
    logging.info(f"启动多线程独立子进程池，并发数: {workers}")
    
    valid_chunks = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_stock_isolated, code): code for code in codes}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                valid_chunks.append(result)
            if (i + 1) % 50 == 0:
                logging.info(f"Progress: {i + 1} / {len(codes)}")
                
    logging.info(f"开始合并 {len(valid_chunks)} 个分块数据...")
    if valid_chunks:
        df_list = []
        for chunk_file in valid_chunks:
            try:
                df_list.append(pd.read_parquet(chunk_file))
            except Exception as e:
                logging.warning(f"Failed to read {chunk_file}: {e}")
                
        if df_list:
            final_df = pd.concat(df_list, ignore_index=True)
            final_df['date'] = pd.to_datetime(final_df['date']).dt.normalize()
            final_df = final_df.drop_duplicates(subset=['date', 'code'], keep='last')
            
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce').astype('float32')
            for col in ['tradestatus', 'isST']:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce').astype('Int8')
                
            logging.info(f"保存至 Parquet: {out_file}, 总行数: {len(final_df)}")
            final_df.to_parquet(out_file, index=False, compression='snappy')
            logging.info("离线数据湖构建完成！")
    else:
        logging.error("没有获取到任何数据！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='拉取前N只股票用于测试')
    args = parser.parse_args()
    build_data_lake(max_stocks=args.limit)
