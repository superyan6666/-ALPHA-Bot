import pandas as pd
import akshare as ak
import os
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_code(code, start_date, end_date):
    try:
        # code is like 'sz.000001', akshare needs '000001'
        clean_code = code.replace('sz.', '').replace('sh.', '').replace('bj.', '')
        df = ak.stock_zh_a_hist(symbol=clean_code, start_date=start_date, end_date=end_date, adjust="hfq")
        if not df.empty:
            df['code'] = code
            return df
    except Exception as e:
        pass
    return None

def main():
    parquet_path = '.quantbot_data/ashare_daily.parquet'
    if not os.path.exists(parquet_path):
        logging.error("No parquet found to patch.")
        return
        
    logging.info("Loading existing parquet...")
    panel = pd.read_parquet(parquet_path)
    latest_date = panel['date'].max()
    start_date = (latest_date + pd.Timedelta(days=1)).strftime('%Y%m%d')
    end_date = datetime.now().strftime('%Y%m%d')
    
    if start_date > end_date:
        logging.info("Already up to date.")
        return
        
    logging.info(f"Patching data from {start_date} to {end_date}...")
    
    # We only care about the 16 codes from our feedback to be fast, but user said "expand data lake". 
    # Let's get unique codes from recent 10 days to save time, or all codes.
    # To be extremely fast, let's just get the 16 codes plus a few, or just all codes with ThreadPool.
    # Actually, getting 5000 codes via akshare takes ~3 mins with ThreadPool.
    
    codes = panel['code'].unique()
    logging.info(f"Total codes to patch: {len(codes)}")
    
    results = []
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_code, c, start_date, end_date): c for c in codes}
        for i, future in enumerate(as_completed(futures)):
            if i % 500 == 0:
                logging.info(f"Progress: {i}/{len(codes)}")
            res = future.result()
            if res is not None:
                results.append(res)
                
    if results:
        new_data = pd.concat(results, ignore_index=True)
        # Rename columns to match baostock format
        rename_map = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '换手率': 'turn',
            '涨跌幅': 'pctChg'
        }
        new_data = new_data.rename(columns=rename_map)
        new_data['date'] = pd.to_datetime(new_data['date'])
        new_data['tradestatus'] = 1
        new_data['isST'] = 0
        new_data['adjustflag'] = 2
        
        cols = ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', 'isST']
        # Filter only existing columns
        new_data = new_data[[c for c in cols if c in new_data.columns]]
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']:
            if col in new_data.columns:
                new_data[col] = pd.to_numeric(new_data[col], errors='coerce').astype('float32')
                
        logging.info("Merging and saving...")
        final_panel = pd.concat([panel, new_data], ignore_index=True)
        final_panel = final_panel.drop_duplicates(subset=['date', 'code'], keep='last')
        final_panel.to_parquet(parquet_path, index=False, compression='snappy')
        logging.info(f"Data lake expanded successfully to {end_date}.")
    else:
        logging.warning("No new data fetched.")

if __name__ == '__main__':
    main()
