import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import baostock as bs
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_all_a_share_codes():
    """获取所有A股代码（当前活跃股票）"""
    codes = set()
    logging.info("Querying current A-share stock pool...")
    rs = bs.query_all_stock()
    if rs is not None and rs.error_code == '0':
        while rs.next():
            row = rs.get_row_data()
            code = row[0]
            if code.startswith('sh.6') or code.startswith('sz.0') or code.startswith('sz.3'):
                codes.add(code)
    return list(codes)

def build_data_lake(output_dir=".quantbot_data", max_stocks=None):
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "ashare_daily.parquet")
    
    bs.login()
    logging.info("获取全市场股票列表...")
    codes = get_all_a_share_codes()
    logging.info(f"共获取到 {len(codes)} 只A股标的（含退市）")
    
    if max_stocks:
        codes = codes[:max_stocks]
        logging.info(f"开启测试模式，限制拉取前 {max_stocks} 只标的")
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')
    
    all_data = []
    
    for i, code in enumerate(codes):
        if i % 100 == 0:
            logging.info(f"正在抓取进度: {i}/{len(codes)}")
        
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="2"
        )
        
        if rs is not None and rs.error_code == '0':
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            if data_list:
                df = pd.DataFrame(data_list, columns=rs.fields)
                all_data.append(df)
        else:
            logging.warning(f"获取 {code} 失败: {rs.error_msg if rs else 'None'}")
            
    bs.logout()
    
    if all_data:
        logging.info("开始合并全市场数据...")
        final_df = pd.concat(all_data, ignore_index=True)
        # 优化数据类型以减小存储体积
        final_df['date'] = pd.to_datetime(final_df['date'])
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
