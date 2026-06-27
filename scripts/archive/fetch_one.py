import sys
import os
from datetime import datetime
import pandas as pd
import baostock as bs

os.environ['NO_PROXY'] = '*' 

def fetch_one(code):
    chunk_dir = ".quantbot_data/chunks"
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_file = os.path.join(chunk_dir, f"{code}.parquet")
    
    if os.path.exists(chunk_file):
        return
        
    bs.login()
    start_date = '2019-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    try:
        rs = bs.query_history_k_data_plus(code,
            "date,code,open,high,low,close,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="2")
        
        data_list = []
        if rs is not None and rs.error_code == '0':
            while rs.next():
                data_list.append(rs.get_row_data())
        bs.logout()
        
        if data_list:
            df = pd.DataFrame(data_list, columns=rs.fields)
            df.to_parquet(chunk_file, index=False, compression='snappy')
    except Exception as e:
        bs.logout()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        fetch_one(sys.argv[1])
