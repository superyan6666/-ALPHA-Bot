import pandas as pd
from datetime import datetime
import json

parquet_path = '.quantbot_data/ashare_daily.parquet'
try:
    df_pq = pd.read_parquet(parquet_path)
    print("Parquet loaded.", df_pq.shape)
    
    # print head columns
    print(df_pq.columns)
except Exception as e:
    print(e)
