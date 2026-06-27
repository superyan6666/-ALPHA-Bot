import os
import glob
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_chunks_early():
    chunk_dir = ".quantbot_data/chunks"
    output_dir = ".quantbot_data"
    out_file = os.path.join(output_dir, "ashare_daily.parquet")
    
    chunk_files = glob.glob(os.path.join(chunk_dir, "*.parquet"))
    logging.info(f"Found {len(chunk_files)} chunks. Merging...")
    
    df_list = []
    for chunk_file in chunk_files:
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
        logging.info("Early offline data lake built!")

if __name__ == "__main__":
    merge_chunks_early()
