import pandas as pd
from main import extract_market_context, Config
import akshare as ak
df = ak.stock_zh_a_spot_em()
conf = Config()
try:
    df_clean, ok, msg, idx_ret, m_overheated, m_regime, vol_surge = extract_market_context(df, conf)
    
    from main import AdvisoryTracker
    tracker_msgs = AdvisoryTracker.evaluate_and_clean(df_clean)
    if tracker_msgs:
        msg += "\n\n**📢 往期辅助信号跟踪**\n" + "\n".join(tracker_msgs) + "\n"
        
    print("Market msg:")
    print(msg)
except Exception as e:
    import traceback
    traceback.print_exc()
