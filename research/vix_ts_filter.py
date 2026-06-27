import os
import json
import logging
from datetime import datetime
import pandas as pd
import yfinance as yf

log = logging.getLogger('ablog')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'research', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
VIX_CACHE_FILE = os.path.join(DATA_DIR, 'vix_data.csv')
VIX_STATE_FILE = os.path.join(DATA_DIR, 'vix_state.json')

def load_vix_state() -> dict:
    if os.path.exists(VIX_STATE_FILE):
        try:
            with open(VIX_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Failed to read vix_state.json: {e}")
    return {"last_extreme_date": None}

def save_vix_state(state: dict):
    try:
        with open(VIX_STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        log.warning(f"Failed to save vix_state.json: {e}")

def get_vix_ts_signal(now: datetime) -> tuple[float, str, str]:
    """
    Fetch VIX-TS and calculate its rolling 252-day percentile.
    Returns:
        pct: The VIX-TS percentile
        state: 'halt_buy', 'half_pos', 'golden_pit', 'normal'
        msg: Display message for notification
    """
    df = None
    try:
        # Retry logic for yfinance
        for attempt in range(3):
            try:
                # Need enough history for 252d rolling rank + 5d smoothing
                df_raw = yf.download(['^VIX', '^VIX3M'], start='2015-01-01', progress=False)
                if not df_raw.empty:
                    if isinstance(df_raw.columns, pd.MultiIndex):
                        close_vix = df_raw['Close']['^VIX']
                        close_vix3m = df_raw['Close']['^VIX3M']
                        df = pd.DataFrame({'vix': close_vix, 'vix3m': close_vix3m}).reset_index()
                    else:
                        df = df_raw[['Close']].reset_index() 
                    df.columns = ['Date', 'vix', 'vix3m']
                    df.to_csv(VIX_CACHE_FILE, index=False)
                    break
            except Exception as e:
                if attempt == 2:
                    raise e
    except Exception as e:
        log.error(f"Failed to fetch VIX from yfinance: {e}")
        if os.path.exists(VIX_CACHE_FILE):
            df = pd.read_csv(VIX_CACHE_FILE)
            log.info("Using cached VIX data.")
        else:
            return 0.0, 'normal', "⚠️ VIX 数据获取失败且无缓存，VIX-TS 过滤器本周期跳过。"

    if df is None or len(df) < 252:
        return 0.0, 'normal', "⚠️ VIX 历史数据不足，VIX-TS 过滤器本周期跳过。"
        
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df = df.sort_values('Date')
    
    # Calculate VIX-TS
    vix_ratio = df['vix'] / df['vix3m']
    vix_ts_smooth = vix_ratio.rolling(5).mean().dropna()
    
    # Calculate 252-day percentile
    if len(vix_ts_smooth) < 252:
        return 0.0, 'normal', "⚠️ VIX 历史数据不足(清洗后)，VIX-TS 过滤器本周期跳过。"

    percentile_series = vix_ts_smooth.rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    current_pct = percentile_series.iloc[-1]
    current_date = df['Date'].iloc[-1]
    
    if pd.isna(current_pct):
        return 0.0, 'normal', ""

    state_dict = load_vix_state()
    last_extreme_str = state_dict.get('last_extreme_date')
    
    # Logic handling
    state = 'normal'
    msg = f"📊 **VIX-TS 情绪探测**: 当前分位数 **{current_pct:.1%}** "
    
    if current_pct > 0.95:
        state = 'halt_buy'
        msg += "\n🚨 **[全局熔断]** 海外恐慌指数出现极度异常倒挂！暂停一切开仓买入，全防守状态！"
        state_dict['last_extreme_date'] = current_date.isoformat()
        save_vix_state(state_dict)
    elif current_pct > 0.90:
        state = 'half_pos'
        msg += "\n⚠️ **[高压预警]** 海外恐慌指数异常，全局风险极高，强制要求半仓操作，禁止重仓！"
        state_dict['last_extreme_date'] = current_date.isoformat()
        save_vix_state(state_dict)
    elif current_pct < 0.70:
        if last_extreme_str:
            last_extreme_date = datetime.fromisoformat(last_extreme_str).date()
            days_since = (now.date() - last_extreme_date).days
            if 0 < days_since <= 30:
                state = 'golden_pit'
                msg += "\n💎 **[黄金坑确立]** 恐慌情绪已显著退潮！在经历极致宣泄后，当前是介入中长线反弹的绝佳窗口，允许满仓操作！"
                # Clear state so we don't repeat this everyday until 30 days are up
                state_dict['last_extreme_date'] = None
                save_vix_state(state_dict)
            else:
                msg += "(处于安全水域)"
        else:
            msg += "(处于安全水域)"
    else:
        msg += "(处于安全水域)"
        
    return current_pct, state, msg

if __name__ == '__main__':
    # Test script
    logging.basicConfig(level=logging.INFO)
    pct, state, msg = get_vix_ts_signal(datetime.now())
    print(f"Percentile: {pct}")
    print(f"State: {state}")
    print(f"Message: {msg}")
