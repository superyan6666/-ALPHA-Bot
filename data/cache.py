
"""
本地数据缓存模块
"""
import os
import time
import json
import pickle
import glob
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, Any
import pandas as pd

from core.config import Config

# 日志配置
log = logging.getLogger(__name__)

# 缓存目录配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST_CACHE_DIR = os.path.join(BASE_DIR, "hist_cache")
PAPER_TRADES_PATH = os.path.join(BASE_DIR, "paper_trades.json")
PUSHED_STATE_PATH = os.path.join(BASE_DIR, "pushed_state.json")

# 确保缓存目录存在
os.makedirs(HIST_CACHE_DIR, exist_ok=True)


def _cleanup_hist_cache(max_age_days: int = 30) -> int:
    """
    清理过期的历史缓存文件
    
    Args:
        max_age_days: 最大缓存天数，超过该天数的文件会被清理
        
    Returns:
        清理的文件数量
    """
    cleaned = 0
    now_ts = time.time()
    for f in os.listdir(HIST_CACHE_DIR):
        fpath = os.path.join(HIST_CACHE_DIR, f)
        if os.path.isfile(fpath):
            age_days = (now_ts - os.path.getmtime(fpath)) / 86400
            if age_days > max_age_days:
                try:
                    os.remove(fpath)
                    cleaned += 1
                except Exception:
                    pass
    return cleaned


def _today_str() -> str:
    """获取今日日期字符串"""
    return datetime.now().strftime('%Y-%m-%d')


# 启动时清理过期缓存（仅在缓存文件超过100个时执行）
if os.path.exists(HIST_CACHE_DIR):
    cache_files = [f for f in os.listdir(HIST_CACHE_DIR) 
                  if os.path.isfile(os.path.join(HIST_CACHE_DIR, f))]
    if len(cache_files) > 100:
        cleaned = _cleanup_hist_cache()
        if cleaned > 0:
            log.info(f"🧹 启动时清理过期缓存文件 {cleaned} 个")


def get_score_bucket(score: float) -> str:
    """
    根据分数获取分档标签
    
    Args:
        score: 分数值
        
    Returns:
        分档字符串
    """
    if score >= 85:
        return '85-100'
    elif score >= 80:
        return '80-85'
    elif score >= 75:
        return '75-80'
    elif score >= 70:
        return '70-75'
    return '<70'


class LocalDataLake:
    """本地数据缓存层"""
    def __init__(self, proxy):
        self.proxy = proxy
        self._cache = {}
        self._cache_ttl = {}

    def _get_cache(self, key: str, ttl_seconds: int = 600) -> Optional[Any]:
        """获取缓存值"""
        now = time.time()
        if key in self._cache and now - self._cache_ttl.get(key, 0) < ttl_seconds:
            return self._cache[key]
        
        # 检查文件缓存
        cache_file = os.path.join(HIST_CACHE_DIR, f"{key}.pkl")
        if os.path.exists(cache_file):
            if now - os.path.getmtime(cache_file) < ttl_seconds:
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    pass
        return None

    def _set_cache(self, key: str, value: Any):
        """设置缓存"""
        now = time.time()
        self._cache[key] = value
        self._cache_ttl[key] = now
        
        # 文件缓存
        try:
            cache_file = os.path.join(HIST_CACHE_DIR, f"{key}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception:
            pass

    def fetch_spot(self) -> pd.DataFrame:
        cached = self._get_cache("spot", 300)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached
        val = self.proxy.get_spot()
        self._set_cache("spot", val)
        return val

    def fetch_hist(self, code: str, start: str, end: str) -> pd.DataFrame:
        key = f"hist_{code}_{end}"
        cached = self._get_cache(key, 86400)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached
        val = self.proxy.get_hist(code, start, end)
        self._set_cache(key, val)
        return val

    def fetch_index(self, symbol: str) -> pd.DataFrame:
        key = f"index_{symbol}"
        cached = self._get_cache(key, 3600)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached
        val = self.proxy.get_index(symbol)
        self._set_cache(key, val)
        return val

    def fetch_core_pool(self) -> set:
        key = "core_pool"
        cached = self._get_cache(key, 3600 * 6)
        if isinstance(cached, set):
            return cached
        val = self.proxy.get_core_pool()
        self._set_cache(key, val)
        return val

    def fetch_hot_sectors(self) -> dict:
        cached = self._get_cache("hot_sectors", 300)
        if isinstance(cached, dict):
            return cached
        val = self.proxy.get_hot_sectors()
        self._set_cache("hot_sectors", val)
        return val

    def fetch_sector_rotation(self) -> list:
        """获取行业轮动信号"""
        try:
            import akshare as ak
            from core.config import Cols as C
            
            df = ak.stock_board_industry_index_em()
            if df is not None and not df.empty:
                results = []
                for _, row in df.iterrows():
                    try:
                        symbol = row.get('板块代码', '')
                        if not symbol:
                            continue
                        hist = self.proxy.get_index(symbol)
                        if hist is not None and len(hist) >= 25:
                            ret_20d = (hist['close'].iloc[-1] / hist['close'].iloc[-21] - 1) * 100 if len(hist) >= 21 else 0
                            ret_5d = (hist['close'].iloc[-1] / hist['close'].iloc[-6] - 1) * 100 if len(hist) >= 6 else 0
                            today_pct = hist['close'].pct_change().iloc[-1] * 100 if len(hist) >= 1 else 0
                            
                            signal_type = None
                            if ret_20d > 5 and hist['volume'].iloc[-5:].mean() > hist['volume'].iloc[-20:-5].mean() * 1.2:
                                signal_type = "主升"
                            elif ret_5d < -8 and today_pct > 2:
                                signal_type = "反弹"
                            
                            if signal_type:
                                results.append({
                                    'name': row.get('板块名称', ''),
                                    'pct': row.get('涨跌幅', 0),
                                    'ret_20d': ret_20d,
                                    'signal': signal_type
                                })
                    except Exception:
                        continue
                results.sort(key=lambda x: x['ret_20d'], reverse=True)
                return results[:10]
        except Exception as e:
            log.debug(f"行业轮动分析失败: {e}")
        return []

    def fetch_northbound_flow(self) -> Tuple[float, str]:
        cached = self._get_cache("northbound", 300)
        if isinstance(cached, tuple):
            return cached
        val = self.proxy.get_northbound_flow()
        self._set_cache("northbound", val)
        return val

    def fetch_etf_spot(self) -> pd.DataFrame:
        cached = self._get_cache("etf_spot", 1800)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached
        val = self.proxy.get_etf_spot()
        self._set_cache("etf_spot", val)
        return val

    def fetch_etf_hist(self, symbol: str) -> pd.DataFrame:
        key = f"etf_hist_{symbol}"
        cached = self._get_cache(key, 86400)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached
        val = self.proxy.get_etf_hist(symbol)
        self._set_cache(key, val)
        return val

    def fetch_convertible_bonds(self) -> pd.DataFrame:
        cached = self._get_cache("cb_spot", 600)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached
        val = self.proxy.get_convertible_bonds()
        self._set_cache("cb_spot", val)
        return val

    def fetch_southbound_flow(self) -> Tuple[float, str]:
        cached = self._get_cache("southbound", 300)
        if isinstance(cached, tuple):
            return cached
        val = self.proxy.get_southbound_flow()
        self._set_cache("southbound", val)
        return val

    def fetch_hk_spot(self) -> pd.DataFrame:
        cached = self._get_cache("hk_spot", 300)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached
        val = self.proxy.get_hk_spot()
        self._set_cache("hk_spot", val)
        return val


def load_pushed_state() -> dict:
    """加载推送状态"""
    if os.path.exists(PUSHED_STATE_PATH):
        try:
            with open(PUSHED_STATE_PATH, 'r') as f:
                data = json.load(f)
                if 'date' in data and 'pushed_codes' in data:
                    return {code: data['date'] for code in data.get('pushed_codes', [])}
                return data
        except Exception as e:
            log.warning(f"读取推送记录失败: {e}")
    return {}


def save_pushed_state(pushed_dict: dict) -> None:
    """保存推送状态"""
    from core.config import Config
    
    today_date = datetime.now().date()
    clean_dict = {}
    for code, date_str in pushed_dict.items():
        try:
            expire_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            if (today_date - expire_date).days <= 7:
                clean_dict[code] = date_str
        except Exception:
            pass
    try:
        with open(PUSHED_STATE_PATH, 'w') as f:
            json.dump(clean_dict, f)
    except Exception:
        pass


def is_recently_pushed(code: str, pushed: dict) -> bool:
    """检查是否在最近推送过"""
    if code not in pushed:
        return False
    try:
        expire_date = datetime.strptime(pushed[code], '%Y-%m-%d').date()
        today_date = datetime.now().date()
        return today_date <= expire_date
    except Exception:
        return False


def load_and_update_paper_trades(df_spot: pd.DataFrame) -> Tuple[list, dict]:
    """
    加载并更新模拟交易记录
    
    Args:
        df_spot: 实时行情数据
        
    Returns:
        (交易记录列表, 胜率统计字典)
    """
    from core.config import Cols as C
    
    trades = []
    if os.path.exists(PAPER_TRADES_PATH):
        try:
            with open(PAPER_TRADES_PATH, 'r') as f:
                trades = json.load(f)
        except Exception:
            pass

    spot_dict = {}
    if df_spot is not None and not df_spot.empty:
        spot_dict = df_spot.set_index(C.S_CODE).to_dict('index')
    
    stats = {
        '85-100': {'win': 0, 'total': 0},
        '80-85': {'win': 0, 'total': 0},
        '75-80': {'win': 0, 'total': 0},
        '70-75': {'win': 0, 'total': 0},
        '<70': {'win': 0, 'total': 0}
    }

    active_trades = []
    today_date = datetime.now().date()
    AI_EVO_LOOKBACK_DAYS = 60
    
    for t in trades:
        status = t.get('status', 'PENDING')
        code = t.get('code')
        buy_date = datetime.strptime(t['date'], '%Y-%m-%d').date()
        days_since_buy = (today_date - buy_date).days
        
        # 丢弃超过 90 天的已完结历史记录
        if days_since_buy > 90 and status != 'PENDING':
            continue
        if days_since_buy > 120 and status == 'PENDING':
            continue
        
        if status == 'PENDING' and code in spot_dict:
            row = spot_dict[code]
            high = float(row.get(C.S_HIGH, 0))
            low = float(row.get(C.S_LOW, 0))
            close = float(row.get(C.S_PRICE, 0))

            if high >= t['target']:
                t['status'] = 'WIN'
            elif low <= t['stop']:
                t['status'] = 'LOSS'
            elif days_since_buy > 10:  
                t['status'] = 'TIME_EXIT'

        if t['status'] in ('WIN', 'LOSS'):
            if days_since_buy <= AI_EVO_LOOKBACK_DAYS:
                bucket = t.get('score_bucket', '<70')
                if bucket in stats:
                    stats[bucket]['total'] += 1
                    if t['status'] == 'WIN':
                        stats[bucket]['win'] += 1

        active_trades.append(t)

    return active_trades, stats


def save_paper_trades(trades: list) -> None:
    """保存模拟交易记录"""
    try:
        with open(PAPER_TRADES_PATH, 'w') as f:
            json.dump(trades, f)
    except Exception as e:
        log.error(f"保存模拟盘账本失败: {e}")

