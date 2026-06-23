import os
# --- B7 强制防御：绕过失效的本地代理 (如 Clash 10808)，防止单点脆弱 ---
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
os.environ['NO_PROXY'] = '*'

import time
import json
import socket
import random
import logging
from urllib.parse import urlparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Tuple, Callable, Any

import requests
import numpy as np
import pandas as pd

import traceback
from ml_engine import PyTorchDLModel
from feature_engine import build_ml_features
import os

import pytz
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

_GLOBAL_SEMAPHORE = threading.Semaphore(2)
_BS_LOCK = threading.Lock()
_CONSECUTIVE_FAILURES = 0
_MAX_FAILURES = 10

from factors_config import Factor, get_factors_config
# 导入区域结束

class ConfigurationError(ValueError):
    pass

# ═════════════════════════════════════════════════════════════════════════════
# 1. 环境与核心配置 (Environment & Config)
# ═════════════════════════════════════════════════════════════════════════════
class AppConfig:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance._init_env()
        return cls._instance
        
    def _init_env(self):
        # 环境变量仅在启动时全量读取并缓存一次。
        # 此设计保证了在本次运行生命周期内配置的绝对一致性。
        self._env = dict(os.environ)
        
        # 预定义核心配置
        self.IS_MANUAL = self.get('GITHUB_EVENT_NAME') == 'workflow_dispatch'
        self.PUSH_EMPTY = self.get('PUSH_EMPTY_RESULT', 'true').lower() in ('true', '1', 'yes')
        self.LOG_LEVEL = self.get('LOG_LEVEL', 'INFO').upper()
        self.DATA_CACHE_MODE = self.get('DATA_CACHE_MODE', 'online').lower()
        self.OFFLINE_MAX_AGE_DAYS = int(self.get('OFFLINE_MAX_AGE_DAYS', 7))
        self.TUSHARE_TOKEN = self.get('TUSHARE_TOKEN', '').strip()
        self.DINGTALK_WEBHOOK = self.get('DINGTALK_WEBHOOK', '')
        self.FEISHU_WEBHOOK = self.get('FEISHU_WEBHOOK', '')
        self.NOTIFY_SEC_KEYWORD = self.get('NOTIFY_SEC_KEYWORD', 'AI量化').strip()
        self.RUN_MODE = self.get('RUN_MODE', 'normal')
        
        # 校验错配：防止用户将飞书链接错填入钉钉变量，或将钉钉链接错填入飞书变量
        if self.DINGTALK_WEBHOOK and ("feishu.cn" in self.DINGTALK_WEBHOOK or "larksuite.com" in self.DINGTALK_WEBHOOK):
            raise ConfigurationError("ConfigurationError: DINGTALK_WEBHOOK contains Feishu URL. Please check your configuration!")
        if self.FEISHU_WEBHOOK and ("oapi.dingtalk.com" in self.FEISHU_WEBHOOK):
            raise ConfigurationError("ConfigurationError: FEISHU_WEBHOOK contains DingTalk URL. Please check your configuration!")
        
    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._env:
            if default is not None:
                # 在 config 初始化阶段 logger 可能尚未 setup，暂时 print 或者等后续再 log
                print(f"[Config Warning] Missing '{key}', using default: {default}")
            return default
        return self._env[key]

    def print_summary(self, logger: logging.Logger):
        safe_env = {}
        for k, v in self._env.items():
            if 'TOKEN' in k.upper() or 'WEBHOOK' in k.upper() or 'SECRET' in k.upper():
                val_str = str(v)
                safe_env[k] = f"{val_str[:4]}...{val_str[-4:]}" if len(val_str) > 8 else "***"
            else:
                safe_env[k] = v
                
        summary = (
            f"🔧 核心配置已加载:\n"
            f"  - RUN_MODE: {self.RUN_MODE}\n"
            f"  - DATA_CACHE_MODE: {self.DATA_CACHE_MODE}\n"
            f"  - OFFLINE_MAX_AGE_DAYS: {self.OFFLINE_MAX_AGE_DAYS}\n"
            f"  - LOG_LEVEL: {self.LOG_LEVEL}\n"
            f"  - IS_MANUAL: {self.IS_MANUAL}\n"
            f"  - PUSH_EMPTY: {self.PUSH_EMPTY}"
        )
        logger.info(summary)

config = AppConfig()

TZ_BJS       = pytz.timezone('Asia/Shanghai')
STATE_FILE   = 'pushed_state.json'
SPOT_CACHE   = 'spot_cache.pkl'
HIST_CACHE_DIR = 'hist_cache'


IS_MANUAL    = config.IS_MANUAL
PUSH_EMPTY   = config.PUSH_EMPTY

socket.setdefaulttimeout(15.0)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

# 全局网络防拦截伪装 (Global WAF Bypass)
# 强制为指定行情接口的 requests 请求注入现代浏览器的 User-Agent，防止被东方财富/新浪识别为云端爬虫直接掐断连接
_original_request = requests.Session.request

def _patched_request(self, method, url, **kwargs):
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    
    # 仅针对常见行情域名的白名单进行 UA 伪装，避免污染钉钉等原生请求
    whitelist_domains = ('eastmoney.com', 'dfcfw.com', 'sinajs.cn', 'money.163.com', '126.net', 'gtimg.cn', '10jqka.com.cn', 'tushare.pro', 'csindex.com.cn', 'szse.cn')
    needs_patch = any(hostname == d or hostname.endswith('.' + d) for d in whitelist_domains)
    
    if needs_patch:
        headers = kwargs.get('headers', {})
        if not isinstance(headers, dict):
            headers = dict(headers)
        headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
        
        # 针对特定域名强行添加 Referer 以通过防盗链/WAF 检测
        if 'eastmoney.com' in hostname or 'dfcfw.com' in hostname:
            headers['Referer'] = 'https://quote.eastmoney.com/'
        elif 'sina.com.cn' in hostname or 'sinajs.cn' in hostname:
            headers['Referer'] = 'https://finance.sina.com.cn/'
        elif '126.net' in hostname:
            headers['Referer'] = 'http://quotes.money.163.com/'
        elif 'gtimg.cn' in hostname:
            headers['Referer'] = 'https://finance.qq.com/'
        elif 'tushare.pro' in hostname:
            headers['Referer'] = 'https://www.tushare.pro/'
        elif 'csindex.com.cn' in hostname:
            headers['Referer'] = 'https://www.csindex.com.cn/'
        elif 'szse.cn' in hostname:
            headers['Referer'] = 'https://www.szse.cn/'
            
        kwargs['headers'] = headers
    else:
        log.debug(f"[WAF Patch] 原生放行 -> {hostname}")
        
    kwargs['timeout'] = kwargs.get('timeout', 15.0)
    return _original_request(self, method, url, **kwargs)

requests.Session.request = _patched_request

def _today_str() -> str:
    return datetime.now(TZ_BJS).strftime('%Y-%m-%d')


# ═════════════════════════════════════════════════════════════════════════════
# 2. 推送状态与模拟盘自进化系统 (State & AI Evolution)
# ═════════════════════════════════════════════════════════════════════════════
def load_pushed_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
                if 'date' in data and 'pushed_codes' in data:
                    return {code: data['date'] for code in data.get('pushed_codes', [])}
                return data
        except Exception as e:
            log.warning(f"读取推送记录失败: {e}")
    return {}

def save_pushed_state(pushed_dict: dict) -> None:
    today_dt = datetime.now(TZ_BJS).date()
    clean_dict = {}
    for code, date_str in pushed_dict.items():
        try:
            expire_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            if (today_dt - expire_date).days <= 7:
                clean_dict[code] = date_str
        except Exception:
            pass
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(clean_dict, f)
    except Exception:
        pass

def is_recently_pushed(code: str, pushed: dict) -> bool:
    if code not in pushed:
        return False
    try:
        expire_date = datetime.strptime(pushed[code], '%Y-%m-%d').date()
        today_date = datetime.now(TZ_BJS).date()
        return today_date < expire_date
    except Exception:
        return False

# ═════════════════════════════════════════════════════════════════════════════
# 3. 数据契约模型 (Data Schema & Models)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Cols:
    S_PRICE: str = '最新价'
    S_HIGH: str  = '最高'
    S_LOW: str   = '最低'
    S_OPEN: str  = '今开'
    S_PCT: str   = '涨跌幅'
    S_TURN: str  = '换手率'
    S_AMT: str   = '成交额'
    S_VOL: str   = '成交量'
    S_CODE: str  = '代码'
    S_NAME: str  = '名称'
    S_MCAP: str  = '流通市值'
    S_PE: str    = '市盈率-动态'
    S_PB: str    = '市净率'
    S_VR: str    = '量比'
    H_DATE: str  = '日期'
    H_OPEN: str  = '开盘'
    H_CLOSE: str = '收盘'
    H_HIGH: str  = '最高'
    H_LOW: str   = '最低'
    H_VOL: str   = '成交量'
    I_CLOSE: str = 'close'
    B_NAME: str  = '板块名称'
    B_PCT: str   = '涨跌幅'

C = Cols()

class EnvParser:
    @staticmethod
    def get_float(key: str, default: float) -> float:
        val = config.get(key)
        if not val: return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

@dataclass(frozen=True)
class Config:
    MIN_CAP: float       = field(default_factory=lambda: EnvParser.get_float('MIN_CAP', 30e8)) 
    MAX_CAP: float       = field(default_factory=lambda: EnvParser.get_float('MAX_CAP', 2000e8))
    MAX_PRICE: float     = field(default_factory=lambda: EnvParser.get_float('MAX_PRICE', 500.0))  
    MIN_PE: float        = field(default_factory=lambda: EnvParser.get_float('MIN_PE', 0))    
    MAX_PE: float        = field(default_factory=lambda: EnvParser.get_float('MAX_PE', 300))      
    MIN_TURNOVER: float  = field(default_factory=lambda: EnvParser.get_float('MIN_TURNOVER', 0.5))
    MAX_TURNOVER: float  = field(default_factory=lambda: EnvParser.get_float('MAX_TURNOVER', 40.0)) 
    MIN_PCT_CHG: float   = field(default_factory=lambda: EnvParser.get_float('MIN_PCT_CHG', -4.0))  
    MIN_VOL_RATIO: float = field(default_factory=lambda: EnvParser.get_float('MIN_VOL_RATIO', 0.5))  
    MAX_VOL_RATIO: float = field(default_factory=lambda: EnvParser.get_float('MAX_VOL_RATIO', 15.0))
    
    REQUIRED_COLS: tuple = (C.S_PRICE, C.S_OPEN, C.S_HIGH, C.S_LOW, C.S_VOL, C.S_AMT, 
                            C.S_PCT, C.S_CODE, C.S_NAME)
    OPTIONAL_COLS: tuple = (C.S_VR, C.S_TURN, C.S_MCAP, C.S_PE, C.S_PB)
    HIST_COLS: tuple     = (C.H_DATE, C.H_OPEN, C.H_CLOSE, C.H_HIGH, C.H_LOW, C.H_VOL)

@dataclass
class Signal:
    code: str
    name: str
    price: float
    pct_chg: str
    score: int
    level: str
    trigger_time: str 
    reasons: str
    stop_loss: float
    target1: float
    ma10: float
    
    money_risk_msg: str = ""
    tranche_plan_msg: str = ""
    plan_b_msg: str = ""
    hold_period_msg: str = ""

import json

class AdvisoryTracker:
    FILE_PATH = "advisory_tracker.json"
    
    @classmethod
    def load_tracker(cls) -> dict:
        if os.path.exists(cls.FILE_PATH):
            try:
                with open(cls.FILE_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}
        
    @classmethod
    def save_tracker(cls, data: dict):
        with open(cls.FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    @classmethod
    def add_signals(cls, signals: list[Signal], horizon_name: str):
        tracker = cls.load_tracker()
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        max_days = 10
        try:
            h_val = int(horizon_name.replace('T+', ''))
            max_days = h_val * 2
        except:
            pass
            
        for s in signals:
            tracker[s.code] = {
                'name': s.name,
                'entry_date': today_str,
                'target': s.target1,
                'stop': s.stop_loss,
                'horizon': horizon_name,
                'max_days': max_days
            }
        cls.save_tracker(tracker)
        
    @classmethod
    def evaluate_and_clean(cls, current_spot: pd.DataFrame) -> list[str]:
        tracker = cls.load_tracker()
        if not tracker: return []
        
        today_date = datetime.now()
        spot_dict = current_spot.set_index(C.S_CODE)[C.S_PRICE].to_dict()
        
        messages = []
        codes_to_remove = []
        
        for code, info in tracker.items():
            if code not in spot_dict:
                continue
            
            curr_price = float(spot_dict[code])
            target = float(info.get('target', 0))
            stop = float(info.get('stop', 0))
            entry_date = datetime.strptime(info.get('entry_date', today_date.strftime('%Y-%m-%d')), '%Y-%m-%d')
            max_days = int(info.get('max_days', 10))
            
            days_held = (today_date - entry_date).days
            
            if curr_price >= target and target > 0:
                pct = (curr_price / (target / 1.05) - 1) * 100 if target > 0 else 0
                messages.append(f"- **{code} ({info['name']})**: 🟢 **调仓建议**：已达或突破第一目标价 `¥{target}` (现价 `¥{curr_price}`)，建议获利了结或减仓。")
                codes_to_remove.append(code)
            elif curr_price <= stop and stop > 0:
                messages.append(f"- **{code} ({info['name']})**: 🔴 **平仓建议**：已跌破防守线 `¥{stop}` (现价 `¥{curr_price}`)，强烈建议止损离场！")
                codes_to_remove.append(code)
            elif days_held > max_days:
                messages.append(f"- **{code} ({info['name']})**: ⏳ **跟踪到期**：已震荡跟踪 {days_held} 天 ({info['horizon']})，超过参考阈值，主动结束跟踪。")
                codes_to_remove.append(code)
                
        for code in codes_to_remove:
            del tracker[code]
            
        cls.save_tracker(tracker)
        return messages

# ═════════════════════════════════════════════════════════════════════════════
# 4. 专业量化算法核心库 (Quant Algorithms)
# ═════════════════════════════════════════════════════════════════════════════
class MathUtils:
    @staticmethod
    def calc_vcp_quality(df: pd.DataFrame) -> Tuple[float, bool]:
        if len(df) < 31:
            return 0.5, False
            
        segments = []
        for i in [(-31, -21), (-21, -11), (-11, -1)]:
            seg = df.iloc[i[0]:i[1]]
            low = seg[C.H_LOW].min()
            if low > 0:
                amp = (seg[C.H_HIGH].max() - low) / low
                segments.append(amp)
                
        if len(segments) < 3:
            return segments[-1] if segments else 0.5, False
            
        is_vcp = segments[0] > segments[1] > segments[2]
        return segments[-1], is_vcp

    @staticmethod
    def calc_atr_adx(hist: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        high, low, close = hist[C.H_HIGH], hist[C.H_LOW], hist[C.H_CLOSE]
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        up, dn = high.diff(), -low.diff()
        plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
        minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
        plus_di = 100 * pd.Series(plus_dm, index=hist.index).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=hist.index).rolling(period).mean() / atr
        denom = (plus_di + minus_di).replace(0, np.nan)
        dx = ((plus_di - minus_di).abs() / denom * 100)
        return atr, dx.rolling(period).mean()

def is_earnings_danger_zone(now: datetime) -> tuple[bool, str]:
    month = now.month
    DANGER_WINDOWS = [
        (3, 25, 4, 30, "年报/一季报披露末期"),
        (8, 15, 8, 31, "半年报披露末期"),
        (10, 15, 10, 31, "三季报披露末期"),
    ]
    for s_m, s_d, e_m, e_d, label in DANGER_WINDOWS:
        start_dt = now.replace(month=s_m, day=s_d, hour=0, minute=0)
        end_dt = now.replace(month=e_m, day=e_d, hour=23, minute=59)
        if start_dt <= now <= end_dt:
            return True, label
    return False, ""

def calc_target_price(price: float, stop: float, data: dict) -> float:
    risk_amt = price - stop
    if data.get('has_chip_break'):
        max_1y = data.get('max_1y', price * 1.20)
        chip_target = price + (max_1y - price) * 0.5
        min_target = price + risk_amt * 1.5
        return round(max(chip_target, min_target), 2)
    return round(price + risk_amt * 2.0, 2)


# ═════════════════════════════════════════════════════════════════════════════
# 5. 排版与文案渲染器 (Markdown Renderers)
# ═════════════════════════════════════════════════════════════════════════════
def format_money_risk_msg(price: float, stop_loss: float, target1: float) -> str:
    one_hand_cost = price * 100
    budget_per_hand = 10000
    hands = max(1, int(budget_per_hand / one_hand_cost))
    total_cost = hands * one_hand_cost
    
    loss_per_share = price - stop_loss
    total_loss = loss_per_share * hands * 100
    gain_1 = (target1 - price) * hands * 100
    
    ratio = gain_1 / max(total_loss, 1)
    ratio_str = f"{ratio:.1f}"
    
    if ratio >= 2.5:
        evaluation = "🎯 **高容错**：做对一次抵消多次亏损！"
    elif ratio >= 1.5:
        evaluation = "✅ **尚可**：跌势有限，可防守建仓。"
    else:
        evaluation = "⚠️ **需谨慎**：操作要求高，务必**减半仓位**！"
    
    return f"- ⚖️ **盈亏预估**：{hands}手约 `¥{total_cost:.0f}` | 盈亏比 `1:{ratio_str}` ({evaluation}) | 潜在回撤 `-¥{total_loss:.0f}` | 预期 `+¥{gain_1:.0f}`"

def generate_tranche_plan(price: float, score: int, market_ok: bool, market_overheated: bool) -> str:
    if market_overheated:
        return "🛑 **【市场情绪警报】当前大盘极度过热！随时可能面临获利盘踩踏，强烈建议暂停买入或保持空仓观望！**"
        
    base_pct = 30 if score >= 85 else 20 if score >= 70 else 10
    if not market_ok:
        base_pct = base_pct // 2
        
    t1 = max(1, base_pct // 3)
    t2 = max(1, base_pct // 3)
    t3 = max(1, base_pct - t1 - t2)
    
    lower_bound = round(price * 0.985, 2)
    upper_bound = round(price * 1.005, 2)
    add_price   = round(price * 1.025, 2)
    stop_add    = round(price * 1.05,  2)
    
    return f"- 🎯 **分批建仓**：支撑区 `¥{lower_bound}-¥{upper_bound}`({t1}%) ➡️ 站稳 `¥{add_price}`({t2}%) ➡️ 突破 `¥{stop_add}`({t3}%)"

def generate_plan_b(price: float, stop_loss: float, ma20: float) -> str:
    normal_shake = round(price * 0.97, 2)  
    normal_shake = max(normal_shake, stop_loss + 0.01)
    
    return f"- 🛡️ **防守红线**：未破 `¥{normal_shake:.2f}` 为正常洗盘；若有效跌破 `¥{stop_loss:.2f}`，**必须无条件止损**。"

def generate_hold_period(adx: float, price_pct: float, has_chip_break: bool) -> str:
    if price_pct < 0.35 and adx < 20:
        return "- **⏳ 持股参考**：🐢 **【底部潜伏型】(1~3个月)**，属于左侧蓄势，建议保持耐心不宜频繁操作。"
    elif adx > 25 or has_chip_break:
        return "- **⏳ 持股参考**：🐎 **【右侧趋势型】(3~10天)**，当前正处爆发期，建议见好就收，避免过度贪婪。"
    else:
        return "- **⏳ 持股参考**：🐕 **【稳健震荡型】(2~4周)**，建议等待趋势明朗。"


# ═════════════════════════════════════════════════════════════════════════════
# 6. 统一数据代理与本地数据湖 (Data Proxy & Data Lake)
# ═════════════════════════════════════════════════════════════════════════════
import glob
import pickle
import akshare as ak

try:
    import baostock as bs
except ImportError:
    bs = None

try:
    import efinance as ef
except ImportError:
    ef = None

def retry(times=4, delay=2, exceptions=(Exception,)):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt < times - 1:
                        log.debug(f"抓取受挫，将在 {delay * (2 ** attempt)}s 后重试: {e}")
                        time.sleep(delay * (2 ** attempt))
                    else:
                        raise
        return wrapper
    return decorator

class DataProxy:
    """数据获取多源路由层 (Fallback Waterfall)"""
    def __init__(self):
        self.bs_logged_in = False
        self.ts_pro = None
        ts_token = config.TUSHARE_TOKEN
        if ts_token:
            try:
                import tushare as ts
                ts.set_token(ts_token)
                self.ts_pro = ts.pro_api()
                log.info("🌟 成功挂载 Tushare Pro 机构级数据核心")
            except Exception as e:
                log.warning(f"Tushare 初始化失败: {e}")

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        if self.bs_logged_in and bs is not None:
            try: 
                bs.logout()
            except: pass
            self.bs_logged_in = False

    def _login_baostock(self):
        if bs is not None and not self.bs_logged_in:
            bs.login()
            self.bs_logged_in = True

    # ---- [1. Historical Data] ----
    def _fetch_hist_tushare(self, code, start, end):
        if not self.ts_pro: return None
        try:
            import tushare as ts
            ts_code = f"{code}.SH" if code.startswith(('6', '5')) else f"{code}.SZ"
            start_fmt = f"{start[:4]}{start[4:6]}{start[6:]}"
            end_fmt = f"{end[:4]}{end[4:6]}{end[6:]}"
            asset_type = 'FD' if code.startswith(('51', '15', '588', '56')) else 'E'
            df_adj = ts.pro_bar(ts_code=ts_code, api=self.ts_pro, start_date=start_fmt, end_date=end_fmt, adj='qfq', asset=asset_type)
            if df_adj is None or df_adj.empty: return None
            
            df_adj = df_adj.rename(columns={'trade_date': C.H_DATE, 'open': C.H_OPEN, 'close': C.H_CLOSE, 'high': C.H_HIGH, 'low': C.H_LOW, 'vol': C.H_VOL})
            df_adj[C.H_DATE] = pd.to_datetime(df_adj[C.H_DATE]).dt.strftime('%Y-%m-%d')
            df_adj = df_adj.sort_values(C.H_DATE).reset_index(drop=True)
            for col in [C.H_OPEN, C.H_CLOSE, C.H_HIGH, C.H_LOW, C.H_VOL]:
                df_adj[col] = pd.to_numeric(df_adj[col], errors='coerce')
            return df_adj[list(Config.HIST_COLS)]
        except Exception as e:
            log.debug(f"[Tier 1 Tushare] 获取历史失败: {e}")
            return None

    def _get_tushare_fundamentals_df(self) -> pd.DataFrame:
        if not self.ts_pro: return pd.DataFrame()
        try:
            for days_back in range(1, 10):
                trade_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
                df = self.ts_pro.daily_basic(trade_date=trade_date)
                if df is not None and not df.empty:
                    df[C.S_CODE] = df['ts_code'].str.slice(0, 6)
                    df[C.S_TURN] = df.get('turnover_rate_f', df.get('turnover_rate', 2.0)).astype(float)
                    df[C.S_VR] = df.get('volume_ratio', 1.0).astype(float)
                    df[C.S_PE] = df.get('pe_ttm', -1.0).astype(float)
                    df[C.S_PB] = df.get('pb', 2.0).astype(float)
                    df[C.S_MCAP] = df.get('circ_mv', 0.0).astype(float) * 10000
                    return df[[C.S_CODE, C.S_TURN, C.S_VR, C.S_PE, C.S_PB, C.S_MCAP]]
        except Exception as e:
            log.debug(f"Tushare 向量化获取基本面失败: {e}")
        return pd.DataFrame()

    def _fetch_hist_baostock(self, code, start, end):
        if bs is None: return None
        with _BS_LOCK:
            self._login_baostock()
            try:
                prefix = 'sh.' if code.startswith('6') else 'sz.'
                start_fmt = f"{start[:4]}-{start[4:6]}-{start[6:]}"
                end_fmt = f"{end[:4]}-{end[4:6]}-{end[6:]}"
                
                rs = bs.query_history_k_data_plus(
                    prefix + code,
                    "date,open,close,high,low,volume,amount",
                    start_date=start_fmt, end_date=end_fmt,
                    frequency="d", adjustflag="2"
                )
                
                if rs is None or rs.error_code != '0':
                    return None
                    
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                if not data_list: return None
                
                df = pd.DataFrame(data_list, columns=rs.fields)
                df = df.rename(columns={'date': C.H_DATE, 'open': C.H_OPEN, 'close': C.H_CLOSE, 'high': C.H_HIGH, 'low': C.H_LOW, 'volume': C.H_VOL})
                for col in [C.H_OPEN, C.H_CLOSE, C.H_HIGH, C.H_LOW, C.H_VOL]:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                return df[list(Config.HIST_COLS)]
            except Exception as e:
                log.debug(f"[Tier 1 BaoStock] 获取历史失败: {e}")
                return None

    def _fetch_hist_akshare(self, code, start, end):
        global _CONSECUTIVE_FAILURES
        if _CONSECUTIVE_FAILURES >= _MAX_FAILURES:
            log.error(f"🔥 [RateLimit_CIRCUIT] 连续 {_CONSECUTIVE_FAILURES} 次获取历史失败，熔断机制触发，冷却10秒后重置。")
            time.sleep(10)
            _CONSECUTIVE_FAILURES = 0
            return None
        for attempt in range(3):
            with _GLOBAL_SEMAPHORE:
                try:
                    # 引入随机微型延迟 (0.3s ~ 0.8s) 以平滑并发请求，避免触发 WAF 行情接口封锁限制
                    time.sleep(random.uniform(0.3, 0.8))
                    df = ak.stock_zh_a_hist(symbol=code, period='daily', start_date=start, end_date=end, adjust='qfq')
                    if df is not None and not df.empty:
                        _CONSECUTIVE_FAILURES = 0
                        return df[list(Config.HIST_COLS)].copy()
                except Exception as e:
                    try:
                        df = ak.stock_zh_a_hist_tx(symbol=code, start_date=start, end_date=end, adjust='qfq')
                        if df is not None and not df.empty:
                            col_map = {'日期': C.H_DATE, '开盘': C.H_OPEN, '收盘': C.H_CLOSE, '最高': C.H_HIGH, '最低': C.H_LOW, '成交量': C.H_VOL}
                            df = df.rename(columns=col_map)
                            _CONSECUTIVE_FAILURES = 0
                            return df[list(Config.HIST_COLS)].copy()
                    except Exception:
                        pass
                    
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                    log.warning(f"⚠️ [Akshare] 获取 {code} 历史失败, 冷却 {backoff:.1f}s 后重试...")
                    time.sleep(backoff)
        _CONSECUTIVE_FAILURES += 1
        raise ValueError(f'akshare history empty for {code}')
            
    def get_hist(self, code, start, end) -> pd.DataFrame:
        df = self._fetch_hist_baostock(code, start, end)
        if df is not None: return df
        df = self._fetch_hist_tushare(code, start, end)
        if df is not None: return df
        return self._fetch_hist_akshare(code, start, end)

    # ---- [2. Spot Data (实时横截面)] ----
    def _fetch_spot_qmt(self):
        return None

    def _fetch_spot_efinance(self):
        if ef is None: return None
        try:
            df = ef.stock.get_realtime_quotes()
            if df is not None and not df.empty:
                rename_map = {'代码': C.S_CODE, '名称': C.S_NAME, '最新价': C.S_PRICE,
                              '涨跌幅': C.S_PCT, '今开': C.S_OPEN, '最高': C.S_HIGH,
                              '最低': C.S_LOW, '成交量': C.S_VOL, '成交额': C.S_AMT,
                              '换手率': C.S_TURN, '市盈率-动态': C.S_PE, '市净率': C.S_PB, '量比': C.S_VR}
                df = df.rename(columns=rename_map)
                return df
        except Exception as e:
            log.debug(f"[Tier 2 efinance] 获取实时行情失败: {e}")
        return None

    @retry(times=3, delay=5)
    def _fetch_spot_akshare(self):
        try:
            time.sleep(random.uniform(1.0, 3.0))
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                return df
        except Exception as e:
            log.warning(f"行情主接口(EastMoney)异常: {e}，将触发级联降级...")
        return None

    def _fetch_spot_tencent(self) -> pd.DataFrame:
        log.info("🚀 启动备用源: 腾讯原生 API (gtimg)...")
        pool = list(self.get_core_pool())
        if not pool: return None
        
        # 将 pool 切分为每批 50 个以防止 URL 过长
        batch_size = 50
        results = []
        for i in range(0, len(pool), batch_size):
            batch_codes = pool[i:i+batch_size]
            formatted_codes = []
            for c in batch_codes:
                if c.startswith('6'): formatted_codes.append(f'sh{c}')
                elif c.startswith('0') or c.startswith('3'): formatted_codes.append(f'sz{c}')
                elif c.startswith('8') or c.startswith('4'): formatted_codes.append(f'bj{c}')
                else: formatted_codes.append(f'sh{c}')
                
            url = f"http://qt.gtimg.cn/q={','.join(formatted_codes)}"
            try:
                resp = requests.get(url, timeout=5)
                resp.encoding = 'gbk'
                lines = resp.text.strip().split('\n')
                for line in lines:
                    if '=' not in line: continue
                    var, data = line.split('=', 1)
                    parts = data.replace('"', '').replace(';', '').split('~')
                    if len(parts) < 45: continue
                    
                    parsed = {
                        C.S_NAME: parts[1],
                        C.S_CODE: parts[2],
                        C.S_PRICE: float(parts[3]) if parts[3] else None,       # 价格=3
                        C.S_OPEN: float(parts[5]) if parts[5] else None,        # 开盘=5
                        C.S_HIGH: float(parts[33]) if parts[33] else None,      # 最高=33
                        C.S_LOW: float(parts[34]) if parts[34] else None,       # 最低=34
                        C.S_PCT: float(parts[32]) if parts[32] else None,       # 涨跌幅=32
                        C.S_VOL: float(parts[36]) if parts[36] else None,       # 成交量(手)=36
                        C.S_AMT: float(parts[37]) * 10000 if parts[37] else None, # 成交额(万)=37
                        C.S_TURN: float(parts[38]) if parts[38] else 2.0,       # 换手率=38
                        C.S_PE: float(parts[39]) if parts[39] else -1.0,        # 市盈率=39
                        C.S_PB: float(parts[46]) if len(parts)>46 and parts[46] else 2.0, # 市净率=46
                        C.S_VR: float(parts[49]) if len(parts)>49 and parts[49] else 1.0, # 量比=49
                        'source_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    results.append(parsed)
            except Exception as e:
                log.warning(f"腾讯源批量获取异常: {e}")
                
        df = pd.DataFrame(results)
        if df.empty: return None
        
        # 腾讯缺少市值信息，用 tushare fundamentals 补全
        funds_df = self._get_tushare_fundamentals_df()
        if not funds_df.empty:
            df = df.set_index(C.S_CODE).combine_first(funds_df.set_index(C.S_CODE)).reset_index()
            df[C.S_MCAP] = df[C.S_MCAP].fillna(100e8)
            log.info("💎 已通过 Tushare 成功向量化补全腾讯备用源缺失的市值数据。")
        else:
            if C.S_MCAP not in df.columns:
                df[C.S_MCAP] = 100e8
        return df

    def _fetch_spot_netease(self) -> pd.DataFrame:
        log.warning("⚠️ 启动三级备用源: 网易原生 API (126.net)...")
        pool = list(self.get_core_pool())
        if not pool: return None
        
        batch_size = 50
        results = []
        for i in range(0, len(pool), batch_size):
            batch_codes = pool[i:i+batch_size]
            formatted_codes = []
            for c in batch_codes:
                if c.startswith('6'): formatted_codes.append(f'0{c}')
                elif c.startswith('0') or c.startswith('3'): formatted_codes.append(f'1{c}')
                elif c.startswith('8') or c.startswith('4'): formatted_codes.append(f'1{c}')
                else: formatted_codes.append(f'0{c}')
                
            url = f"http://api.money.126.net/data/feed/{','.join(formatted_codes)},money.api"
            try:
                resp = requests.get(url, timeout=5)
                text = resp.text
                start = text.find('(')
                end = text.rfind(')')
                if start != -1 and end != -1:
                    data = json.loads(text[start+1:end])
                    for k, v in data.items():
                        parsed = {
                            C.S_NAME: v.get('name', ''),
                            C.S_CODE: v.get('symbol', ''),
                            C.S_PRICE: v.get('price') if v.get('price') is not None else float('nan'),
                            C.S_OPEN: v.get('open') if v.get('open') is not None else float('nan'),
                            C.S_HIGH: v.get('high') if v.get('high') is not None else float('nan'),
                            C.S_LOW: v.get('low') if v.get('low') is not None else float('nan'),
                            C.S_PCT: v.get('percent') * 100 if v.get('percent') is not None else float('nan'),
                            C.S_VOL: v.get('volume') if v.get('volume') is not None else float('nan'),
                            C.S_AMT: v.get('turnover') if v.get('turnover') is not None else float('nan'),
                            C.S_TURN: v.get('turnoverrate') if v.get('turnoverrate') is not None else 2.0,
                            'source_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        results.append(parsed)
            except Exception as e:
                log.warning(f"网易源批量获取异常: {e}")
                
        df = pd.DataFrame(results)
        if df.empty: return None
        
        funds_df = self._get_tushare_fundamentals_df()
        if not funds_df.empty:
            df = df.set_index(C.S_CODE).combine_first(funds_df.set_index(C.S_CODE)).reset_index()
            df[C.S_MCAP] = df[C.S_MCAP].fillna(100e8)
            df[C.S_PE] = df[C.S_PE].fillna(-1.0)
            df[C.S_PB] = df[C.S_PB].fillna(2.0)
            df[C.S_VR] = df[C.S_VR].fillna(1.0)
            log.info("💎 已通过 Tushare 成功补全网易备用源缺失的基础特征。")
        else:
            fallback_defaults = {C.S_MCAP: 100e8, C.S_PE: -1.0, C.S_PB: 2.0, C.S_VR: 1.0}
            for col, val in fallback_defaults.items():
                if col not in df.columns: df[col] = val
        return df

    def _fetch_spot_tushare_fallback(self) -> pd.DataFrame:
        """极寒时刻的终极兜底：当全市场实时接口死掉，用日频历史伪装截面"""
        if not self.ts_pro:
            raise ValueError("Tushare 未配置，终极兜底失败！网络严重受阻。")
        log.warning("⚠️ [FALLBACK] 所有实时数据源失效，正尝试使用 Tushare 日终历史充当伪装截面数据！")
        
        try:
            for days_back in range(0, 10):
                trade_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
                df = self.ts_pro.daily(trade_date=trade_date)
                if df is not None and not df.empty:
                    df_basic = self.ts_pro.daily_basic(trade_date=trade_date)
                    if df_basic is not None and not df_basic.empty:
                        df = pd.merge(df, df_basic, on='ts_code', how='left')
                    
                    df[C.S_CODE] = df['ts_code'].str.slice(0, 6)
                    df[C.S_PRICE] = df['close']
                    df[C.S_OPEN] = df['open']
                    df[C.S_HIGH] = df['high']
                    df[C.S_LOW] = df['low']
                    df[C.S_VOL] = df['vol']
                    df[C.S_AMT] = df['amount'] * 1000
                    df[C.S_PCT] = 0.0 # 强制置零，防止使用T-1的涨跌幅误导下游策略
                    df[C.S_TURN] = df.get('turnover_rate', df.get('turnover_rate_f', 2.0))
                    df[C.S_PE] = df.get('pe_ttm', -1.0)
                    df[C.S_PB] = df.get('pb', 2.0)
                    df[C.S_MCAP] = df.get('circ_mv', 0.0) * 10000
                    df[C.S_VR] = df.get('volume_ratio', 1.0)
                    df[C.S_NAME] = "TS_" + df[C.S_CODE]
                    df['DATA_MODE'] = 'T+1_FALLBACK'
                    log.warning(f"⚠️ [FALLBACK_SUCCESS] 成功拉取 {trade_date} Tushare数据作为伪装截面。注意时效性！")
                    return df
            raise ValueError("Tushare returned empty daily data across 10 days.")
        except Exception as e:
            log.error(f"❌ [FATAL] Tushare 终极兜底方案也失败，彻底断网: {e}")
            raise e

    def get_spot(self) -> pd.DataFrame:
        df = self._fetch_spot_qmt()
        if df is None: df = self._fetch_spot_efinance()
        
        if df is None:
            try:
                df = self._fetch_spot_akshare()
            except Exception as e:
                log.debug(f"akshare spot failed: {e}")
                
        if df is None:
            try:
                df = self._fetch_spot_tencent()
            except Exception as e:
                log.debug(f"tencent spot failed: {e}")
                
        if df is None:
            try:
                # df = self._fetch_spot_netease() # [B10] 禁用已失效的网易接口，防止无意义的超时堵塞
                df = None
            except Exception as e:
                log.debug(f"netease spot failed: {e}")
                
        if df is None:
            df = self._fetch_spot_tushare_fallback()

        if df is not None and not df.empty:
            col_map = {}
            for c in df.columns:
                if c in ['股票代码', 'code', 'ts_code', 'symbol', 'f12']: col_map[c] = C.S_CODE
                if c in ['股票名称', 'name', 'f14']: col_map[c] = C.S_NAME
            if col_map:
                df = df.rename(columns=col_map)
                
            if C.S_CODE in df.columns:
                df[C.S_CODE] = df[C.S_CODE].astype(str).str.zfill(6)
                df = df.drop_duplicates(subset=[C.S_CODE], keep='first')
                
            # 全局兜底强制填补缺失的财务列，防止下游 KeyError
            fallback_defaults = {C.S_MCAP: 100e8, C.S_PE: -1.0, C.S_PB: 2.0, C.S_VR: 1.0, C.S_TURN: 2.0, C.S_PCT: 0.0, C.S_PRICE: 0.0}
            for col, val in fallback_defaults.items():
                if col not in df.columns:
                    df[col] = val
        return df

    # ---- [3. Index & Context] ----
    @retry(times=4, delay=2)
    def get_index(self, symbol: str) -> pd.DataFrame:
        try:
            df = ak.stock_zh_index_daily_tx(symbol=symbol)
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
                return df
        except Exception as e:
            log.warning(f"腾讯指数接口波动 ({symbol}): {e}，切换东方财富源...")
        
        try:
            df = ak.stock_zh_index_daily_em(symbol=symbol)
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
                return df
        except Exception as e:
            log.warning(f"东方财富指数接口也波动 ({symbol}): {e}，切换 Baostock 源...")
            
        if bs is not None:
            self._login_baostock()
            bs_symbol = 'sh.' + symbol[-6:] if 'sh' in symbol.lower() else 'sz.' + symbol[-6:]
            start_fmt = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            def _do_index_query():
                return bs.query_history_k_data_plus(bs_symbol, "date,open,close,high,low,volume", start_date=start_fmt, frequency="d")
                
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_index_query)
                rs = future.result(timeout=10)
                
            if rs is None or rs.error_code != '0':
                return None
                
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            if data_list:
                df = pd.DataFrame(data_list, columns=rs.fields)
                df = df.rename(columns={'date': 'date', 'open': 'open', 'close': 'close', 'high': 'high', 'low': 'low', 'volume': 'volume'})
                for col in ['open', 'close', 'high', 'low', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
        raise ValueError(f'index_empty_{symbol}')

    @retry(times=3, delay=2)
    def get_core_pool(self) -> set:
        pool = set()
        
        # 1. 优先使用 Akshare 常规源与 CSIndex 降级源
        for idx in ["000300", "000905", "000852", "399006"]:
            try:
                df = None
                try:
                    df = ak.index_stock_cons(symbol=idx)
                except Exception as ex_normal:
                    log.debug(f"Akshare 常规 index_stock_cons 接口失效 ({idx}): {ex_normal}，尝试中证/深证备用源...")
                    try:
                        df = ak.index_stock_cons_csindex(symbol=idx)
                    except Exception as ex_cs:
                        log.debug(f"Akshare csindex 降级接口也失效 ({idx}): {ex_cs}")
                
                if df is not None and not df.empty:
                    col = next((c for c in df.columns if '代码' in c), None)
                    if col: 
                        pool.update(df[col].astype(str).str.zfill(6).tolist())
            except Exception as e:
                log.warning(f"Akshare 获取指数 {idx} 成分股失败: {e}")
                
        if pool: return pool

        # 2. 降级使用 Tushare
        if self.ts_pro:
            try:
                for idx in ["399300.SZ", "000905.SH", "000852.SH", "399006.SZ"]:
                    df = self.ts_pro.index_weight(index_code=idx, start_date=(datetime.now() - timedelta(days=60)).strftime('%Y%m%d'))
                    if not df.empty:
                        pool.update(df['con_code'].str.slice(0, 6).tolist())
                if pool: return pool
            except Exception as e:
                log.warning(f"Tushare 获取成分股失败: {e}")
                
        # 3. 终极兜底：静态核心50池，防止全市场扫描导致性能爆炸
        log.warning("⚠️ 核心池所有动态接口失效，已降级为静态核心50股票池！")
        return {"600519", "601318", "600036", "601166", "000858", "002594", "000333", "600276", "601012", "601899", "601888", "603288", "002415", "600030", "600887", "600900", "000568", "002304", "002714", "300750", "300760", "600438", "601398", "601288", "601939", "601988", "600000", "601328", "601138", "002475", "000001", "000002", "300015", "300059", "600104", "600690", "601668", "601816", "601857", "601088", "600028", "601066", "600585", "601111", "000157", "000651", "002142", "002271", "300122", "600809"}

    @retry(times=2, delay=2)
    def get_hot_sectors(self) -> dict:
        hot_stocks = {}
        try:
            df = ak.stock_board_industry_name_em()
            if df is not None and not df.empty:
                top_sectors = df.nlargest(5, '涨跌幅')['板块名称'].tolist()
                log.info(f"🌋 (东方财富) 今日领涨主线板块抓取成功: {', '.join(top_sectors)}")
                for sector in top_sectors:
                    try:
                        time.sleep(0.5) 
                        cons = ak.stock_board_industry_cons_em(symbol=sector)
                        if cons is not None and not cons.empty:
                            col = next((c for c in cons.columns if '代码' in c), None)
                            if col:
                                for code in cons[col].astype(str).str.zfill(6).tolist():
                                    hot_stocks[code] = sector
                    except Exception: pass
                if hot_stocks: return hot_stocks
        except Exception as e:
            log.warning(f"东方财富主线板块榜单获取失败(可能被云端拦截): {e}，正在切换同花顺备用源...")

        try:
            df = ak.stock_board_industry_name_ths()
            if df is not None and not df.empty:
                name_col = next((c for c in df.columns if '板块' in c or 'name' in c.lower()), None)
                pct_col = next((c for c in df.columns if '涨跌' in c or 'pct' in c.lower()), None)
                if name_col and pct_col:
                    df[pct_col] = pd.to_numeric(df[pct_col], errors='coerce')
                    top_sectors = df.nlargest(5, pct_col)[name_col].tolist()
                    log.info(f"🌋 (同花顺) 今日领涨主线板块抓取成功: {', '.join(top_sectors)}")
                    for sector in top_sectors:
                        try:
                            time.sleep(0.5)
                            cons = ak.stock_board_industry_cons_ths(symbol=sector)
                            if cons is not None and not cons.empty:
                                col = next((c for c in cons.columns if '代码' in c or 'code' in c.lower()), None)
                                if col:
                                    for code in cons[col].astype(str).str.zfill(6).tolist():
                                        hot_stocks[code] = sector
                        except Exception as e:
                            log.debug(f"同花顺获取板块【{sector}】成分股跳过: {e}")
                    if hot_stocks: return hot_stocks
        except Exception as e:
            log.warning(f"同花顺板块备用源获取也失败: {e}")
            
        return hot_stocks

    @retry(times=2, delay=2)
    def get_northbound_flow(self) -> tuple[float, str]:
        try:
            df = ak.stock_em_hsgt_north_net_flow_in(indicator="沪深港通")
            if df is not None and not df.empty:
                col = 'value' if 'value' in df.columns else df.columns[-1]
                today_flow = float(df.iloc[-1][col]) / 1e8
                if today_flow > 30: return today_flow, f"\n- 🌊 **聪明钱流向**：北水大举流入 **+{today_flow:.0f}亿**"
                elif today_flow < -30: return today_flow, f"\n- ❄️ **聪明钱流向**：北水大幅流出 **{today_flow:.0f}亿**"
                else: return today_flow, f"\n- ⚖️ **聪明钱流向**：北向资金温和 (**{today_flow:+.0f}亿**)"
        except Exception: pass
        return 0.0, ""

class LocalDataLake:
    """本地数据湖缓存拦截层"""
    def __init__(self, proxy: DataProxy):
        self.proxy = proxy
        self.cache_dir = HIST_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self._offline_summary_printed = False

    def _print_offline_summary(self):
        if self._offline_summary_printed: return
        self._offline_summary_printed = True
        log.warning(f"📴 [OFFLINE_MODE] 已开启无网络强制缓存模式！")
        log.warning(f"📴 [OFFLINE_MODE] 超过 {config.OFFLINE_MAX_AGE_DAYS} 天的缓存将被拒绝。若需强制刷新请删除 hist_cache 目录。")
        files = glob.glob(os.path.join(self.cache_dir, "*.pkl"))
        if not files:
            log.warning("📴 [OFFLINE_MODE] 警告：本地无任何缓存文件！后续请求可能抛出异常。")
            return
        log.warning("📴 [OFFLINE_MODE] 本地缓存资产库摘要:")
        for f in sorted(files, key=os.path.getmtime, reverse=True)[:10]:
            mtime = os.path.getmtime(f)
            time_src = "文件系统"
            try:
                with open(f, 'rb') as pf:
                    payload = pickle.load(pf)
                if isinstance(payload, dict) and 'created_at' in payload:
                    mtime = payload['created_at']
                    time_src = "内部烙印"
            except Exception:
                pass
            
            time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            log.warning(f"  - {os.path.basename(f)} ({time_src}时间: {time_str})")
        if len(files) > 10: log.warning(f"  ...及其他 {len(files)-10} 个缓存文件。")

    def _get_cache(self, key: str, ttl_seconds: int):
        parquet_filename = os.path.join(self.cache_dir, f"{key}.parquet")
        filename = os.path.join(self.cache_dir, f"{key}.pkl")
        
        target_file = None
        is_parquet = False
        if os.path.exists(parquet_filename):
            target_file = parquet_filename
            is_parquet = True
        elif os.path.exists(filename):
            target_file = filename
            
        if not target_file: return None
        
        try:
            mtime = os.path.getmtime(target_file)
            time_src = "文件系统"
            
            if is_parquet:
                data = pd.read_parquet(target_file)
            else:
                with open(target_file, 'rb') as f:
                    payload = pickle.load(f)
                
                if isinstance(payload, dict) and 'created_at' in payload and 'data' in payload:
                    mtime = payload['created_at']
                    data = payload['data']
                    time_src = "内部烙印"
                else:
                    data = payload

            age_seconds = time.time() - mtime
            age_days = age_seconds / 86400.0

            if config.DATA_CACHE_MODE == 'offline':
                self._print_offline_summary()
                if age_days > config.OFFLINE_MAX_AGE_DAYS:
                    log.warning(f"⚠️ [CACHE_REJECT] {key} 最新离线缓存 ({time_src}) 已超过 {config.OFFLINE_MAX_AGE_DAYS} 天，拒绝使用。")
                    return None
                return data
                
            if age_seconds < ttl_seconds:
                return data
        except Exception as e:
            log.debug(f"读取缓存异常 {key}: {e}")
            
        return None

    def _set_cache(self, key: str, data):
        if data is None: return
        if isinstance(data, (pd.DataFrame, pd.Series)) and data.empty: return
        
        timestamp = int(time.time())
        try:
            if isinstance(data, pd.DataFrame):
                filename = os.path.join(self.cache_dir, f"{key}.parquet")
                # Ensure column names are strings for Parquet compatibility
                data.columns = data.columns.astype(str)
                data.to_parquet(filename, index=True)
                os.utime(filename, (timestamp, timestamp))
            else:
                filename = os.path.join(self.cache_dir, f"{key}.pkl")
                payload = {
                    'created_at': timestamp,
                    'data': data
                }
                with open(filename, 'wb') as f:
                    pickle.dump(payload, f)
        except Exception as e:
            log.debug(f"缓存写入失败 {key}: {e}")
            
        pattern = os.path.join(self.cache_dir, f"{key}_*.pkl")
        for old_file in glob.glob(pattern):
            try: os.remove(old_file)
            except Exception: pass

    def fetch_spot(self) -> pd.DataFrame:
        now = datetime.now(TZ_BJS)
        ttl = 3600 if (now.hour > 15 or (now.hour == 15 and now.minute >= 30)) else 300 # 盘后延长TTL避免无意义请求
        cached = self._get_cache("spot", ttl)
        if cached is not None:
            if config.DATA_CACHE_MODE != 'offline': log.info(f"📦 命中 spot 本地实时缓存(当前时效 {ttl} 秒)...")
            return cached
        df = self.proxy.get_spot()
        self._set_cache("spot", df)
        return df

    def fetch_hist(self, code: str, start: str, end: str) -> Optional[pd.DataFrame]:
        key = f"hist_{code}_{end}"
        cached = self._get_cache(key, 86400) # 日线数据1天时效
        if cached is not None: return cached
        df = self.proxy.get_hist(code, start, end)
        self._set_cache(key, df)
        return df

    def fetch_index(self, symbol: str):
        key = f"index_{symbol}"
        cached = self._get_cache(key, 43200) # 12小时时效
        if cached is not None: return cached
        df = self.proxy.get_index(symbol)
        self._set_cache(key, df)
        return df
        
    def fetch_core_pool(self):
        cached = self._get_cache("core_pool", 86400)
        if cached is not None: 
            if isinstance(cached, set): return cached
        pool = self.proxy.get_core_pool()
        self._set_cache("core_pool", pool)
        return pool
        
    def fetch_hot_sectors(self):
        cached = self._get_cache("hot_sectors", 300) # 5分钟时效
        if cached is not None: 
            if isinstance(cached, dict): return cached
        val = self.proxy.get_hot_sectors()
        self._set_cache("hot_sectors", val)
        return val
        
    def fetch_northbound_flow(self):
        cached = self._get_cache("northbound", 300) # 5分钟时效
        if cached is not None: 
            if isinstance(cached, tuple): return cached
        val = self.proxy.get_northbound_flow()
        self._set_cache("northbound", val)
        return val

# ── 实例化全局单例，保持对外接口完全兼容 ──
_DATA_PROXY = DataProxy()
_DATA_LAKE = LocalDataLake(_DATA_PROXY)

def fetch_spot(): return _DATA_LAKE.fetch_spot()
def fetch_hist(code, start, end): return _DATA_LAKE.fetch_hist(code, start, end)
def fetch_index(symbol): return _DATA_LAKE.fetch_index(symbol)
def fetch_core_pool(): return _DATA_LAKE.fetch_core_pool()
def fetch_hot_sectors(): return _DATA_LAKE.fetch_hot_sectors()
def fetch_northbound_flow(): return _DATA_LAKE.fetch_northbound_flow()


# ═════════════════════════════════════════════════════════════════════════════
# 7. 技术面特征抽取引擎 (Feature Extraction Engine)
# ═════════════════════════════════════════════════════════════════════════════
class AShareTechnicals:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        close = self.df[C.H_CLOSE]
        high, low, vol = self.df[C.H_HIGH], self.df[C.H_LOW], self.df[C.H_VOL]
        
        for span in (5, 10, 20, 60): 
            self.df[f'MA{span}'] = close.rolling(span).mean()
        self.df['MA5_V'] = vol.rolling(5).mean()
        self.df['MA20_V'] = vol.rolling(20).mean()

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        self.df['DIF'] = ema12 - ema26
        self.df['DEA'] = self.df['DIF'].ewm(span=9, adjust=False).mean()
        self.df['MACD'] = (self.df['DIF'] - self.df['DEA']) * 2

        self.df['ATR'], self.df['ADX'] = MathUtils.calc_atr_adx(self.df)
        
        delta = close.diff()
        gain = delta.clip(lower=0.0).rolling(14).mean()
        loss = (-delta.clip(upper=0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        self.df['RSI14'] = 100 - (100 / (1 + rs))

        self.df['REF_C'] = close.shift()
        self.df['PCT_CHG'] = close.pct_change() * 100
        self.df['OBV'] = np.where(close > self.df['REF_C'], vol, np.where(close < self.df['REF_C'], -vol, 0)).cumsum()
        
        # [Smart Money Correlation] 
        self.df['SM_CORR'] = self.df['PCT_CHG'].rolling(20).corr(vol)
        
        # [Amihud Illiquidity]
        amplitude = (high - low) / self.df['REF_C'].replace(0, np.nan) * 100
        is_limit = (amplitude == 0) & (self.df['PCT_CHG'].abs() > 4.5)
        amihud_raw = self.df['PCT_CHG'].abs() / (vol * close + 1e-5) * 1e6
        self.df['AMIHUD'] = np.where(is_limit, 99999.0, amihud_raw)
        self.df['AMIHUD_20'] = self.df['AMIHUD'].rolling(20).mean()
        self.df['AMIHUD_20_RANK'] = self.df['AMIHUD_20'].expanding(min_periods=20).rank(pct=True)
        self.df['IS_LIMIT'] = is_limit
        
        self.today = self.df.iloc[-1]
        self.yest = self.df.iloc[-2]
        self.two_days_ago = self.df.iloc[-3] if len(self.df) >= 3 else None

    def get_features(self) -> Optional[dict]:
        df, today, yest = self.df, self.today, self.yest
        if pd.isna(today['ATR']) or today['ATR'] <= 1e-5: return None

        min_1y, max_1y = df[C.H_LOW].min(), df[C.H_HIGH].max()
        rng = max_1y - min_1y
        if rng <= 0: return None
        
        price_pct = (today[C.H_CLOSE] - min_1y) / rng
        
        if today[C.H_CLOSE] < today['MA20'] * 0.85: return None

        rsi = float(today.get('RSI14', 50))
        if pd.isna(rsi) or rsi > 85: return None 

        consecutive_down = 0
        for i in range(1, 8):
            if len(df) >= i and df[C.H_CLOSE].iloc[-i] < df[C.H_OPEN].iloc[-i]:
                consecutive_down += 1
            else:
                break

        extreme_shrink_vol = yest[C.H_VOL] < today['MA20_V'] * 0.75

        rec120 = df.iloc[-121:-1]
        has_chip_break = False
        if len(rec120) > 20 and rec120[C.H_VOL].sum() > 0:
            counts, edges = np.histogram(rec120[C.H_CLOSE].values, bins=20, weights=rec120[C.H_VOL].values)
            poc = (edges[counts.argmax()] + edges[counts.argmax() + 1]) / 2
            has_chip_break = bool((today['REF_C'] <= poc) and (today[C.H_CLOSE] > poc))

        red_days = 0
        for i in range(1, 4):
            if df[C.H_CLOSE].iloc[-i] > df[C.H_OPEN].iloc[-i]: red_days += 1
            else: break
            
        total_range = today[C.H_HIGH] - today[C.H_LOW]
        upper_shadow = today[C.H_HIGH] - max(today[C.H_OPEN], today[C.H_CLOSE])
        upper_shadow_pct = (upper_shadow / total_range * 100) if total_range > 1e-5 else 0.0

        last_hist_pct = float(df['PCT_CHG'].iloc[-2]) if len(df) >= 2 else 0.0
        has_pullback = bool(
            today[C.H_CLOSE] >= today['MA20'] * 0.97 and 
            today[C.H_VOL] < today['MA5_V'] * 1.2 and
            -6.0 <= last_hist_pct <= 3.5
        )
        
        surge_5d = (today[C.H_CLOSE] / df[C.H_CLOSE].iloc[-6] - 1) * 100 if len(df) >= 6 else 0.0
        
        vcp_amp, is_true_vcp = MathUtils.calc_vcp_quality(df)

        macd_divergence = False
        if len(df) >= 40:
            w1_low = float(df[C.H_LOW].iloc[-40:-20].min())
            w2_low = float(df[C.H_LOW].iloc[-20:].min())
            w1_macd_min = float(df['MACD'].iloc[-40:-20].min())
            w2_macd_min = float(df['MACD'].iloc[-20:].min())
            if w2_low < w1_low and w2_macd_min > w1_macd_min and w2_macd_min < 0:
                if float(today.get('ADX', 50)) < 30 or extreme_shrink_vol or is_true_vcp:
                    macd_divergence = True

        recent_consec_zt = False
        if len(df) >= 5:
            recent_consec_zt = bool(((df['PCT_CHG'].iloc[-6:-1] >= 9.5).rolling(2).sum() >= 2).any())

        is_first_dip = False
        if recent_consec_zt:
            today_is_green = today[C.H_CLOSE] < today[C.H_OPEN] or float(df['PCT_CHG'].iloc[-1]) < 0
            yest_is_red = yest[C.H_CLOSE] >= yest[C.H_OPEN]
            
            above_ma5 = today[C.H_CLOSE] >= today['MA5']
            no_nuclear = float(df['PCT_CHG'].iloc[-1]) > -6.5
            no_trap = upper_shadow_pct < 20.0
            vol_shrink = today[C.H_VOL] <= yest[C.H_VOL] * 1.1
            
            if today_is_green and yest_is_red and above_ma5 and no_nuclear and no_trap and vol_shrink:
                is_first_dip = True

        return {
            'is_first_dip': is_first_dip,
            'macd_divergence': macd_divergence,
            'price_pct': price_pct, 'max_1y': max_1y, 'adx': float(today['ADX']),
            'bull_rank': (today['MA20'] > today['MA60']),
            'extreme_shrink_vol': extreme_shrink_vol,
            'has_zt': bool((df['PCT_CHG'].iloc[-61:-1] >= 9.5).any()),
            'has_consecutive_zt': bool(((df['PCT_CHG'].iloc[-61:-1] >= 9.5).rolling(2).sum() >= 2).any()),
            'vcp_amp': vcp_amp,
            'is_true_vcp': is_true_vcp,
            'upper_shadow_pct': upper_shadow_pct,
            'lower_shadow_ratio': (min(today[C.H_OPEN], today[C.H_CLOSE]) - today[C.H_LOW]) / today[C.H_OPEN] if pd.notna(today[C.H_OPEN]) and today[C.H_OPEN] > 0 else 0.0,
            'has_obv_break': bool(df['OBV'].iloc[-1] > df['OBV'].iloc[-21:-1].max()),
            'has_pullback': has_pullback,
            'has_chip_break': has_chip_break,
            'dist_ma20': (today[C.H_CLOSE] / today['MA20'] - 1) * 100,
            'dist_ma60': (today[C.H_CLOSE] / today['MA60'] - 1) * 100 if 'MA60' in today and today['MA60'] > 0 else 0.0,
            'red_days': red_days,
            'rsi': rsi,
            'consecutive_down': consecutive_down,
            'surge_5d': surge_5d,
            'macd_dea': float(today['DEA']),
            'ma10_val': float(today['MA10']), 'ma20_val': float(today['MA20']), 'atr_val': float(today['ATR']),
            'close_val': float(today[C.H_CLOSE]),
            'low_val': float(today[C.H_LOW]), 'recent_20_low': float(df[C.H_LOW].iloc[-20:].min()),
            'boll_lower': float(today['MA20'] - 2 * df[C.H_CLOSE].iloc[-20:].dropna().std()) if len(df[C.H_CLOSE].iloc[-20:].dropna()) >= 2 else np.nan,
            'close_60d_ago': float(df[C.H_CLOSE].iloc[-60]) if len(df) >= 60 else 0.0,
            'sm_corr': float(today.get('SM_CORR', 0.0)) if len(df) >= 60 and not bool(today.get('IS_LIMIT', False)) and not pd.isna(today.get('SM_CORR')) else 0.0,
            'amihud_20': float(today.get('AMIHUD_20', 0.0)) if not pd.isna(today.get('AMIHUD_20')) else 0.0,
            'amihud_20_rank': float(today.get('AMIHUD_20_RANK', 0.0)) if not pd.isna(today.get('AMIHUD_20_RANK')) else 0.0,
            'wq_41_divergence': bool((today[C.H_CLOSE] >= today[C.H_OPEN]) and 
                                     (len(df) >= 60 and (today[C.H_CLOSE] - df[C.H_CLOSE].iloc[-60:].min()) / (df[C.H_CLOSE].iloc[-60:].max() - df[C.H_CLOSE].iloc[-60:].min() + 1e-5) > 0.90) and 
                                     (today[C.H_VOL] < today['MA20_V'] * 0.5)),
            'clv': float((today[C.H_CLOSE] - today[C.H_LOW]) / (today[C.H_HIGH] - today[C.H_LOW])) if today[C.H_HIGH] > today[C.H_LOW] else -1.0,
            'overnight_return': float((today[C.H_OPEN] / yest[C.H_CLOSE] - 1) * 100) if pd.notna(today[C.H_OPEN]) and pd.notna(yest[C.H_CLOSE]) and yest[C.H_CLOSE] > 0 else 0.0,
            'pct_chg': float(today.get('PCT_CHG', 0.0)),
        }


# ═════════════════════════════════════════════════════════════════════════════
# 8. 打分与自适应演化引擎 (Scoring & Evolution Engine)
# ═════════════════════════════════════════════════════════════════════════════
def apply_scoring(data: dict, now: datetime, m_regime: str, vol_surge: bool, win_stats: dict, is_fallback: bool = False) -> tuple[int, str, str]:
    adx = data['adx']
    tw, rw = (1.4, 0.7) if adx > 25 else (0.8, 1.4) if adx < 15 else (1.0, 1.0)
    
    f_val, f_mom, f_rev, f_risk = 1.0, 1.0, 1.0, 1.0
    regime_msg = ""
    if m_regime == 'BULL':
        f_mom, f_val, f_risk = 1.3, 0.8, 0.8  
        regime_msg = "🔥 **[多头加权]** 重动量突破，容忍高位波动"
    elif m_regime == 'BEAR':
        f_val, f_mom, f_rev, f_risk = 1.3, 0.6, 1.2, 1.5  
        regime_msg = "🐻 **[空头加权]** 重防守低估，严惩高位接盘"
    elif m_regime == 'PANIC':
        f_rev, f_mom, f_val, f_risk = 1.5, 0.3, 1.2, 1.5  
        regime_msg = "🧊 **[冰点加权]** 重超跌反转，规避连板接力"
    else:
        regime_msg = "⚖️ **[均衡加权]** 因子权重保持中立映射"

    if vol_surge:
        f_mom += 0.2
        regime_msg += " | 🌊 **[量能爆发]** 大盘放量，动量进一步加权"
        
    meta = (
        f"- 🧭 **趋势雷达**：{'处于强势主升浪中' if adx > 25 else '正处于底部反转期' if adx < 15 else '平稳震荡蓄势中'}\n"
        f"- ⚙️ **因子暴露**：{regime_msg}"
    )

    in_danger, danger_label = is_earnings_danger_zone(now)

    factors = get_factors_config(f_val, f_mom, f_rev, f_risk, tw, rw, m_regime, in_danger, danger_label)

    group_scores = {}
    group_reasons = {}
    penalty_score = 0
    penalty_reasons = []
    
    for f in factors:
        if f.condition(data):
            pts = int(f.points * f.weight)
            try:
                msg = f.template.format(**data)
            except KeyError:
                msg = f.template
                
            if f.points < 0:
                penalty_score += pts
                penalty_reasons.append(msg)
            else:
                # 互斥分组：同组只取最高分的一个因子，防止逻辑重叠导致分值无限拔高
                if not f.group:
                    log.warning(f"⚠️ 因子缺失 group 配置，已强制分配到默认组: {f.template[:20]}...")
                group = f.group if f.group else "DEFAULT_GROUP"
                if group not in group_scores or pts > group_scores[group]:
                    group_scores[group] = pts
                    group_reasons[group] = msg

    # 正向加分总和硬性封顶 (+45分上限)
    total_bonus = sum(group_scores.values())
    capped_bonus = min(total_bonus, 45)
    
    if total_bonus > 45:
        group_reasons['CAPPED'] = f"- 🛡️ **[溢出截断]**：多项利好共振(理论+{total_bonus}分)，为防多重共线性过拟合，强行封顶至 +45 分。"

    raw_score = 45 + capped_bonus + penalty_score
    reasons = [meta] if meta else []
    reasons.extend(group_reasons.values())
    reasons.extend(penalty_reasons)
                
    if data.get('is_etf', False):
        raw_score += 25
        reasons.append("- 🧬 **[ETF 纯血通道]**：触发专属被动基金过滤逻辑，跳过基本面考核，执行纯形态打分 (+25分)。")
    elif is_fallback:
        raw_score += 25
        reasons.append("- 🛟 **[失明补偿]**：因基本面数据链断裂，系统强行加权 25 分以维持基础纯形态决选。")

    raw_score = max(0, min(raw_score, 100))
    
    # ── 【AI 胜率自进化机制】 ──
    bucket = get_score_bucket(raw_score)
    b_stats = win_stats.get(bucket, {'win': 0, 'total': 0})
    if b_stats['total'] >= 5:  
        wr = b_stats['win'] / b_stats['total']
        multiplier = 0.8 + 0.4 * wr
        final_score = int(raw_score * multiplier)
        reasons.append(f"- 🧬 **AI自进化**：该分数段实盘历史胜率 `{wr*100:.1f}%`，系统执行动态调分：**{raw_score} ➡️ {final_score}**")
    else:
        final_score = raw_score
        reasons.append(f"- 🧬 **AI自进化**：该分数段暂无足够历史样本以供进化。")
    
    final_score = max(0, min(final_score, 100))
    
    if final_score >= 85:
        level = '⭐⭐⭐⭐⭐ 🐯 **[S级·老虎机]** (胜率极高，跌势有限)'
    elif final_score >= 75:
        level = '⭐⭐⭐⭐ 🐕 **[A级·看门狗]** (防守兼备，需耐心等涨)'
    elif final_score >= 70:
        level = '⭐⭐⭐ 🦊 **[B+级·小狐狸]** (次优机会，必须控制仓位)'
    else:
        level = '⭐⭐ 🐒 **[B级·小猕猴]** (上蹿下跳振幅大，新手回避)'
        
    return final_score, level, '\n'.join(reasons)

def vectorized_prescreen(pool: pd.DataFrame, is_fallback: bool = False) -> pd.Series:
    """[性能优化] 向量化预筛分引擎，彻底消除 apply 带来的行级遍历开销"""
    s = pd.Series(50.0, index=pool.index)
    
    vr = pool.get(C.S_VR, pd.Series(1.0, index=pool.index)).fillna(1.0).astype(float)
    pct = pool.get(C.S_PCT, pd.Series(0.0, index=pool.index)).fillna(0.0).astype(float)
    pe = pool.get(C.S_PE, pd.Series(-1.0, index=pool.index)).fillna(-1.0).astype(float)
    pb = pool.get(C.S_PB, pd.Series(99.0, index=pool.index)).fillna(99.0).astype(float)
    mcap = pool.get(C.S_MCAP, pd.Series(0.0, index=pool.index)).fillna(0.0).astype(float)
    amt = pool.get(C.S_AMT, pd.Series(0.0, index=pool.index)).fillna(0.0).astype(float)
    
    if not is_fallback:
        s += np.where((vr > 1.5) & (pct > 0), 15.0, 0.0)
        s -= np.where(vr < 0.7, 10.0, 0.0)
        s += np.where((pe > 0) & (pe < 40), 8.0, 0.0)
        s += np.where((pb > 0) & (pb < 2), 5.0, 0.0)
        s += np.where((mcap > 50e8) & (mcap < 500e8), 8.0, 0.0)
        
    s += np.where(amt > 1e8, 5.0, 0.0)
    
    s += np.where((pct > 1.0) & (pct < 7.0), 10.0, 0.0)
    s -= np.where(pct > 9.0, 15.0, 0.0)
    
    if is_fallback:
        # [降级补偿] 兜底模式下由于缺失涨跌幅等特征，给予适当的基础分补偿 (由20降为10，防止过度放宽预筛分门槛)
        s += 10.0
        
    is_etf = pool.index.astype(str).str.startswith(('51', '15', '588', '56'))
    s += np.where(is_etf, 20.0, 0.0)
        
    return s.clip(lower=0.0, upper=100.0)


# ═════════════════════════════════════════════════════════════════════════════
# 9. 核心流水线与主控 (Pipeline & Orchestrator)
# ═════════════════════════════════════════════════════════════════════════════
def is_valid_run_time(now: datetime) -> bool:
    if IS_MANUAL:
        return True
    t = now.hour * 100 + now.minute
    # 【重磅更新：尾盘法放行】将原本 15:05 的锁解除，提前至 14:45，支持在收盘前 15 分钟介入
    return t >= 1445

def process_stock(row: pd.Series, raw_hist: pd.DataFrame, now: datetime, market_ok: bool, index_ret: float, hot_sectors_map: dict) -> Optional[tuple]:
    if len(raw_hist) < 120: return None
    
    hist = raw_hist.copy()
    # 仅在盘中 (15:00 之前) 允许拼接实时 K 线。盘后强制依赖历史日线，杜绝周末/节假日的重复/错乱 K 线拼接。
    is_intraday = now.hour < 15
    if is_intraday and str(hist[C.H_DATE].iloc[-1]) != now.strftime('%Y-%m-%d') and is_valid_run_time(now):
        synthetic = pd.DataFrame([{
            C.H_DATE: now.strftime('%Y-%m-%d'), C.H_OPEN: float(row.get(C.S_OPEN, row[C.S_PRICE])),
            C.H_HIGH: float(row[C.S_HIGH]), C.H_LOW: float(row.get(C.S_LOW, row[C.S_PRICE])),
            C.H_CLOSE: float(row[C.S_PRICE]), C.H_VOL: float(row.get(C.S_VOL, 1.0))
        }])
        hist = pd.concat([hist, synthetic], ignore_index=True)

    if hist.iloc[-1][C.H_VOL] <= 0: return None
    
    engine = AShareTechnicals(hist)
    data = engine.get_features()
    if not data: return None

    atr_pct = (data['atr_val'] / data['close_val']) * 100
    if atr_pct > 8.0:
        return None

    # [Right-Side Filter] 绝对右侧确认：过滤左侧接飞刀
    ma5 = hist['MA5'].iloc[-1] if 'MA5' in hist.columns else hist[C.H_CLOSE].rolling(5).mean().iloc[-1]
    close_today = hist[C.H_CLOSE].iloc[-1]
    close_yest = hist[C.H_CLOSE].iloc[-2] if len(hist) >= 2 else close_today
    close_2d_ago = hist[C.H_CLOSE].iloc[-3] if len(hist) >= 3 else close_yest
    close_3d_ago = hist[C.H_CLOSE].iloc[-4] if len(hist) >= 4 else close_2d_ago
    
    # 1. 均线防守：必须站上 5 日均线
    if close_today < ma5:
        return None
        
    # 2. 拒绝连阴：过去3天重心向下，且今天未反包
    is_3d_down = (close_yest < close_2d_ago) and (close_2d_ago < close_3d_ago)
    if is_3d_down and (close_today <= close_yest):
        return None
        
    # 3. MACD 走弱过滤：水下且未金叉则拒绝
    macd = hist['MACD'].iloc[-1] if 'MACD' in hist.columns else 0
    dif = hist['DIF'].iloc[-1] if 'DIF' in hist.columns else 0
    dea = hist['DEA'].iloc[-1] if 'DEA' in hist.columns else 0
    if macd < 0 and dif < dea:
        return None

    data['pe'] = float(row.get(C.S_PE, 0))
    data['pb'] = float(row.get(C.S_PB, 0))
    data['mcap'] = float(row.get(C.S_MCAP, 0))
    data['is_etf'] = str(row[C.S_CODE]).startswith(('51', '15', '588', '56'))
    data['vol_ratio'] = float(row.get(C.S_VR, 1.0))
    data['rs_rating'] = ((row[C.S_PRICE] / data['close_60d_ago'] - 1) * 100 - index_ret) if data['close_60d_ago'] > 0 else 0
    data['code'] = str(row[C.S_CODE])
    
    mom_3m = hist[C.H_CLOSE].pct_change(63).iloc[-1] if len(hist) > 63 else 0
    mom_12m = hist[C.H_CLOSE].pct_change(252).iloc[-1] if len(hist) > 252 else (hist[C.H_CLOSE].iloc[-1] / hist[C.H_CLOSE].iloc[0] - 1)
    mom_3m_ann = (1 + mom_3m) ** 4 - 1 if not pd.isna(mom_3m) else 0
    mom_12m_ann = mom_12m if not pd.isna(mom_12m) else 0
    data['mom_accel'] = mom_3m_ann - mom_12m_ann
    
    high_250d = hist[C.H_HIGH].rolling(250, min_periods=60).max().iloc[-1]
    current_price = hist[C.H_CLOSE].iloc[-1]
    dist_to_high = (current_price - high_250d) / high_250d if high_250d > 0 else -1
    data['breakout_intensity'] = max(0, 1 + dist_to_high)
    
    data['in_hot_sector'] = data['code'] in hot_sectors_map
    data['hot_sector_name'] = hot_sectors_map.get(data['code'], "热门")
    
    atr_stop = data['close_val'] - 1.5 * data['atr_val']
    stop = atr_stop
    stop = round(stop, 2)
    
    risk_pct = ((row[C.S_PRICE] - stop) / row[C.S_PRICE]) * 100 if row[C.S_PRICE] > 0 else 99
    
    if risk_pct > 25.0: return None 

    return (data, stop, risk_pct) 



def generate_macro_section() -> str:
    """获取外围宏观数据，生成早盘宏观快报的内容"""
    try:
        import yfinance as yf
        tickers = yf.Tickers("^TNX ^VIX ^SKEW HG=F GC=F CL=F ^GSPC")
        hist = tickers.history(period="5d")
        close_df = hist['Close']

        def get_last_pct(ticker):
            s = close_df[ticker].dropna()
            if len(s) >= 2:
                last = s.iloc[-1]
                prev = s.iloc[-2]
                pct = (last - prev) / prev * 100
                return last, pct
            return 0.0, 0.0

        tnx_l, tnx_p = get_last_pct('^TNX')
        vix_l, vix_p = get_last_pct('^VIX')
        skew_l, skew_p = get_last_pct('^SKEW')
        gc_l, gc_p = get_last_pct('GC=F')
        cl_l, cl_p = get_last_pct('CL=F')
        sp500_l, sp500_p = get_last_pct('^GSPC')

        msg = (
            f"### 🌍 隔夜外围与宏观风控快报\n"
            f"- **标普500 (^GSPC)**: `{sp500_l:.2f}` ({sp500_p:+.2f}%)\n"
            f"- **恐慌指数 (^VIX)**: `{vix_l:.2f}` ({vix_p:+.2f}%) " + ("⚠️ **极度恐慌**" if vix_l > 25 else "✅ 情绪稳定") + "\n"
            f"- **黑天鹅指数 (^SKEW)**: `{skew_l:.2f}`\n"
            f"- **美债10年期 (^TNX)**: `{tnx_l:.2f}%` ({tnx_p:+.2f}%)\n"
            f"- **COMEX 黄金 (GC=F)**: `{gc_l:.2f}` ({gc_p:+.2f}%)\n"
            f"- **WTI 原油 (CL=F)**: `{cl_l:.2f}` ({cl_p:+.2f}%)\n\n"
            f"> *数据源: Yahoo Finance (yfinance)*"
        )
        return msg
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"获取宏观数据失败: {e}")
        return f"### 🌍 隔夜外围与宏观指标快报\n⚠️ 外围数据获取失败 ({e})"

def get_ma_trend(cl_series: pd.Series) -> tuple[str, str]:
    """根据收盘价序列判断长短均线趋势"""
    if len(cl_series) < 60: return "数据不足", ""
    ma5 = cl_series.rolling(5).mean().iloc[-1]
    ma20 = cl_series.rolling(20).mean().iloc[-1]
    ma60 = cl_series.rolling(60).mean().iloc[-1]
    close = cl_series.iloc[-1]
    
    mas = [ma5, ma20, ma60]
    max_ma, min_ma = max(mas), min(mas)
    spread = (max_ma - min_ma) / min_ma
    
    if spread < 0.02:
        return "均线粘连", "面临方向性变盘选择，资金观望情绪浓厚"
    elif ma5 > ma20 > ma60:
        if close > ma5:
            return "三线开花(强势多头)", "全面多头排列，上行动能极强，顺势做多"
        else:
            return "多头排列(短期回踩)", "大趋势向上但短期回踩，关注下方均线支撑"
    elif ma5 < ma20 < ma60:
        if close < ma5:
            return "空头瀑布(极度弱势)", "全面空头排列，下行趋势加速，严控仓位"
        else:
            return "空头排列(超跌反弹)", "大级别处于下降通道，当前属于超跌反弹"
    elif ma60 > ma20 and ma5 > ma20:
        return "筑底反弹", "中长线偏空但短期均线拐头向上，左侧资金试盘"
    else:
        return "震荡分化", "长短均线方向不一，无明显单边趋势"

def extract_market_context(df_raw: pd.DataFrame, c_conf: Config) -> tuple[pd.DataFrame, bool, str, float, bool, str, bool]:

    market_ok, market_msg, index_ret, market_overheated = True, "", 0.0, False
    market_regime = "NEUTRAL"
    vol_surge = False
    
    if len(df_raw) < 1000: return pd.DataFrame(), False, "API 异常，横截面数据不足", 0.0, False, market_regime, vol_surge
    
    north_flow, north_msg = fetch_northbound_flow()
    
    try:
        df_raw[C.S_PE] = pd.to_numeric(df_raw[C.S_PE], errors='coerce')
        df_raw[C.S_PB] = pd.to_numeric(df_raw[C.S_PB], errors='coerce')

        idx_df = fetch_index('sh000001')
        cl = idx_df['close']
        ma60 = cl.rolling(60).mean().iloc[-1] if len(cl) >= 60 else cl.iloc[-1]
        ma20 = cl.rolling(20).mean().iloc[-1] if len(cl) >= 20 else cl.iloc[-1]
        ma5 = cl.rolling(5).mean().iloc[-1] if len(cl) >= 5 else cl.iloc[-1]
        pct = (cl.iloc[-1] - cl.iloc[-2]) / cl.iloc[-2] * 100 if len(cl) >= 2 else 0.0
        
        # MACD
        exp1 = cl.ewm(span=12, adjust=False).mean()
        exp2 = cl.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9, adjust=False).mean()
        macd_dead_cross = macd.iloc[-1] < signal_line.iloc[-1]
        beta_broken = (cl.iloc[-1] < ma60) and (ma5 < ma20) and macd_dead_cross
        vol_col = 'volume' if 'volume' in idx_df.columns else 'amount' if 'amount' in idx_df.columns else None
        if vol_col and len(idx_df) >= 6:
            today_vol = float(idx_df[vol_col].iloc[-1])
            ma5_vol = float(idx_df[vol_col].iloc[-6:-1].mean())
            if ma5_vol > 0 and today_vol > ma5_vol * 1.25:
                vol_surge = True
        
        market_trend_ok = cl.iloc[-1] > ma20
        up_count = (df_raw[C.S_PCT] > 0).sum()
        down_count = (df_raw[C.S_PCT] < 0).sum()
        total_count = up_count + down_count
        breadth = up_count / total_count if total_count > 0 else 0.5
        vix_proxy = cl.pct_change().abs().tail(5).mean() * 100
        index_ret = ((cl.iloc[-1] / cl.iloc[-60]) - 1) * 100 if len(cl) >= 60 else 0.0
        
        zt_count = (df_raw[C.S_PCT] >= 9.0).sum() 
        dt_count = (df_raw[C.S_PCT] <= -9.0).sum() 
        total_amt = df_raw[C.S_AMT].sum() / 1e8 
        
        sentiment_addon = ""
        if zt_count > 150:
            market_overheated = True
            sentiment_addon = "\n- 🚨 **情绪熔断**：今日涨停破百市场极度狂欢，系统禁止推荐个股防踩踏！"

        if breadth < 0.25 and vix_proxy > 1.5:
            market_regime = "PANIC"
            market_state = "🧊 **恐慌冰点 (PANIC)**"
            advice = "仓位 10%-20%。系统性风险急剧释放，多看少动，仅适合轻仓左侧防守试错。"
            market_ok = False
        elif market_trend_ok and breadth > 0.6:
            market_regime = "BULL"
            market_state = "🔥 **强势多头 (BULL)**"
            advice = "仓位 60%-80%。赚钱效应极佳，资金活跃，跟随主线积极做多。"
            market_ok = True
        elif not market_trend_ok and breadth <= 0.4:
            market_regime = "BEAR"
            market_state = "🐻 **弱势空头 (BEAR)**"
            advice = "仓位 20%-30%。均线压制且空头力量主导，控制手管住回撤。"
            market_ok = False
        else:
            market_regime = "NEUTRAL"
            market_state = "⚖️ **震荡均衡 (NEUTRAL)**"
            advice = "仓位 40%-60%。指数暂无大级别风险，重个股轻大盘，不盲目追高。"
            market_ok = True
            
        vix_20d = cl.pct_change().std() * np.sqrt(252) * 100 if len(cl) >= 20 else vix_proxy
        is_crashing = (cl.iloc[-1] < ma20) and (pct < -1.5) and (vix_20d > 20.0 or vix_proxy > 2.0)
        
        if is_crashing or (beta_broken and vol_surge and pct < -1.0):
            advice = "🚨 **【大盘绝对熔断警报】** 大盘遭遇放量暴跌且波动率极速放大（典型主跌浪/股灾前兆）！系统已触发 Level 2 级别防守熔断，今日强制空仓，停止一切选股运算，绝不接飞刀！\n\n" + advice
            # 如果大盘极度恶劣，直接返回空 DataFrame 熔断后续一切算股逻辑
            return pd.DataFrame(), False, advice, index_ret, market_overheated, "PANIC", vol_surge
            
        elif beta_broken:
            advice = "🚨 **【大盘结构性走熊警告】** 大盘日线跌破 60 日均线且 MACD 死叉，处于绝对熊市结构！建议空仓或极低仓位试错，由于个股可能分化，今日仍推送高潜质标的供观察，但严禁盲目重仓做多！\n\n" + advice

        if north_flow <= -80.0:
            market_ok = False
            market_state += " ⚠️(外资砸盘)"
            advice = "🚨 **外资大举出逃，强行压制做多情绪，建议立即防守并缩减仓位！** " + advice
        elif north_flow >= 50.0 and not market_overheated:
            market_ok = True
            market_state += " 🚀(外资抢筹)"

        fallback_warning = "\n\n> ⚠️ **数据源降级警报**\n> 频繁测试触发东方财富接口临时限制，已切至新浪备用源。基本面过滤(市盈率/量比等)暂时失效，请自行排雷！"

        # Use get_ma_trend for CSI 300 or SH000001
        trend_name, trend_desc = get_ma_trend(cl)
        
        # Append hot sectors explicitly
        hot_map = fetch_hot_sectors()
        hot_str = ""
        if hot_map:
            # Reverse map to count occurrences
            from collections import Counter
            sec_counts = Counter(hot_map.values())
            top_sectors = [f"{s}({c})" for s, c in sec_counts.most_common(5)]
            hot_str = f"\n- **核心主线**：{', '.join(top_sectors)}"
            
        # Optional: pull macro block if run_mode is normal and we want to attach it? Wait, we can attach macro block in normal mode too!
        # The user requested: "将宏观诊断置于报告头部，热点主线紧随其后"
        macro_str = generate_macro_section() + "\n\n"

        market_msg = (
            f"{macro_str}"
            f"### 📊 A股深度诊断\n"
            f"- **大盘趋势 (MA系统)**：`{trend_name}` - {trend_desc}\n"
            f"- **上证指数**：`{cl.iloc[-1]:.2f}` (今日 **{pct:+.2f}%**)\n"
            f"- **综合判定**：{market_state}\n"
            f"- **市场广度**：红盘 `{up_count}` 家 / 绿盘 `{down_count}` 家 (涨停 `{zt_count}` / 跌停 `{dt_count}`)\n"
            f"- **两市量能**：约 `{total_amt:.0f}` 亿元{sentiment_addon}{north_msg}{hot_str}\n\n"
            f"**💡 仓位建议**：{advice}{fallback_warning}"
        )
    except Exception as e:
        log.warning(f"宏观状态解析失败: {e}")
        market_msg = f"大盘深度解析由于网络原因失败: {e}\n"
    
    df = df_raw.dropna(subset=list(c_conf.REQUIRED_COLS))
    df = df[~df[C.S_NAME].str.contains('ST|退')]
    return df, market_ok, market_msg, index_ret, market_overheated, market_regime, vol_surge

# ═════════════════════════════════════════════════════════════════════════════
# 10. 统一通知网关 (Unified Notification Gateway)
# ═════════════════════════════════════════════════════════════════════════════
class NotificationGateway:
    @staticmethod
    def _send_to_webhook(url: str, is_feishu: bool, msg_title: str, msg_text: str, sec_keyword: str, template: str = "blue") -> requests.Response:
        headers = {"Content-Type": "application/json"}
        if is_feishu:
            payload = {
                "msg_type": "interactive",
                "card": {
                    "header": {
                        "title": {"tag": "plain_text", "content": msg_title},
                        "template": template
                    },
                    "elements": [{"tag": "markdown", "content": msg_text}]
                }
            }
        else:
            final_title = msg_title if sec_keyword in msg_title else f"{sec_keyword} | {msg_title}"
            final_text = msg_text
            if sec_keyword not in final_text:
                final_text = f"### {sec_keyword}\n\n{final_text}"
            payload = {
                'msgtype': 'markdown',
                'markdown': {
                    'title': final_title,
                    'text': final_text
                }
            }
            
        # 异常与超时重试 (最高尝试2次)
        for attempt in range(2):
            try:
                res = requests.post(url, json=payload, headers=headers, timeout=10)
                res.raise_for_status()
                return res
            except Exception as e:
                if attempt == 1:
                    raise e
                time.sleep(1)
        raise RuntimeError("Push failed after retries")

    @classmethod
    def send(cls, title: str, content: str, template: str = "blue") -> None:
        webhooks = []
        if config.DINGTALK_WEBHOOK:
            webhooks.append((config.DINGTALK_WEBHOOK, False, "钉钉"))
        if config.FEISHU_WEBHOOK:
            webhooks.append((config.FEISHU_WEBHOOK, True, "飞书"))
            
        if not webhooks:
            log.warning("⚠️ 未配置任何 Webhook 环境变量 (DINGTALK_WEBHOOK / FEISHU_WEBHOOK)，通知已跳过！")
            return
            
        sec_keyword = config.NOTIFY_SEC_KEYWORD
        CHUNK_SIZE = 18000
        chunks = [content[i:i+CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
        
        if len(chunks) > 3:
            log.warning(f"⚠️ 推送消息过长 (片段数: {len(chunks)})，强行截断至前 3 篇以防流控！")
            chunks = chunks[:3]
            chunks[-1] += f"\n\n> ⚠️ *(本文因超出承载极限，尾部数据已被系统强制截断)*"
            
        for idx, chunk in enumerate(chunks):
            text = chunk if idx == 0 else f"_(续上条)_\n\n{chunk}"
            msg_title = title if len(chunks) == 1 else f"{title} (Part {idx+1}/{len(chunks)})"
            
            for url, is_feishu, name in webhooks:
                try:
                    res = cls._send_to_webhook(url, is_feishu, msg_title, text, sec_keyword, template)
                    res_dict = res.json()
                    
                    is_err = False
                    if is_feishu:
                        if res_dict.get('code', 0) != 0: is_err = True
                    else:
                        if res_dict.get('errcode', 0) != 0: is_err = True
                        
                    if is_err:
                        log.error(f"❌ {name} 推送接口拒绝: {res_dict}")
                    else:
                        log.info(f"✅ {name} 推送成功")
                except Exception as e:
                    log.error(f"❌ {name} 推送失败 (已隔离保护): {e}")
            
            if idx < len(chunks) - 1:
                time.sleep(1)

def send_dingtalk(signals: dict[str, list[Signal]], watchlist: list, total_pool: int, total_market: int, market_msg: str) -> None:
    now_ts = datetime.now(TZ_BJS)
    now_str = now_ts.strftime('%Y-%m-%d %H:%M')
    run_mode = config.RUN_MODE
    
    header = f"## 🤖 AI量化选股系统\n> **{now_str}**\n\n"
    if run_mode == 'market_only' or run_mode == 'morning':
        header = f"## 🤖 AI量化大盘深度体检\n> **{now_str}**\n\n"
    elif run_mode not in ('market_only', 'morning') and total_market > 0:
        total_signals = sum(len(sigs) for sigs in signals.values()) if isinstance(signals, dict) else len(signals)
        pass_rate = total_signals / max(total_pool, 1) * 100 if total_pool > 0 else 0
        header += f"**🔬 漏斗数据**：全市场白名单 `{total_market}` 只，异动提取 `{total_pool}` 只，完美过线 `{total_signals}` 只 (优选率 **{pass_rate:.1f}%**)\n\n"
        
    if market_msg:
        header += f"{market_msg}\n\n---\n\n"

    if run_mode == 'morning':
        # 早盘仅发宏观快报，这里直接用 generate_macro_section 生成极简版
        content = f"## 🌅 AI量化开盘快报\n> **{now_str}**\n\n" + generate_macro_section() + "\n\n> 🔔 早盘重点监控外围风险，避免开盘盲目冲动。尾盘 14:45 将发送完整选股报告。"
    elif run_mode == 'market_only':
        content = header + "✅ 大盘分析播报完毕，本次任务短路了全量个股运算。"
    elif "接口异常" in market_msg or "网络原因失败" in market_msg:
        content = header + "⚠️ 今日部分个股数据扫描因接口受限中断，已为您提供核心大盘分析参考。"
    elif not any(signals.values()) and not watchlist:
        if not PUSH_EMPTY: return
        content = f"{header}✅ **系统检测结果**：今日未发现形态完全符合安全边际的标的，建议**空仓防守**。"
    else:
        content = header
        has_any_signal = any(signals.values())
        
        if has_any_signal:
            # Removed subjective cold gate
            
            def format_signal(s):
                warn_msg = "> ⚡ **【风险警示】** 该股为创业板(波动±20%)，请务必**缩减仓位**。\n\n" if str(s.code).startswith('300') else ""
                prefix = '1' if str(s.code).startswith('6') else '0'
                tdx_market = 'SH' if str(s.code).startswith('6') else 'SZ' 
                
                sina_market = 'sh' if str(s.code).startswith('6') else 'sz'
                code_str = str(s.code)
                
                if code_str.startswith(('8', '4', '9')):
                    kline_url = "https://dummyimage.com/800x400/f3f4f6/9ca3af.png&text=No+Chart+Available"
                else:
                    kline_url = f"http://image.sinajs.cn/newchart/weekly/n/{sina_market}{s.code}.gif"
                
                return (
                    f"#### 🎯 {s.name} (`{s.code}`)\n"
                    f"{warn_msg}"
                    f"- **综合评级**：`{s.score}` 分 {s.level}\n"
                    f"- **今日收盘**：`¥{s.price}` ({s.pct_chg}) [📈 周K图]({kline_url})\n\n"
                    f"**💡 核心逻辑**\n{s.reasons}\n\n"
                    f"**🛡️ 交易计划**\n"
                    f"{s.money_risk_msg}\n"
                    f"{s.tranche_plan_msg}\n"
                    f"{s.plan_b_msg}\n"
                    f"> ⚠️ 纪律: 破防守线 `¥{s.stop_loss}` 止损; 高开>4%放弃; 创新高按ATR止盈。\n\n"
                    f"[🔗 东财App看盘](https://quote.eastmoney.com/unify/r/{prefix}.{s.code}) | 通达信: `{s.code}`"
                )

            # --- Formatting Sections ---
            if signals.get('Resonance'):
                content += "### 🔥 今日唯一上榜：全周期共振精选 (Top 5)\n\n"
                parts = [format_signal(s) for s in signals['Resonance']]
                content += "\n\n---\n\n".join(parts) + "\n\n---\n\n"
                
        else:
            content += "✅ 今日未发现 B+ 级以上核心机会，正式推荐列表空仓防守中。\n"

        if watchlist:
            watch_lines = "\n".join(
                f"- `{code}` **{name}** (¥{price}) 得分: **{score}**"
                for name, code, score, price in watchlist[:5]
            )
            content += (
                f"### 👁️ 候补观察池（只看不买）\n"
                f"{watch_lines}\n\n"
                f"*注：以上标的评级不足 70 分，系统判断波动或风险偏大，暂不提供操作剧本。待其评级升至发车线后再考虑介入。*"
            )
        
        pass # Removed subjective reflection
        
    NotificationGateway.send('🤖 AI量化盘后提醒', content)

def get_signals() -> tuple[list[Signal], list, set, int, str, int]:
    now = datetime.now(TZ_BJS)
    
    log.info('🚀 防呆长线安全级·盘后复盘引擎启动...')
    if not IS_MANUAL and not is_valid_run_time(now): 
        return {}, [], set(), 0, "", 0

    pushed = load_pushed_state() 

    try:
        df_raw = fetch_spot()
    except Exception as e:
        log.error(f"❌ 核心横截面行情获取失败: {e}")
        return {}, [], pushed, 0, f"⚠️ **行情接口异常，体检中断**: {e}", 0

    c_conf = Config()
    df_clean, m_ok, m_msg, idx_ret, m_overheated, m_regime, vol_surge = extract_market_context(df_raw, c_conf)

    if 'DATA_MODE' in df_raw.columns and (df_raw['DATA_MODE'] == 'T+1_FALLBACK').any():
        m_msg += "\n\n> 🚨 **严重警告**：今日所有实时行情流中断，当前所有技术信号均基于【昨日 T-1 收盘截面】生成，严禁用于今日盘中实盘交易！\n\n"

    tracker_msgs = AdvisoryTracker.evaluate_and_clean(df_raw)
    if tracker_msgs:
        m_msg += "\n\n**📢 往期辅助信号跟踪**\n" + "\n".join(tracker_msgs) + "\n"

    if config.RUN_MODE in ('market_only', 'morning'):
        log.info(f"🤖 [{config.RUN_MODE}模式] 完毕，退出个股运算。")
        return {}, [], pushed, 0, m_msg, len(df_raw)


    hot_sectors_map = fetch_hot_sectors()

    if df_clean.empty:
        return {}, [], pushed, 0, m_msg, 0

    # 统一标准化股票代码为 6 位数字符串，剥离可能存在的市场前缀(如 sh/sz/bj)与后缀(如 .SH/.SZ)
    df_clean[C.S_CODE] = df_clean[C.S_CODE].astype(str).str.extract(r'(\d{6})')[0].fillna('').str.zfill(6)

    core_pool = fetch_core_pool()
    if core_pool:
        str_core_pool = {str(c).zfill(6) for c in core_pool}
        df_clean = df_clean[df_clean[C.S_CODE].isin(str_core_pool)]
        log.info(f"💎 已开启【核心优质股池】模式，限定扫描 {len(core_pool)} 只成分股，匹配后过滤出 {len(df_clean)} 只。")

    # 强制将这些核心筛选列转换为数值型，防止由于数据源格式微调（如 string）导致 .between() 失败
    for col in [C.S_PE, C.S_PB, C.S_MCAP, C.S_TURN, C.S_PRICE, C.S_PCT]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
    if C.S_VR in df_clean.columns:
        df_clean[C.S_VR] = pd.to_numeric(df_clean[C.S_VR], errors='coerce')

    is_t1_fallback = 'DATA_MODE' in df_raw.columns and (df_raw['DATA_MODE'] == 'T+1_FALLBACK').any()
    is_fallback = ((df_clean[C.S_PE] == -1.0).sum() > len(df_clean) * 0.9) or is_t1_fallback
    is_etf = df_clean[C.S_CODE].astype(str).str.startswith(('51', '15', '588', '56'))
    
    # 严格基本面初筛，切断亏损股和严重破净/高估股的入围路径
    pe_cond = (df_clean[C.S_PE] > 0) | (df_clean[C.S_PE].isna()) | is_fallback | is_etf
    pb_cond = (df_clean[C.S_PB].between(0.1, 20.0)) | (df_clean[C.S_PB].isna()) | is_etf
    
    stock_mask = (df_clean[C.S_MCAP].between(c_conf.MIN_CAP, c_conf.MAX_CAP) | df_clean[C.S_MCAP].isna()) & \
                 (df_clean[C.S_TURN].between(c_conf.MIN_TURNOVER, c_conf.MAX_TURNOVER) | df_clean[C.S_TURN].isna()) & \
                 pe_cond & pb_cond & \
                 (~df_clean[C.S_CODE].astype(str).str.startswith(('688', '8', '4', '9')))
    
    mask = (df_clean[C.S_PCT] >= c_conf.MIN_PCT_CHG) & \
           (df_clean[C.S_PRICE] <= c_conf.MAX_PRICE) & \
           (df_clean[C.S_HIGH] > df_clean[C.S_LOW]) & \
           (stock_mask | is_etf)
    
    # 量比过滤放宽：当处于盘后、或量比数据缺失/为0时予以放行，防止盘后量比数据空洞导致个股“全军覆没”
    if C.S_VR in df_clean.columns and not is_fallback:
        mask &= (df_clean[C.S_VR].between(c_conf.MIN_VOL_RATIO, c_conf.MAX_VOL_RATIO) | df_clean[C.S_VR].isna() | (df_clean[C.S_VR] <= 0) | is_etf)

    recent_pushed_codes = {str(c) for c in df_clean[C.S_CODE] if is_recently_pushed(str(c), pushed)}
    pool = df_clean[mask].pipe(lambda d: d[~d[C.S_CODE].isin(recent_pushed_codes)]).copy()
    
    if pool.empty: return {}, [], pushed, len(df_clean), m_msg, len(df_clean)
    
    if len(pool) > 200:
        log.info(f"💡 触发防爆流截断，基于 Spot 截面数据执行廉价预筛分，保留前 200 只高潜标的参与决选。")
        pool['_pre_score'] = vectorized_prescreen(pool, is_fallback)
        pool = pool.sort_values(by='_pre_score', ascending=False).head(200)
        pool = pool.drop(columns=['_pre_score'])
        
    # [风控守门人] 全局 NaN 空洞扫描
    nan_count = pool.isna().sum().sum()
    if nan_count > len(pool) * 2:
        log.warning(f"⚠️ [风控预警] 横截面池中存在 {nan_count} 个数据空洞(NaN)，可能会在后续特征计算中引发静默崩塌。")

    confirmed_data = [] 
    watchlist_data = [] 
    
    end_s, start_s = now.strftime('%Y%m%d'), (now - timedelta(days=450)).strftime('%Y%m%d')
    
    ex2 = ThreadPoolExecutor(max_workers=4)
    futures = {ex2.submit(fetch_hist, r[C.S_CODE], start_s, end_s): r for _, r in pool.iterrows()}
    
    all_hists = []
    stock_infos = {}
    try:
        for f in as_completed(futures, timeout=1200): 
            row = futures[f]
            try:
                hist = f.result()
                hist_ml = hist.copy()
                hist_ml.rename(columns={
                    C.H_DATE: 'date', C.H_OPEN: 'open', C.H_HIGH: 'high',
                    C.H_LOW: 'low', C.H_CLOSE: 'close', C.H_VOL: 'vol'
                }, inplace=True)
                hist_ml['code'] = row[C.S_CODE]
                all_hists.append(hist_ml)
                
                result = process_stock(row, hist, now, m_ok, idx_ret, hot_sectors_map)
                if result:
                    data, stop, risk = result
                    stock_infos[row[C.S_CODE]] = {
                        'row': row, 'data': data, 'stop': stop, 'risk': risk
                    }
            except Exception as e:
                log.warning(f"⚠️ 计算个股 {row[C.S_CODE]} 时发生异常: {e}")
                pass
    except FuturesTimeoutError:
        log.warning("⚠️ 后台运算达到极值，提前熔断保存已有成果。")
    finally:
        ex2.shutdown(wait=False, cancel_futures=True)

    if not all_hists:
        return {}, [], pushed, len(pool), m_msg, len(df_clean)

    # ML Feature Engineering
    panel = pd.concat(all_hists, ignore_index=True)
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel.sort_values(['date', 'code'])
    
    try:
        panel = build_ml_features(panel)
        feature_success = True
    except Exception as e:
        log.error(f"🚨 ML Feature Computation Failed: {e}")
        log.error(traceback.format_exc())
        m_msg += "\n\n> ⚠️ **风控告警**：ML特征计算异常，今日信号基于中性基准 (0.5)！\n\n"
        feature_success = False

    # Extract today's cross section
    today_str = now.strftime('%Y-%m-%d')
    today_panel = panel[panel['date'] == pd.to_datetime(today_str)].copy()
    if len(today_panel) == 0:
        latest_date = panel['date'].max()
        log.warning(f"⚠️ 今日 ({today_str}) 行情尚未落库，使用最新可用日期 ({latest_date.strftime('%Y-%m-%d')}) 作为截面。")
        today_panel = panel[panel['date'] == latest_date].copy()
    
    # Load Models and Predict
    horizons = [1, 5, 10, 20]
    
    import json
    for h in horizons:
        model_path = f'.quantbot_data/prod_pt_model_t{h}.pth'
        meta_path = f'.quantbot_data/prod_pt_meta_t{h}.json'
        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            log.warning(f"⚠️ 找不到 T+{h} 模型或元数据，跳过此周期的选股。")
            today_panel[f'xgb_score_t{h}'] = 0.5
            today_panel[f'rank_t{h}'] = 0.5
            continue
            
        with open(meta_path, 'r') as f:
            features = json.load(f)['features']
            
        ltr = PyTorchDLModel(input_dim=len(features))
        ltr.load_model(model_path)
        
        if feature_success and len(today_panel) > 0:
            xgb_preds = ltr.predict(today_panel, features)
            if len(xgb_preds) > 0 and np.isnan(xgb_preds).all():
                log.critical(f"🚨 致命错误：T+{h} DL输出全部NaN！")
                today_panel[f'xgb_score_t{h}'] = 0.5
            else:
                today_panel[f'xgb_score_t{h}'] = xgb_preds
        else:
            today_panel[f'xgb_score_t{h}'] = 0.5
            
        # Cross-sectional rank
        today_panel[f'rank_t{h}'] = today_panel[f'xgb_score_t{h}'].rank(pct=True)

    # Apply Veto System
    today_panel['core_rank'] = (today_panel.get('rank_t10', 0.5) + today_panel.get('rank_t20', 0.5)) / 2.0
    
    # Baseline pool: Top 10% of core
    candidates = today_panel[today_panel['core_rank'] >= 0.90].copy()
    
    # Filter A: T+1 Anti-Chasing (rank_t1 < 0.95)
    candidates = candidates[candidates['rank_t1'] < 0.95]
    
    # Filter B: T+5 Anti-Bleeding (rank_t5 > 0.40)
    candidates = candidates[candidates['rank_t5'] > 0.40]
    
    # Sort and pick top K
    candidates = candidates.sort_values('core_rank', ascending=False).head(5)
    
    resonance_signals = []
    for _, ml_row in candidates.iterrows():
        code = ml_row['code']
        if code not in stock_infos: continue
        info = stock_infos[code]
        row, data, stop = info['row'], info['data'], info['stop']
        
        score = float(ml_row['core_rank']) * 100 
        t1_rank = float(ml_row.get('rank_t1', 0.5)) * 100
        t5_rank = float(ml_row.get('rank_t5', 0.5)) * 100
        
        t1_tag = "⚖️ 短线波动中性，支持按计划常规建仓"
        if t1_rank >= 80.0:  
            t1_tag = "🎯 短线超跌反弹动能强，早盘绝佳买点"
        elif t1_rank <= 35.0:
            t1_tag = "⏳ 短线存在获利盘抛压，建议观望或逢低挂单"
            
        level = f"🔥 共振得分: {score:.1f} (T+10/T+20主脑)\n  ⚡ T+1独立择时: {t1_rank:.1f} ({t1_tag})\n  ⚡ T+5波段评估: {t5_rank:.1f}"
        
        reas_list = [f"🏆 **核心基本盘打分**: `{score:.2f}`"]
        if 'alpha_reversal_5d' in ml_row and not pd.isna(ml_row['alpha_reversal_5d']):
            reas_list.append(f"🔄 **短期反转强度**: `{ml_row['alpha_reversal_5d']:.3f}`")
        if 'clv' in ml_row and not pd.isna(ml_row['clv']):
            reas_list.append(f"📌 **收盘价位置分布 (CLV)**: `{ml_row['clv']:.2f}`")
        if 'volatility_5d' in ml_row and not pd.isna(ml_row['volatility_5d']):
            reas_list.append(f"📉 **5日波动率**: `{ml_row['volatility_5d']:.2f}%`")
        reas = "\n".join([f"- {r}" for r in reas_list])
        
        target1_price = calc_target_price(row[C.S_PRICE], stop, data)
        money_msg = format_money_risk_msg(row[C.S_PRICE], stop, target1_price)
        tranche_msg = generate_tranche_plan(row[C.S_PRICE], score, m_ok, m_overheated)
        plan_b_msg = generate_plan_b(row[C.S_PRICE], stop, data['ma20_val'])
        
        hold_msg = "> ⏳ **建议持仓周期 (T+10 趋势)**：预期持有参考约半个月。属于长线信号，经过了双重滤网清洗，极其适合作为底仓。"
        
        sig = Signal(
            code=row[C.S_CODE], name=row[C.S_NAME], price=row[C.S_PRICE],
            pct_chg=f"{row[C.S_PCT]}%", score=score, level=level,
            trigger_time=now.strftime('%H:%M'), reasons=reas,
            stop_loss=round(stop, 2), target1=target1_price,
            ma10=round(data['ma10_val'], 2),
            money_risk_msg=money_msg, tranche_plan_msg=tranche_msg,
            plan_b_msg=plan_b_msg, hold_period_msg=hold_msg
        )
        resonance_signals.append(sig)
        
    confirmed_data_dict = {
        'Resonance': resonance_signals
    }
    
    # Push the generated signals to AdvisoryTracker
    AdvisoryTracker.add_signals(resonance_signals, 'T+10') # Use T+10 as default tracking horizon for Resonance
    
    watchlist_data.sort(key=lambda x: (x[2], x[1]), reverse=True) 
    
    today_str = _today_str()
    # 仅将决选且展出给用户的个股记录进状态锁与模拟盘账本
    for group in confirmed_data_dict.values():
        for s in group:
            cd_days = 1 if s.score >= 85 else 3
            expire_dt = now + timedelta(days=cd_days)
            pushed[s.code] = expire_dt.strftime('%Y-%m-%d')


    return confirmed_data_dict, watchlist_data, pushed, len(pool), m_msg, len(df_clean)

# ═════════════════════════════════════════════════════════════════════════════
# 7. Crucible Backtest Engine (VectorBT)
# ═════════════════════════════════════════════════════════════════════════════
import vectorbt as vbt
import gc

class CrucibleBacktestEngine:
    def __init__(self, initial_capital=1000000, max_trials=30):
        self.initial_capital = initial_capital
        self.max_trials = max_trials
        self.trials_run = 0
        log.info("[CRUCIBLE] Backtest Engine Initialized. Max trials: 30")
        
    def run_chunked_backtest(self, panel_path, signal_df, chunk_size_years=5):
        if self.trials_run >= self.max_trials:
            log.error("[CRUCIBLE] Trial budget exceeded (30). Forcing stop to prevent overfitting.")
            return None
            
        self.trials_run += 1
        log.info(f"[CRUCIBLE] Running chunked VectorBT backtest. Trial {self.trials_run}/{self.max_trials}")
        
        df = pd.read_parquet(panel_path)
        df['date'] = pd.to_datetime(df['date'])
        
        start_year = df['date'].dt.year.min()
        end_year = df['date'].dt.year.max()
        
        all_portfolios = []
        for y in range(start_year, end_year + 1, chunk_size_years):
            chunk = df[(df['date'].dt.year >= y) & (df['date'].dt.year < y + chunk_size_years)].copy()
            if chunk.empty: continue
            
            close = chunk.pivot(index='date', columns='code', values='close').ffill()
            volume = chunk.pivot(index='date', columns='code', values='vol').fillna(0)
            
            chunk_sigs = signal_df[(signal_df['date'].dt.year >= y) & (signal_df['date'].dt.year < y + chunk_size_years)]
            entries = chunk_sigs[chunk_sigs['xgb_quantile'] == 5].pivot(index='date', columns='code', values='xgb_quantile').notna()
            exits = chunk_sigs[chunk_sigs['xgb_quantile'] == 1].pivot(index='date', columns='code', values='xgb_quantile').notna()
            
            entries = entries.reindex(index=close.index, columns=close.columns, fill_value=False)
            exits = exits.reindex(index=close.index, columns=close.columns, fill_value=False)
            
            # [CRUCIBLE PROTOCOL] Volume Capacity Constraint (Max 10%)
            max_size = volume * 0.10
            
            # [CRUCIBLE PROTOCOL] Dynamic Slippage during crisis
            slippage = 0.002 # 20 bps base
            
            pf = vbt.Portfolio.from_signals(
                close,
                entries,
                exits,
                size=max_size,
                size_type='shares',
                init_cash=self.initial_capital,
                slippage=slippage,
                freq='D'
            )
            all_portfolios.append(pf)
            
            del chunk, close, volume, entries, exits
            gc.collect()
            
        log.info("[CRUCIBLE] Chunked backtest completed.")
        return all_portfolios

if __name__ == '__main__':
    try:
        config.print_summary(log)
        
        if config.RUN_MODE == 'test_conn':
            log.info("🔔 检测到测试模式，执行推送连通性测试...")
            now = datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M:%S')
            NotificationGateway.send(
                "🤖 AI量化引擎连通性测试",
                f"通知链路测试成功！\n- 时间: {now}\n- 状态: GitHub Actions 触发器已打通。"
            )
        elif not config.DINGTALK_WEBHOOK and not config.FEISHU_WEBHOOK:
            log.warning("未配置 WEBHOOK，将切换为本地打印模式供测试。")
            sigs, watch, pushed, pool_size, m_msg, total_mkt = get_signals()
            log.info("============== 每日投研简报 ==============")
            now_ts = datetime.now(TZ_BJS)
            now_str = now_ts.strftime('%Y-%m-%d %H:%M')
            
            header = f"## 🤖 AI量化选股系统\n> **{now_str}**\n\n"
            total_signals = sum(len(sigs_list) for sigs_list in sigs.values()) if isinstance(sigs, dict) else len(sigs)
            if total_mkt > 0:
                pass_rate = total_signals / max(pool_size, 1) * 100 if pool_size > 0 else 0
                header += f"**🔬 漏斗数据**：全市场白名单 `{total_mkt}` 只，异动提取 `{pool_size}` 只，完美过线 `{total_signals}` 只 (优选率 **{pass_rate:.1f}%**)\n\n"
            
            if m_msg:
                header += f"{m_msg}\n\n---\n\n"
            
            if not any(sigs.values()) and not watch:
                content = f"{header}✅ **系统检测结果**：今日未发现形态完全符合安全边际的标的，建议**空仓防守**。"
            else:
                content = header
                
                if any(sigs.values()):
                    def format_signal(s):
                        warn_msg = "> ⚡ **【风险警示】** 该股为创业板(波动±20%)，请务必**缩减仓位**。\n\n" if str(s.code).startswith('300') else ""
                        return (
                            f"#### 🎯 {s.name} (`{s.code}`)\n"
                            f"{warn_msg}"
                            f"- **综合评级**：`{s.score:.1f}` 分\n"
                            f"- **今日收盘**：`¥{s.price}` ({s.pct_chg})\n\n"
                            f"**💡 核心逻辑**\n{s.reasons}\n\n"
                            f"**🛡️ 交易计划**\n"
                            f"{s.money_risk_msg}\n"
                            f"{s.tranche_plan_msg}\n"
                            f"{s.plan_b_msg}\n"
                            f"{s.hold_period_msg}\n"
                            f"> ⚠️ 纪律: 破防守线 `¥{s.stop_loss}` 止损; 高开>4%放弃; 创新高按ATR止盈。\n\n"
                        )
                    
                    if sigs.get('Resonance'):
                        content += "### 🔥 今日唯一上榜：全周期共振精选 (Top 5)\n\n"
                        parts = [format_signal(s) for s in sigs['Resonance']]
                        content += "\n\n---\n\n".join(parts) + "\n\n---\n\n"
                else:
                    content += "✅ 今日未发现 B+ 级以上核心机会，正式推荐列表空仓防守中。\n"
                
                if watch:
                    watch_lines = "\n".join(
                        f"- `{code}` **{name}** (¥{price}) 得分: **{score:.1f}**"
                        for name, code, score, price in watch[:5]
                    )
                    content += (
                        f"### 👁️ 候补观察池（只看不买）\n"
                        f"{watch_lines}\n\n"
                        f"*注：以上标的评级不足 70 分，系统判断波动或风险偏大，暂不提供操作剧本。待其评级升至发车线后再考虑介入。*"
                    )
            
            print(content)
            log.info("=========================================")
        else:
            sigs, watch, pushed, pool_size, m_msg, total_mkt = get_signals()
            send_dingtalk(sigs, watch, pool_size, total_mkt, m_msg)
            if any(sigs.values()): save_pushed_state(pushed)
            
    except Exception as e:
        log.critical(f"系统崩溃: {e}", exc_info=True)
        error_msg = f"🚨 **AI量化引擎崩溃告警**\n\n**时间**: {_today_str()}\n**环境**: GitHub Actions\n**异常信息**: {str(e)[:300]}..."
        NotificationGateway.send("🚨 AI量化引擎崩溃告警", error_msg, template="red")
        
    finally:
        _DATA_PROXY.cleanup()
