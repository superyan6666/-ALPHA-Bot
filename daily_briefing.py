import os
import json
import time
import pickle
import logging
import logging.handlers
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd
import akshare as ak
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from main import fetch_spot, fetch_hot_sectors, fetch_northbound_flow, Cols, TZ_BJS

log = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# 常量与配置
# ═════════════════════════════════════════════════════════════════════════════
SINA_HEADERS = {"Referer": "https://finance.sina.com.cn/"}
DINGTALK_MAX_LEN = 18000

class Color:
    RED = "#F04864"
    GREEN = "#2fc25b"
    GRAY = "#8c8c8c"
    UP_CN = RED    # A股红涨
    DOWN_CN = GREEN
    UP_US = GREEN  # 美股绿涨
    DOWN_US = RED

@dataclass
class MarketData:
    ashare_indices: dict = field(default_factory=dict)
    global_indices: dict = field(default_factory=dict)
    northbound_flow: tuple = field(default_factory=lambda: (0.0, ""))
    judgments: dict = field(default_factory=lambda: {"macro": [], "us_tech": "", "cn_tech": "", "risk_alert": ""})
    news: list = field(default_factory=list)
    hot_sectors: dict = field(default_factory=dict)
    status: dict = field(default_factory=lambda: {"warnings": [], "errors": []})

# ═════════════════════════════════════════════════════════════════════════════
# 统一 HTTP 客户端
# ═════════════════════════════════════════════════════════════════════════════
_http_session: Optional[requests.Session] = None

def get_http_session() -> requests.Session:
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        retries = Retry(total=2, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
        _http_session.mount('https://', HTTPAdapter(max_retries=retries))
        _http_session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Referer': 'https://finance.sina.com.cn/'
        })
    return _http_session

def safe_request_get(url: str, timeout: int = 8, max_retries: int = 2) -> Optional[requests.Response]:
    session = get_http_session()
    for attempt in range(max_retries):
        try:
            res = session.get(url, timeout=timeout)
            res.raise_for_status()
            return res
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                log.debug(f"请求失败，重试 ({attempt+1}/{max_retries}): {url}")
                time.sleep(1)
            else:
                log.warning(f"网络请求失败: {url} - {e}")
                return None

# ═════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═════════════════════════════════════════════════════════════════════════════
def _today_str() -> str:
    return datetime.now(TZ_BJS).strftime('%Y-%m-%d')

def format_dingtalk_pct(pct: float, is_us: bool = False) -> str:
    if pct > 0:
        color = Color.UP_US if is_us else Color.UP_CN
        return f'<font color="{color}">+{pct:.2f}%</font>'
    elif pct < 0:
        color = Color.DOWN_US if is_us else Color.DOWN_CN
        return f'<font color="{color}">{pct:.2f}%</font>'
    else:
        return f'<font color="{Color.GRAY}"> 0.00%</font>'

def health_check() -> list:
    issues = []
    for mod in ['pandas', 'requests']:
        try:
            __import__(mod)
        except ImportError:
            issues.append(f"❌ {mod} 未安装（必需）")
    for mod in ['yfinance', 'tushare']:
        try:
            __import__(mod)
        except ImportError:
            issues.append(f"⚠️ {mod} 未安装（可选）")
    return issues

def is_trading_hours() -> tuple[bool, str]:
    now = datetime.now(TZ_BJS)
    weekday = now.weekday()
    if weekday >= 5:
        return False, "周末休市"
    
    current_time = now.hour * 60 + now.minute
    
    if 9 * 60 + 30 <= current_time < 11 * 60 + 30:
        return True, "A股早盘"
    elif 13 * 60 <= current_time < 15 * 60:
        return True, "A股午盘"
    elif current_time >= 15 * 60:
        return False, "A股已收盘"
    elif current_time < 9 * 60 + 30:
        return False, "A股未开盘"
    
    return False, "非交易时段"

# ═════════════════════════════════════════════════════════════════════════════
# 数据源模块
# ═════════════════════════════════════════════════════════════════════════════
class MarketStatus:
    NORMAL = "normal"
    SUSPENDED = "suspended"
    LIMIT_UP = "limit_up"
    LIMIT_DOWN = "limit_down"
    ERROR = "error"

class MacroBrain:
    @staticmethod
    def get_ashare_indices() -> tuple[dict, list]:
        indices_data = {}
        errors = []
        
        try:
            url = "https://hq.sinajs.cn/list=s_sh000001,s_sz399001,s_sz399006,s_sz399300"
            res = safe_request_get(url, timeout=8)
            if res is None:
                errors.append("A股指数接口请求失败")
                return {}, errors
            
            for line in res.text.strip().split('\n'):
                if '="' not in line:
                    continue
                try:
                    parts = line.split('="')[1].strip('";').split(',')
                    if len(parts) < 5:
                        continue
                    
                    name = parts[0]
                    price = float(parts[1])
                    pct = float(parts[3])
                    
                    if name == "沪深300":
                        target_name = "沪深300"
                    elif name == "创业板指":
                        target_name = "创业板指"
                    elif name == "上证指数":
                        target_name = "上证指数"
                    elif name == "深证成指":
                        target_name = "深证成指"
                    else:
                        target_name = name
                    
                    status = MarketStatus.NORMAL
                    if price == 0 or pct == -100:
                        status = MarketStatus.SUSPENDED
                    elif pct >= 9.9:
                        status = MarketStatus.LIMIT_UP
                    elif pct <= -9.9:
                        status = MarketStatus.LIMIT_DOWN
                    
                    indices_data[target_name] = {"pct": pct, "price": price, "status": status}
                    
                except (IndexError, ValueError) as e:
                    errors.append(f"解析A股数据失败: {e}")
            
            log.info(f"新浪A股大盘获取成功: {indices_data}")
        except Exception as e:
            errors.append(f"获取A股指数异常: {e}")
        
        return indices_data, errors

    @staticmethod
    def get_global_indices() -> tuple[dict, list]:
        indices_data = {}
        errors = []
        
        try:
            url = "https://hq.sinajs.cn/list=int_dji,int_nasdaq,int_sp500,b_HSI"
            res = safe_request_get(url, timeout=8)
            if res is None:
                errors.append("外盘指数接口请求失败")
                return {}, errors
            
            for line in res.text.strip().split('\n'):
                if '="' not in line:
                    continue
                try:
                    key = line.split('=')[0]
                    parts = line.split('="')[1].strip('";').split(',')
                    
                    if "int_" in key and len(parts) >= 4:
                        name = parts[0]
                        price = float(parts[1])
                        pct = float(parts[3])
                        indices_data[name] = {"pct": pct, "price": price, "status": MarketStatus.NORMAL if price > 0 else MarketStatus.SUSPENDED}
                    elif "b_HSI" in key and len(parts) >= 7:
                        indices_data["恒生指数"] = {"pct": float(parts[6]), "price": float(parts[1]), "status": MarketStatus.NORMAL}
                except (IndexError, ValueError) as e:
                    errors.append(f"解析外盘数据失败: {e}")
        except Exception as e:
            errors.append(f"获取外盘指数异常: {e}")
            
        return indices_data, errors

class MacroJudgmentEngine:
    @staticmethod
    def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        avg_loss = avg_loss.replace(0, 1e-10)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def get_judgments() -> tuple[dict, list]:
        result = {"macro": [], "us_tech": "", "cn_tech": "", "risk_alert": ""}
        errors = []
        
        cache_file = f"cache_{_today_str()}.pkl"
        yf_data = None
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    yf_data = pickle.load(f)
                log.info("使用缓存的 yfinance 数据")
            except Exception as e:
                errors.append(f"加载缓存失败: {e}")
        
        try:
            import yfinance as yf
            
            proxy = os.environ.get('YF_PROXY')
            download_kwargs = {"period": "6mo"}
            if proxy:
                download_kwargs["proxy"] = proxy
                log.info(f"使用代理: {proxy}")
            
            if yf_data is None:
                tickers = yf.Tickers("^TNX ^VIX ^SKEW HG=F GC=F CL=F ^GSPC 000300.SS")
                hist = tickers.history(**download_kwargs)
                yf_data = hist['Close']
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(yf_data, f)
                    log.info("保存 yfinance 数据到缓存")
                except Exception as e:
                    errors.append(f"保存缓存失败: {e}")
            
            close_df = yf_data

            def get_last(ticker: str) -> float:
                s = close_df[ticker].dropna()
                return s.iloc[-1] if not s.empty else 0.0

            def get_mtm(ticker: str, days: int = 5) -> float:
                s = close_df[ticker].dropna()
                return s.iloc[-1] - s.iloc[-days-1] if len(s) > days else 0.0
            
            def get_mtm_pct(ticker: str, days: int = 5) -> float:
                s = close_df[ticker].dropna()
                if len(s) > days and s.iloc[-days-1] != 0:
                    return (s.iloc[-1] / s.iloc[-days-1] - 1) * 100
                return 0.0

            def get_ma_trend(ticker: str) -> tuple:
                s = close_df[ticker].dropna()
                if len(s) < 30:
                    return "样本不足", "可用交易日不足"
                if len(s) < 60:
                    return "数据有限", "数据量不足60日"
                
                ma5 = s.rolling(5).mean().iloc[-1]
                ma20 = s.rolling(20).mean().iloc[-1]
                ma60 = s.rolling(60).mean().iloc[-1]
                close = s.iloc[-1]
                
                max_ma, min_ma = max(ma5, ma20, ma60), min(ma5, ma20, ma60)
                spread = (max_ma - min_ma) / min_ma
                
                if spread < 0.02:
                    return "均线粘连", "面临方向性变盘"
                elif ma5 > ma20 > ma60:
                    return ("三线开花", "全面多头排列") if close > ma5 else ("多头排列", "短期回踩")
                elif ma5 < ma20 < ma60:
                    return ("空头瀑布", "下行趋势加速") if close < ma5 else ("空头排列", "超跌反弹")
                elif ma60 > ma20 and ma5 > ma20:
                    return "筑底反弹", "短期均线拐头向上"
                else:
                    return "震荡分化", "长短均线方向不一"

            hg, gc, cl = get_last('HG=F'), get_last('GC=F'), get_last('CL=F')
            vix, skew, tnx = get_last('^VIX'), get_last('^SKEW'), get_last('^TNX')
            
            cnh = 7.2
            try:
                res_cnh = safe_request_get("https://hq.sinajs.cn/list=fx_susdcny", timeout=5)
                if res_cnh:
                    cnh = float(res_cnh.text.split('="')[1].split(',')[1])
            except Exception:
                errors.append("离岸人民币获取失败")
            
            cgr = (hg / gc) * 100 if gc else 0
            
            result["macro"].append(
                f"**1. 大宗与汇率**\n> 纽约期金 **{gc:.1f}** | WTI原油 **{cl:.1f}**\n"
                f"> 离岸人民币 **{cnh:.4f}**"
                f"{' 🔴 贬值承压' if cnh > 7.25 else ' 🟢 升值' if cnh < 7.1 else ''}"
                f"\n> 铜金比 **{cgr:.4f}**"
            )
            
            risk_parts = []
            opt_parts = [f"**2. 期权与黑天鹅指标**"]
            
            opt_parts.append(f"> 美10年期国债收益率 **{tnx:.3f}%**"
                f"{' 🔴 高企' if tnx > 4.2 else ' 🟢 回落' if tnx < 3.8 else ''}")
            
            opt_parts.append(f"> VIX恐慌指数 **{vix:.2f}**"
                f"{' 🟢 极度贪婪' if vix < 15 else ' 🔴 对冲启动' if vix > 20 else ''}")
            if vix > 25:
                risk_parts.append("VIX突破25")
            
            opt_parts.append(f"> SKEW黑天鹅指数 **{skew:.1f}**"
                f"{' ⚠️ 尾部风险' if skew > 135 else ''}")
            if skew > 135:
                risk_parts.append("SKEW突破135")
            
            if risk_parts:
                result["risk_alert"] = f"⚠️ **紧急避险提示**: {', '.join(risk_parts)}！"
            
            result["macro"].append("\n".join(opt_parts))
            
            csi300, gspc = close_df['000300.SS'].dropna(), close_df['^GSPC'].dropna()
            csi300_rsi = MacroJudgmentEngine.calc_rsi(csi300).iloc[-1] if not csi300.empty else 50
            gspc_rsi = MacroJudgmentEngine.calc_rsi(gspc).iloc[-1] if not gspc.empty else 50
            
            csi300_mtm, csi300_mtm_pct = get_mtm('000300.SS'), get_mtm_pct('000300.SS')
            gspc_mtm, gspc_mtm_pct = get_mtm('^GSPC'), get_mtm_pct('^GSPC')
            
            csi300_name, csi300_desc = get_ma_trend('000300.SS')
            gspc_name, gspc_desc = get_ma_trend('^GSPC')
            
            result["us_tech"] = f"> 📊 标普500 MTM **{gspc_mtm:+.2f}** ({gspc_mtm_pct:+.2f}%, RSI: {gspc_rsi:.1f})\n> 📈 趋势: **{gspc_name}** - {gspc_desc}"
            result["cn_tech"] = f"> 📊 沪深300 MTM **{csi300_mtm:+.2f}** ({csi300_mtm_pct:+.2f}%, RSI: {csi300_rsi:.1f})\n> 📈 趋势: **{csi300_name}** - {csi300_desc}"

        except ImportError:
            errors.append("yfinance 未安装")
            result["macro"].append("> <font color=\"#8c8c8c\">yfinance未安装</font>")
        except Exception as e:
            errors.append(f"宏观研判异常: {e}")
            result["macro"].append("> <font color=\"#8c8c8c\">引擎异常</font>")

        return result, errors

class NewsDigest:
    @staticmethod
    def score_news(title: str, news_time: datetime = None) -> float:
        score = 0
        
        block_words = ["辞职", "离职", "减持", "亏损", "立案", "退市", "违规", "跌停", "暴跌", "不及预期", "恶化"]
        if any(w in title for w in block_words):
            return -1
        
        low_value = {"早报": -2, "必读": -2, "提示性公告": -1, "互动平台": -3, "董事": -2, "股东大会": -2, "高管": -1, "聘任": -1, "例行": -2}
        for word, penalty in low_value.items():
            if word in title:
                score += penalty
        
        t1_words = ["发改委", "工信部", "央行", "国务院", "新规", "印发", "降准", "降息", "证监会", "政治局", "重磅", "刺激", "利好", "支持"]
        for w in t1_words:
            if w in title:
                score += 10
        
        t2_words = ["超预期", "指引", "订单", "需求爆发", "上调", "产能", "供不应求", "扭亏", "净利", "商业化", "突破", "暴增", "中标", "合作", "发布", "研发", "获批", "新高"]
        for w in t2_words:
            if w in title:
                score += 5
        
        t3_words = ["新能源", "人工智能", "AI", "算力", "半导体", "芯片", "光伏", "储能", "锂电", "数据中心", "云计算", "数字经济", "国产替代", "高端制造", "一带一路", "国企改革"]
        for w in t3_words:
            if w in title:
                score += 3
        
        neg_trend = ["下降", "走低", "回落", "下滑", "大跌", "收跌", "低迷"]
        for w in neg_trend:
            if w in title:
                score -= 3
        
        if news_time:
            hours_old = (datetime.now() - news_time).total_seconds() / 3600
            decay = max(0.5, 1 - hours_old * 0.1)
            score *= decay
        
        return score

    @staticmethod
    def get_news(limit: int = 10) -> tuple[list, list]:
        news_list = []
        scored_news = []
        errors = []
        
        token = os.environ.get("TUSHARE_TOKEN", "").strip()
        if token:
            try:
                import tushare as ts
                pro = ts.pro_api(token)
                df = pro.news(src='cls', limit=limit+80)
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        try:
                            news_time = pd.to_datetime(row['datetime'])
                            time_str = news_time.strftime('%H:%M')
                        except Exception:
                            news_time, time_str = None, ''
                        title = row['title'] if row['title'] else row['content'][:50]+"..."
                        score = NewsDigest.score_news(title, news_time)
                        if score > 0:
                            scored_news.append((score, row['datetime'], time_str, title))
                    
                    if scored_news:
                        scored_news.sort(key=lambda x: x[0], reverse=True)
                        for _, _, time_str, title in scored_news[:limit]:
                            prefix = f"> **[{time_str}]** " if time_str else "> "
                            news_list.append(prefix + title)
                        return news_list, errors
            except Exception as e:
                errors.append(f"Tushare失败: {e}")

        try:
            url = "https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2509&k=&num=80&page=1"
            res = safe_request_get(url, timeout=8)
            if res:
                data = res.json().get('result', {}).get('data', [])
                for doc in data:
                    title = doc.get('title', '')
                    try:
                        news_time = datetime.fromtimestamp(int(doc['ctime']))
                        time_str = news_time.strftime('%H:%M')
                    except Exception:
                        news_time, time_str = None, ''
                    score = NewsDigest.score_news(title, news_time)
                    if score > 0:
                        scored_news.append((score, doc['ctime'], time_str, title))
                
                if scored_news:
                    scored_news.sort(key=lambda x: x[0], reverse=True)
                    for _, _, time_str, title in scored_news[:limit]:
                        news_list.append(f"> **[{time_str}]** {title}")
                    return news_list, errors
        except Exception as e:
            errors.append(f"新浪新闻失败: {e}")
        
        return news_list, errors

# ═════════════════════════════════════════════════════════════════════════════
# 数据采集器
# ═════════════════════════════════════════════════════════════════════════════
class DataCollector:
    @staticmethod
    def collect_all() -> MarketData:
        data = MarketData()
        
        def fetch_ashare():
            return MacroBrain.get_ashare_indices()
        
        def fetch_global():
            return MacroBrain.get_global_indices()
        
        def fetch_judgments():
            return MacroJudgmentEngine.get_judgments()
        
        def fetch_northbound():
            try:
                return fetch_northbound_flow()
            except Exception:
                return (0.0, "")
        
        def fetch_news():
            return NewsDigest.get_news(limit=12)
        
        def fetch_hot():
            try:
                return fetch_hot_sectors()
            except Exception:
                return {}
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                'ashare': executor.submit(fetch_ashare),
                'global': executor.submit(fetch_global),
                'judgments': executor.submit(fetch_judgments),
                'northbound': executor.submit(fetch_northbound),
                'news': executor.submit(fetch_news),
                'hot': executor.submit(fetch_hot),
            }
            
            for name, future in futures.items():
                try:
                    result = future.result(timeout=30)
                    if name == 'ashare':
                        data.ashare_indices, warns = result
                        data.status["warnings"].extend([f"A股: {w}" for w in warns])
                    elif name == 'global':
                        data.global_indices, warns = result
                        data.status["warnings"].extend([f"外盘: {w}" for w in warns])
                    elif name == 'judgments':
                        data.judgments, warns = result
                        data.status["warnings"].extend([f"宏观: {w}" for w in warns])
                    elif name == 'northbound':
                        data.northbound_flow = result
                    elif name == 'news':
                        data.news, warns = result
                        data.status["warnings"].extend([f"新闻: {w}" for w in warns])
                    elif name == 'hot':
                        data.hot_sectors = result
                except FuturesTimeoutError:
                    data.status["errors"].append(f"{name} 获取超时")
                except Exception as e:
                    data.status["errors"].append(f"{name} 异常")
        
        return data

# ═════════════════════════════════════════════════════════════════════════════
# 渲染器
# ═════════════════════════════════════════════════════════════════════════════
class BriefingRenderer:
    @staticmethod
    def render(data: MarketData) -> str:
        date_str = _today_str()
        lines = []
        
        lines.append(f"## 🤖 AI 每日市场简报\n*{date_str}*\n")
        
        if data.judgments.get("risk_alert"):
            lines.append(f"### {data.judgments['risk_alert']}\n")
        
        if data.status["errors"]:
            lines.append(f"> ⚠️ **数据异常**: {', '.join(data.status['errors'][:3])}\n")
        
        lines.append("---\n### 🌍 大类资产与衍生品\n")
        if data.judgments.get("macro"):
            lines.append("\n\n".join(data.judgments["macro"]))
        else:
            lines.append("> <font color=\"#8c8c8c\">研判引擎暂无数据</font>")
        
        lines.append("\n---\n### 🇺🇸 美股大盘")
        for name, item in data.global_indices.items():
            if name == "恒生指数":
                continue
            is_us = name in ["纳斯达克", "标普500", "道琼斯"]
            status_marker = " [休市]" if item.get("status") == MarketStatus.SUSPENDED else ""
            lines.append(f"- **{name}**: {item['price']:.2f}{status_marker} {format_dingtalk_pct(item['pct'], is_us)}")
        
        if data.judgments.get("us_tech"):
            lines.append(data.judgments["us_tech"])
        
        lines.append("\n---\n### 🇭🇰 港股大盘")
        if "恒生指数" in data.global_indices:
            hsi = data.global_indices["恒生指数"]
            lines.append(f"- **恒生指数**: {hsi['price']:.2f} {format_dingtalk_pct(hsi['pct'], False)}")
        else:
            lines.append("- <font color=\"#8c8c8c\">暂无数据</font>")
        
        lines.append("\n---\n### 🇨🇳 A股大盘")
        for name, item in data.ashare_indices.items():
            markers = {"suspended": " [休市]", "limit_up": " 🔺涨停", "limit_down": " 🔻跌停"}
            marker = markers.get(item.get("status"), "")
            lines.append(f"- **{name}**: {item['price']:.2f}{marker} {format_dingtalk_pct(item['pct'], False)}")
        
        if data.judgments.get("cn_tech"):
            lines.append(data.judgments["cn_tech"])
        
        flow_amt, flow_msg = data.northbound_flow
        if flow_msg:
            lines.append(f"\n> 💰 **北向资金** ({flow_amt:+.1f}亿元)")
        
        if data.hot_sectors:
            sectors = list(set(data.hot_sectors.values()))[:3]
            lines.append(f"\n> 🔥 **热点板块**: {', '.join(sectors)}")
        
        lines.append("\n---\n### 📰 核心投研资讯\n")
        if data.news:
            lines.append("\n\n".join(data.news))
        else:
            lines.append("> <font color=\"#8c8c8c\">暂无重大新闻</font>")
        
        lines.append("\n---\n*<font color=\"#8c8c8c\">Antigravity 机构级量化引擎</font>*")
        
        content = "\n".join(lines)
        
        if data.status["warnings"]:
            warning_note = f"\n\n> ⚠️ {', '.join(data.status['warnings'][:2])}..."
            if len(content) + len(warning_note) < DINGTALK_MAX_LEN:
                content += warning_note
        
        return content

def truncate_content(content: str, max_len: int = DINGTALK_MAX_LEN) -> str:
    if len(content) <= max_len:
        return content
    
    trunc_marker = "\n---\n*<font color=\"#8c8c8c\">"
    idx = content.rfind(trunc_marker, 0, max_len)
    if idx != -1:
        return content[:idx] + "\n\n*⚠️ 内容过长已截断*"
    return content[:max_len] + "\n\n*⚠️ 内容过长已截断*"

# ═════════════════════════════════════════════════════════════════════════════
# 推送模块
# ═════════════════════════════════════════════════════════════════════════════
def send_dingtalk(content: str):
    webhook = os.environ.get('DINGTALK_WEBHOOK')
    if not webhook:
        log.warning("未配置 DINGTALK_WEBHOOK，仅输出到控制台")
        print(content)
        return
    
    segments = []
    while len(content) > DINGTALK_MAX_LEN:
        split_at = content.rfind('\n', 0, DINGTALK_MAX_LEN)
        if split_at == -1:
            split_at = DINGTALK_MAX_LEN
        segments.append(content[:split_at])
        content = content[split_at:].lstrip('\n')
    segments.append(content)
    
    for i, seg in enumerate(segments):
        title = f'🤖 每日市场简报 ({i+1}/{len(segments)})' if len(segments) > 1 else '🤖 每日市场简报'
        try:
            payload = {'msgtype': 'markdown', 'markdown': {'title': title, 'text': seg}}
            res = requests.post(webhook, json=payload, timeout=10)
            res.raise_for_status()
            res_dict = res.json()
            if res_dict.get('errcode', 0) != 0:
                log.error(f"钉钉推送失败: {res_dict}")
            else:
                log.info(f"✅ 简报片段 ({i+1}/{len(segments)}) 推送成功")
        except Exception as e:
            log.error(f"❌ 推送异常: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# 主入口
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        file_handler = logging.FileHandler(f'briefing_{_today_str()}.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
    except Exception:
        pass
    
    is_trading, trading_msg = is_trading_hours()
    if not is_trading:
        log.info(f"非交易时段: {trading_msg}，跳过执行")
    
    issues = health_check()
    for issue in issues:
        if issue.startswith("❌"):
            log.error(issue)
        else:
            log.warning(issue)
    
    try:
        data = DataCollector.collect_all()
        report = BriefingRenderer.render(data)
        log.info(f"简报长度: {len(report)} 字符")
        
        if len(report) > DINGTALK_MAX_LEN:
            report = truncate_content(report)
        
        send_dingtalk(report)
    except Exception as e:
        log.critical(f"简报生成崩溃: {e}", exc_info=True)
