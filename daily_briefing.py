
import os
import json
import time
import pickle
import logging
import logging.handlers
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Tuple
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
DINGTALK_MAX_LEN = 18000

class Color:
    RED = "#F04864"
    GREEN = "#2fc25b"
    GRAY = "#8c8c8c"
    UP_CN = RED
    DOWN_CN = GREEN
    UP_US = GREEN
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
    extra: dict = field(default_factory=dict)

_http_session: Optional[requests.Session] = None

def get_http_session() -> requests.Session:
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        retries = Retry(total=2, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
        _http_session.mount('https://', HTTPAdapter(max_retries=retries))
        _http_session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
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

class MarketStatus:
    NORMAL = "normal"
    SUSPENDED = "suspended"
    LIMIT_UP = "limit_up"
    LIMIT_DOWN = "limit_down"
    ERROR = "error"

# ═════════════════════════════════════════════════════════════════════════════
# 数据源模块
# ═════════════════════════════════════════════════════════════════════════════
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
                    
                    name_map = {"沪深300": "沪深300", "创业板指": "创业板指", 
                              "上证指数": "上证指数", "深证成指": "深证成指"}
                    target_name = name_map.get(name, name)
                    
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
                        indices_data[name] = {"pct": pct, "price": price, 
                                             "status": MarketStatus.NORMAL if price > 0 else MarketStatus.SUSPENDED}
                    elif "b_HSI" in key and len(parts) >= 7:
                        indices_data["恒生指数"] = {"pct": float(parts[6]), "price": float(parts[1]), 
                                                    "status": MarketStatus.NORMAL}
                except (IndexError, ValueError) as e:
                    errors.append(f"解析外盘数据失败: {e}")
        except Exception as e:
            errors.append(f"获取外盘指数异常: {e}")
        return indices_data, errors

    @staticmethod
    def get_usd_index() -> tuple[Optional[float], list]:
        errors = []
        try:
            res = safe_request_get("https://hq.sinajs.cn/list=hf_USDX", timeout=5)
            if res:
                parts = res.text.split('="')[1].strip('";').split(',')
                if len(parts) >= 2:
                    return float(parts[1]), errors
        except Exception as e:
            errors.append(f"美元指数获取失败: {e}")
        return None, errors

    @staticmethod
    def get_ashare_breadth() -> tuple[dict, list]:
        breadth = {"up_count": 0, "down_count": 0, "limit_up_count": 0, "limit_down_count": 0, "volume": 0}
        errors = []
        try:
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                breadth["up_count"] = int((df['涨跌幅'] > 0).sum())
                breadth["down_count"] = int((df['涨跌幅'] < 0).sum())
                breadth["limit_up_count"] = int((df['涨跌幅'] >= 9.5).sum())
                breadth["limit_down_count"] = int((df['涨跌幅'] <= -9.5).sum())
                if '成交额' in df.columns:
                    breadth["volume"] = df['成交额'].sum()
        except Exception as e:
            errors.append(f"涨跌家数获取失败: {e}")
        return breadth, errors

    @staticmethod
    def get_southbound_flow() -> tuple[float, str, list]:
        errors = []
        try:
            df = ak.stock_em_hsgt_south_net_flow_in(indicator="港股通（沪）")
            if df is not None and not df.empty:
                col = 'value' if 'value' in df.columns else df.columns[-1]
                today_flow = float(df.iloc[-1][col]) / 1e8
                return today_flow, "南向资金", errors
        except Exception as e:
            errors.append(f"南向资金获取失败: {e}")
        return 0.0, "", errors

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
        result = {"macro": [], "us_tech": "", "cn_tech": "", "risk_alert": "", "key_levels": {}, "interpretations": []}
        errors = []
        
        cache_file = f"cache_{_today_str()}.pkl"
        yf_data = None
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    yf_data = pickle.load(f)
                log.info("使用缓存的 yfinance 数据")
            except Exception:
                pass
        
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
                except Exception:
                    pass
            
            close_df = yf_data

            def get_last(ticker: str) -> float:
                s = close_df[ticker].dropna()
                return s.iloc[-1] if not s.empty else 0.0

            def get_mtm_pct(ticker: str, days: int = 5) -> float:
                s = close_df[ticker].dropna()
                if len(s) > days and s.iloc[-days-1] != 0:
                    return (s.iloc[-1] / s.iloc[-days-1] - 1) * 100
                return 0.0

            def get_ma_trend(ticker: str) -> tuple:
                s = close_df[ticker].dropna()
                if len(s) < 30:
                    return "样本不足", "可用数据不足", {}, 0.5
                if len(s) < 60:
                    return "数据有限", "历史数据不足60日", {}, 0.6
                
                ma5 = s.rolling(5).mean().iloc[-1]
                ma20 = s.rolling(20).mean().iloc[-1]
                ma60 = s.rolling(60).mean().iloc[-1]
                close = s.iloc[-1]
                
                max_ma, min_ma = max(ma5, ma20, ma60), min(ma5, ma20, ma60)
                spread = (max_ma - min_ma) / min_ma
                
                key_levels = {
                    "close": close,
                    "ma5": ma5, "ma20": ma20, "ma60": ma60,
                    "resistance": close * 1.02,
                    "support": close * 0.98
                }
                confidence = 0.7
                
                if spread < 0.02:
                    return "均线粘连", "面临方向选择", key_levels, confidence
                elif ma5 > ma20 > ma60:
                    if close > ma5:
                        return "三线多头", "偏强但已接近布林上轨，追高需谨慎", key_levels, confidence
                    else:
                        return "多头排列", "中期向上但短期回踩，关注均线支撑", key_levels, confidence
                elif ma5 < ma20 < ma60:
                    if close < ma5:
                        return "空头排列", "趋势向下，严控仓位", key_levels, confidence
                    else:
                        return "空头反弹", "超跌反弹性质，持续性待观察", key_levels, confidence * 0.8
                elif ma60 > ma20 and ma5 > ma20:
                    return "筑底反弹", "短期企稳，中期仍需确认", key_levels, confidence * 0.6
                else:
                    return "震荡分化", "方向不明，等待突破确认", key_levels, confidence * 0.5

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
            
            interpretations = []
            
            cgr_interp = ""
            if cgr > 5.0:
                cgr_interp = f"铜金比 **{cgr:.2f}**，处于偏高区域，经济扩张预期偏强"
            elif cgr > 4.0:
                cgr_interp = f"铜金比 **{cgr:.2f}**，处于中性区间，工业需求复苏尚不稳固"
            else:
                cgr_interp = f"铜金比 **{cgr:.2f}**，处于偏低区域，暗示需求侧仍有压力"
            interpretations.append(cgr_interp)
            
            vix_interp = ""
            if vix < 15:
                vix_interp = f"VIX **{vix:.1f}**，极度贪婪，期权保护成本极低，可能过于乐观"
            elif vix < 20:
                vix_interp = f"VIX **{vix:.1f}**，偏贪婪，市场整体风险偏好较高"
            else:
                vix_interp = f"VIX **{vix:.1f}**，偏谨慎，大资金已在布局对冲，短期需注意波动"
            interpretations.append(vix_interp)
            
            tnx_interp = ""
            if tnx > 4.5:
                tnx_interp = f"美10Y收益率 **{tnx:.2f}%**，高位，对全球科技股估值形成持续压制"
            elif tnx > 4.2:
                tnx_interp = f"美10Y收益率 **{tnx:.2f}%**，偏高，风险资产仍承压"
            elif tnx > 3.8:
                tnx_interp = f"美10Y收益率 **{tnx:.2f}%**，中性，符合历史均值"
            else:
                tnx_interp = f"美10Y收益率 **{tnx:.2f}%**，偏低，成长股流动性环境友好"
            interpretations.append(tnx_interp)
            
            cnh_interp = ""
            if cnh > 7.3:
                cnh_interp = f"离岸人民币 **{cnh:.4f}**，弱势，外资流出压力较大"
            elif cnh > 7.2:
                cnh_interp = f"离岸人民币 **{cnh:.4f}**，偏弱，关注7.3整数关口"
            elif cnh > 7.1:
                cnh_interp = f"离岸人民币 **{cnh:.4f}**，基本稳定，外资流向平稳"
            else:
                cnh_interp = f"离岸人民币 **{cnh:.4f}**，偏强，外资流入窗口"
            interpretations.append(cnh_interp)
            
            result["macro"].append("\n".join(interpretations))
            
            risk_parts = []
            if vix > 25:
                risk_parts.append(f"VIX突破25（当前{vix:.1f}），市场恐慌情绪升温")
            if skew > 140:
                risk_parts.append(f"SKEW突破140（当前{skew:.0f}），尾部风险需警惕")
            if tnx > 4.5:
                risk_parts.append("美债收益率高位，持续压制风险偏好")
            
            if risk_parts:
                result["risk_alert"] = "⚠️ **风险提示**：" + "；".join(risk_parts) + "。"
            
            csi300_name, csi300_desc, csi300_levels, csi300_conf = get_ma_trend('000300.SS')
            gspc_name, gspc_desc, gspc_levels, gspc_conf = get_ma_trend('^GSPC')
            
            result["key_levels"] = {
                "csi300": csi300_levels,
                "gspc": gspc_levels
            }
            
            csi300_rsi = MacroJudgmentEngine.calc_rsi(close_df['000300.SS'].dropna()).iloc[-1] if not close_df['000300.SS'].dropna().empty else 50
            gspc_rsi = MacroJudgmentEngine.calc_rsi(close_df['^GSPC'].dropna()).iloc[-1] if not close_df['^GSPC'].dropna().empty else 50
            
            result["us_tech"] = (
                f"**标普500** {format_dingtalk_pct(get_mtm_pct('^GSPC'), True)}\n"
                f"> 趋势：**{gspc_name}**（置信度约{int(gspc_conf*100)}%）\n"
                f"> 信号：{gspc_desc}\n"
                f"> RSI：{gspc_rsi:.0f} | 收盘：{gspc_levels.get('close', 0):.0f}\n"
                f"> 参考：阻力 {gspc_levels.get('resistance', 0):.0f} / 支撑 {gspc_levels.get('support', 0):.0f}"
            )
            
            result["cn_tech"] = (
                f"**沪深300** {format_dingtalk_pct(get_mtm_pct('000300.SS'), False)}\n"
                f"> 趋势：**{csi300_name}**（置信度约{int(csi300_conf*100)}%）\n"
                f"> 信号：{csi300_desc}\n"
                f"> RSI：{csi300_rsi:.0f} | 收盘：{csi300_levels.get('close', 0):.0f}\n"
                f"> 参考：阻力 {csi300_levels.get('resistance', 0):.0f} / 支撑 {csi300_levels.get('support', 0):.0f}"
            )

        except ImportError:
            errors.append("yfinance 未安装")
        except Exception as e:
            errors.append(f"宏观研判异常: {e}")
            result["macro"].append("> <font color=\"#8c8c8c\">宏观数据获取异常</font>")

        return result, errors

class NewsDigest:
    @staticmethod
    def score_news(title: str, news_time: datetime = None) -> float:
        score = 0
        
        block_words = ["辞职", "离职", "跌停", "暴跌", "退市", "违规", "立案"]
        if any(w in title for w in block_words):
            return -1
        
        low_value = {"早报": -2, "必读": -2, "提示性公告": -1, "互动平台": -3, "股东大会": -2}
        for word, penalty in low_value.items():
            if word in title:
                score += penalty
        
        t1_words = ["发改委", "工信部", "央行", "国务院", "证监会", "政治局", "降准", "降息", "重磅", "刺激"]
        for w in t1_words:
            if w in title:
                score += 10
        
        t2_words = ["超预期", "指引", "订单", "需求爆发", "上调", "产能", "供不应求", "扭亏", "净利", "暴增", "中标", "合作", "发布", "研发", "获批", "新高"]
        for w in t2_words:
            if w in title:
                score += 5
        
        t3_words = ["新能源", "人工智能", "AI", "算力", "半导体", "芯片", "光伏", "储能", "数字经济", "国产替代", "高端制造"]
        for w in t3_words:
            if w in title:
                score += 3
        
        if news_time:
            hours_old = (datetime.now() - news_time).total_seconds() / 3600
            decay = max(0.5, 1 - hours_old * 0.1)
            score *= decay
        
        return score

    @staticmethod
    def generate_summary(title: str) -> str:
        summaries = []
        
        if "工信部" in title or "发改委" in title or "国务院" in title:
            if "人形机器人" in title:
                summaries.append("政策定调产业化加速，关键零部件（减速器/传感器）及整机龙头有望受益")
            elif "新能源" in title or "光伏" in title:
                summaries.append("行业获政策加持，中长期需求预期上调")
            elif "半导体" in title or "芯片" in title:
                summaries.append("国产替代进程加速，关注设备/材料环节突破")
            elif "人工智能" in title or "AI" in title:
                summaries.append("顶层支持明确，算力基础设施和应用场景双重受益")
        
        if "超预期" in title or "暴增" in title:
            summaries.append("基本面边际改善，盈利预测存在上调空间")
        elif "净利" in title and ("增" in title or "涨" in title):
            summaries.append("业绩表现亮眼，估值性价比凸显")
        
        if "中标" in title or "订单" in title:
            summaries.append("需求侧持续验证，订单落地支撑业绩预期")
        elif "合作" in title:
            summaries.append("产业链协同加深，竞争优势有望扩大")
        
        if summaries:
            return f"→ {summaries[0]}"
        return ""

    @staticmethod
    def get_news(limit: int = 8) -> tuple[list, list]:
        news_list = []
        scored_news = []
        errors = []
        
        token = os.environ.get('TUSHARE_TOKEN', '').strip()
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
                            summary = NewsDigest.generate_summary(title)
                            if summary:
                                news_list.append(f"> **[{time_str}]** {title}\n> {summary}")
                            else:
                                news_list.append(f"> **[{time_str}]** {title}")
                        return news_list, errors
            except Exception as e:
                errors.append(f"Tushare新闻失败: {e}")

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
                        summary = NewsDigest.generate_summary(title)
                        if summary:
                            news_list.append(f"> **[{time_str}]** {title}\n> {summary}")
                        else:
                            news_list.append(f"> **[{time_str}]** {title}")
                    return news_list, errors
        except Exception as e:
            errors.append(f"新浪新闻失败: {e}")
        
        return news_list, errors

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
            return NewsDigest.get_news(limit=8)
        
        def fetch_hot():
            try:
                return fetch_hot_sectors()
            except Exception:
                return {}
        
        def fetch_breadth():
            return MacroBrain.get_ashare_breadth()
        
        def fetch_usd():
            return MacroBrain.get_usd_index()
        
        def fetch_southbound():
            return MacroBrain.get_southbound_flow()
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                'ashare': executor.submit(fetch_ashare),
                'global': executor.submit(fetch_global),
                'judgments': executor.submit(fetch_judgments),
                'northbound': executor.submit(fetch_northbound),
                'news': executor.submit(fetch_news),
                'hot': executor.submit(fetch_hot),
                'breadth': executor.submit(fetch_breadth),
                'usd': executor.submit(fetch_usd),
                'southbound': executor.submit(fetch_southbound)
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
                    elif name == 'breadth':
                        data.extra['breadth'] = result[0]
                        data.status["warnings"].extend([f"宽度: {w}" for w in result[1]])
                    elif name == 'usd':
                        data.extra['usd'] = result[0]
                        data.status["warnings"].extend([f"美元: {w}" for w in result[1]])
                    elif name == 'southbound':
                        data.extra['southbound'] = result[0]
                        data.extra['southbound_desc'] = result[1]
                        data.status["warnings"].extend([f"南向: {w}" for w in result[2]])
                except FuturesTimeoutError:
                    data.status["errors"].append(f"{name} 获取超时")
                except Exception as e:
                    data.status["errors"].append(f"{name} 异常")
        
        return data

class BriefingRenderer:
    @staticmethod
    def generate_key_points(data: MarketData) -> str:
        points = []
        
        judgments = data.judgments
        if "macro" in judgments and judgments["macro"]:
            macro_text = "\n".join(judgments["macro"])
            if "压制" in macro_text or "高位" in macro_text:
                points.append("美债收益率偏高，对全球风险资产形成压制")
            elif "偏低" in macro_text:
                points.append("美债收益率回落，流动性环境友好")
        
        flow_amt, _ = data.northbound_flow
        if flow_amt > 50:
            points.append(f"北向资金大幅净流入（+{flow_amt:.0f}亿），外资情绪回暖")
        elif flow_amt < -50:
            points.append(f"北向资金净流出（{flow_amt:.0f}亿），外资偏谨慎")
        
        southbound = data.extra.get('southbound', 0)
        if southbound > 30:
            points.append(f"南向资金净流入（+{southbound:.0f}亿），内地资金抄底港股")
        
        breadth = data.extra.get('breadth', {})
        if breadth:
            up_count = breadth.get('up_count', 0)
            down_count = breadth.get('down_count', 0)
            total = up_count + down_count
            if total > 0 and up_count / total > 0.65:
                limit_up = breadth.get('limit_up_count', 0)
                points.append(f"A股赚钱效应较强（涨停{limit_up}家，上涨{up_count}家）")
            elif total > 0 and up_count / total < 0.35:
                limit_down = breadth.get('limit_down_count', 0)
                points.append(f"A股整体偏弱（跌停{limit_down}家）")
        
        if judgments.get("risk_alert"):
            points.append(judgments["risk_alert"].replace("⚠️ **风险提示**：", ""))
        
        if data.hot_sectors:
            top_sectors = list(set(data.hot_sectors.values()))[:3]
            if top_sectors:
                points.append(f"热点板块轮动：{', '.join(top_sectors)}")
        
        if not points:
            points.append("市场整体平稳，无重大异常信号")
        
        return "\n".join([f"- {p}" for p in points])

    @staticmethod
    def render(data: MarketData) -> str:
        date_str = _today_str()
        now = datetime.now(TZ_BJS)
        lines = []
        
        lines.append(f"## 🤖 AI 每日市场简报\n*{date_str} {now.strftime('%H:%M')} 生成*\n")
        
        time_note = "> 美股数据截至隔夜收盘 | A股数据截至昨日收盘"
        lines.append(time_note)
        
        lines.append("\n---\n### 🎯 核心结论\n")
        key_points = BriefingRenderer.generate_key_points(data)
        lines.append(key_points)
        
        lines.append("\n---\n### 📊 宏观环境\n")
        if data.judgments.get("macro"):
            lines.append("\n".join(data.judgments["macro"]))
        else:
            lines.append("> <font color=\"#8c8c8c\">暂无宏观数据</font>")
        
        if data.extra.get('usd'):
            lines.append(f"\n> 美元指数 **{data.extra['usd']:.2f}**")
        
        lines.append("\n---\n### 🇺🇸 美股技术面\n")
        if data.judgments.get("us_tech"):
            lines.append(data.judgments["us_tech"])
        
        lines.append("\n---\n### 🇭🇰 港股大盘\n")
        if "恒生指数" in data.global_indices:
            hsi = data.global_indices["恒生指数"]
            lines.append(f"- **恒生指数**：{hsi['price']:.2f} {format_dingtalk_pct(hsi['pct'], False)}")
        
        southbound = data.extra.get('southbound', 0)
        if southbound != 0:
            if southbound > 30:
                lines.append(f"> 🌊 **南向资金**：南水流入 **+{southbound:.0f}亿**")
            elif southbound < -30:
                lines.append(f"> ❄️ **南向资金**：南水流 **{southbound:.0f}亿**")
            else:
                lines.append(f"> ⚖️ **南向资金**：温和 (**{southbound:+.0f}亿**)")
        
        lines.append("\n---\n### 🇨🇳 A股技术面\n")
        if data.judgments.get("cn_tech"):
            lines.append(data.judgments["cn_tech"])
        
        breadth = data.extra.get('breadth', {})
        if breadth:
            lines.append("\n**市场宽度**")
            up_count = breadth.get('up_count', 0)
            down_count = breadth.get('down_count', 0)
            limit_up = breadth.get('limit_up_count', 0)
            limit_down = breadth.get('limit_down_count', 0)
            total = up_count + down_count
            ratio_str = ""
            if total > 0:
                ratio = up_count / total
                if ratio > 0.6:
                    ratio_str = "偏强"
                elif ratio < 0.4:
                    ratio_str = "偏弱"
                else:
                    ratio_str = "均衡"
            volume = breadth.get('volume', 0)
            volume_str = ""
            if volume > 0:
                volume_str = f" | 成交额 {volume/1e8:.1f}亿"
            
            lines.append(f"> 上涨 **{up_count}** 家 / 下跌 **{down_count}** 家（{ratio_str}）\n> 涨停 **{limit_up}** 家 / 跌停 **{limit_down}** 家{volume_str}")
        
        lines.append("\n---\n### 💰 资金与情绪\n")
        flow_amt, flow_msg = data.northbound_flow
        if flow_msg:
            if flow_amt > 50:
                lines.append(f"> 🌊 **北向资金**：北水大举流入 **+{flow_amt:.0f}亿**")
            elif flow_amt < -50:
                lines.append(f"> ❄️ **北向资金**：北水大幅流出 **{flow_amt:.0f}亿**")
            else:
                lines.append(f"> ⚖️ **北向资金**：温和 (**{flow_amt:+.0f}亿**)")
        
        if data.hot_sectors:
            top_sectors = list(set(data.hot_sectors.values()))[:3]
            lines.append(f"\n> 🔥 **热点板块**：{', '.join(top_sectors)}")
        
        lines.append("\n---\n### 📰 投研资讯精选\n")
        if data.news:
            lines.append("\n\n".join(data.news))
        else:
            lines.append("> <font color=\"#8c8c8c\">暂无高价值资讯</font>")
        
        lines.append("\n---\n### ⚠️ 潜在风险关注\n")
        risk_items = []
        judgments = data.judgments
        if judgments.get("risk_alert"):
            risk_items.append(judgments["risk_alert"].replace("⚠️ **风险提示**：", ""))
        
        if flow_amt < -30:
            risk_items.append("北向资金持续流出，外资避险情绪升温")
        
        breadth = data.extra.get('breadth', {})
        if breadth and breadth.get('limit_down_count', 0) > 30:
            risk_items.append(f"跌停家数偏多（{breadth.get('limit_down_count')}家），市场恐慌情绪待释放")
        
        if risk_items:
            lines.append("\n".join([f"- {r}" for r in risk_items]))
        else:
            lines.append("- 暂无明显风险信号")
        
        if data.status["errors"]:
            lines.append(f"\n> ⚠️ **数据异常**：{', '.join(data.status['errors'][:2])}")
        
        lines.append("\n---\n*<font color=\"#8c8c8c\">Antigravity 量化引擎 | 仅供参考，不构成投资建议</font>*")
        
        content = "\n".join(lines)
        return content

def truncate_content(content: str, max_len: int = DINGTALK_MAX_LEN) -> str:
    if len(content) <= max_len:
        return content
    trunc_marker = "\n---\n*<font color=\"#8c8c8c\">"
    idx = content.rfind(trunc_marker, 0, max_len)
    if idx != -1:
        return content[:idx] + "\n\n*⚠️ 内容过长已截断*"
    return content[:max_len] + "\n\n*⚠️ 内容过长已截断*"

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

def main():
    """主入口函数，供外部调用"""
    is_trading, trading_msg = is_trading_hours()
    log.info(f"当前状态: {trading_msg}")
    
    issues = health_check()
    for issue in issues:
        if issue.startswith("❌"):
            log.error(issue)
        else:
            log.warning(issue)
    
    data = DataCollector.collect_all()
    report = BriefingRenderer.render(data)
    log.info(f"简报长度: {len(report)} 字符")
    
    if len(report) > DINGTALK_MAX_LEN:
        report = truncate_content(report)
    
    send_dingtalk(report)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        file_handler = logging.FileHandler(f'briefing_{_today_str()}.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
    except Exception:
        pass
    
    try:
        main()
    except Exception as e:
        log.critical(f"简报生成崩溃: {e}", exc_info=True)

