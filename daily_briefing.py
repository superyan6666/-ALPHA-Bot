import os
import json
import time
import pickle
import requests
import logging
import logging.handlers
from datetime import datetime
import pandas as pd
import akshare as ak
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

from main import fetch_spot, fetch_hot_sectors, fetch_northbound_flow, Cols, TZ_BJS

log = logging.getLogger(__name__)

# 常量集中定义
SINA_HEADERS = {"Referer": "https://finance.sina.com.cn/"}
DINGTALK_MAX_LENGTH = 18000  # 钉钉消息长度限制，预留缓冲

def _today_str() -> str:
    return datetime.now(TZ_BJS).strftime('%Y-%m-%d')

def format_dingtalk_pct(pct, is_us=False, show_emoji=False):
    if pct > 0:
        color = "#2fc25b" if is_us else "#F04864"
        emoji = ("🟢 " if is_us else "🔴 ") if show_emoji else ""
        return f'<font color="{color}">{emoji}+{pct:.2f}%</font>'
    elif pct < 0:
        color = "#F04864" if is_us else "#2fc25b"
        emoji = ("🔴 " if is_us else "🟢 ") if show_emoji else ""
        return f'<font color="{color}">{emoji}{pct:.2f}%</font>'
    else:
        return f'<font color="#8c8c8c"> 0.00%</font>'

def safe_request_get(url, headers=None, timeout=5, max_retries=2):
    """带重试机制的安全网络请求"""
    headers = headers or {}
    for attempt in range(max_retries):
        try:
            res = requests.get(url, headers=headers, timeout=timeout)
            res.raise_for_status()
            return res
        except Exception as e:
            if attempt < max_retries - 1:
                log.debug(f"请求失败，重试中 ({attempt+1}/{max_retries}): {url}")
                time.sleep(1)
            else:
                log.warning(f"请求失败 ({max_retries}次尝试): {url} - {e}")
                return None

def health_check():
    """健康检查，返回问题列表"""
    checks = []
    # 检查必要的包
    for mod in ['pandas', 'requests']:
        try:
            __import__(mod)
        except ImportError:
            checks.append(f"{mod} 未安装")
    # 可选依赖检查
    for mod in ['yfinance', 'tushare']:
        try:
            __import__(mod)
        except ImportError:
            log.warning(f"{mod} 未安装，部分功能将受限")
    return checks

class MacroBrain:
    @staticmethod
    def get_ashare_indices():
        indices_data = {}
        try:
            url = "https://hq.sinajs.cn/list=s_sh000001,s_sz399001,s_sz399006,s_sz399300"
            res = safe_request_get(url, headers=SINA_HEADERS, timeout=8)
            if res is None:
                return indices_data
            
            for line in res.text.strip().split('\n'):
                if '="' in line:
                    try:
                        parts = line.split('="')[1].strip('";').split(',')
                        if len(parts) >= 5:
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
                            
                            indices_data[target_name] = {"pct": pct, "price": price}
                    except (IndexError, ValueError) as e:
                        log.warning(f"解析A股指数数据失败: {e}")
            log.info(f"新浪A股大盘获取成功: {indices_data}")
        except Exception as e:
            log.warning(f"获取新浪A股大盘失败: {e}")
        return indices_data

    @staticmethod
    def get_global_indices():
        indices_data = {}
        try:
            url = "https://hq.sinajs.cn/list=int_dji,int_nasdaq,int_sp500,b_HSI"
            res = safe_request_get(url, headers=SINA_HEADERS, timeout=8)
            if res is None:
                return indices_data
            
            for line in res.text.strip().split('\n'):
                if '="' in line:
                    try:
                        key = line.split('=')[0]
                        parts = line.split('="')[1].strip('";').split(',')
                        
                        if "int_" in key and len(parts) >= 4:
                            name = parts[0]
                            price = float(parts[1])
                            pct = float(parts[3])
                            indices_data[name] = {"pct": pct, "price": price}
                        elif "b_HSI" in key and len(parts) >= 7:
                            indices_data["恒生指数"] = {"pct": float(parts[6]), "price": float(parts[1])}
                    except (IndexError, ValueError) as e:
                        log.warning(f"解析外盘指数数据失败: {e}")
        except Exception as e:
            log.warning(f"新浪外盘数据失败: {e}")
            
        return indices_data

class MacroJudgmentEngine:
    @staticmethod
    def calc_rsi(series, period=14):
        """计算RSI，带有完善的除零保护"""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # 除零保护
        avg_loss = avg_loss.replace(0, 1e-10)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)
        return rsi

    @staticmethod
    def get_judgments():
        result = {"macro": [], "us_tech": "", "cn_tech": "", "risk_alert": ""}
        
        # 尝试使用缓存
        cache_file = f"cache_{_today_str()}.pkl"
        yf_data = None
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    yf_data = pickle.load(f)
                log.info("使用缓存的 yfinance 数据")
            except Exception as e:
                log.warning(f"加载缓存失败: {e}")
        
        try:
            import yfinance as yf
            
            if yf_data is None:
                tickers = yf.Tickers("^TNX ^VIX ^SKEW HG=F GC=F CL=F ^GSPC 000300.SS")
                hist = tickers.history(period="6mo")
                yf_data = hist['Close']
                # 保存缓存
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(yf_data, f)
                    log.info("保存 yfinance 数据到缓存")
                except Exception as e:
                    log.warning(f"保存缓存失败: {e}")
            
            close_df = yf_data

            def get_last(ticker):
                s = close_df[ticker].dropna()
                return s.iloc[-1] if not s.empty else 0.0

            def get_mtm(ticker, days=5):
                s = close_df[ticker].dropna()
                if len(s) > days:
                    return s.iloc[-1] - s.iloc[-days-1]
                return 0.0
            
            def get_mtm_pct(ticker, days=5):
                """计算动量百分比"""
                s = close_df[ticker].dropna()
                if len(s) > days and s.iloc[-days-1] != 0:
                    return (s.iloc[-1] / s.iloc[-days-1] - 1) * 100
                return 0.0

            def get_ma_trend(ticker):
                s = close_df[ticker].dropna()
                if len(s) < 30:
                    return "样本不足", "可用交易日不足，研判参考性有限"
                if len(s) < 60:
                    return "数据有限", "数据量不足60日，趋势判断仅供参考"
                
                ma5 = s.rolling(5).mean().iloc[-1]
                ma20 = s.rolling(20).mean().iloc[-1]
                ma60 = s.rolling(60).mean().iloc[-1]
                close = s.iloc[-1]
                
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

            hg = get_last('HG=F')
            gc = get_last('GC=F')
            cl = get_last('CL=F')
            vix = get_last('^VIX')
            skew = get_last('^SKEW')
            tnx = get_last('^TNX')
            
            cnh = 7.2  # 默认值
            try:
                res_cnh = safe_request_get("https://hq.sinajs.cn/list=fx_susdcny", headers=SINA_HEADERS, timeout=5)
                if res_cnh:
                    cnh = float(res_cnh.text.split('="')[1].split(',')[1])
                else:
                    log.warning("离岸人民币接口返回空，使用默认值 7.2")
            except Exception as e:
                log.warning(f"获取离岸人民币失败，使用默认值 7.2: {e}")
            
            cgr = (hg / gc) * 100 if gc else 0
            
            comm_msg = f"**1. 大宗与汇率**\n> 纽约期金 **{gc:.1f}** | WTI原油 **{cl:.1f}**"
            cnh_str = f"\n> 离岸人民币 **{cnh:.4f}**"
            if cnh > 7.25: 
                cnh_str += " 🔴 <font color=\"#F04864\">(离岸人民币贬值承压，外资被动流出风险加剧)</font>"
            elif cnh < 7.1:
                cnh_str += " 🟢 <font color=\"#2fc25b\">(离岸人民币升值，外资流入窗口打开)</font>"
            
            cgr_str = f"\n> 💡 铜金比(宏观复苏先行器) **{cgr:.4f}**"
            comm_msg += cnh_str + cgr_str
            result["macro"].append(comm_msg)
            
            opt_msg = "**2. 期权与黑天鹅指标**"
            
            tnx_str = f"\n> 美10年期国债收益率 **{tnx:.3f}%**"
            if tnx > 4.2:
                tnx_str += " 🔴 <font color=\"#F04864\">(美债高企，强力压制全球科技股估值)</font>"
            elif tnx < 3.8:
                tnx_str += " 🟢 <font color=\"#2fc25b\">(美债回落，成长股迎流动性溢价)</font>"
            
            vix_str = f"\n> VIX恐慌指数 **{vix:.2f}**"
            if vix < 15:
                vix_str += " 🟢 <font color=\"#2fc25b\">(期权市场极度贪婪)</font>"
            elif vix > 20:
                vix_str += " 🔴 <font color=\"#F04864\">(华尔街大资金已启动对冲)</font>"
                if vix > 25:
                    result["risk_alert"] = "⚠️ **紧急避险提示**: VIX突破25，市场恐慌情绪升温！"
            
            skew_str = f"\n> SKEW黑天鹅指数 **{skew:.1f}**"
            if skew > 135:
                skew_str += " ⚠️ <font color=\"#F04864\">(尾部风险指标异动，需防范系统性黑天鹅！)</font>"
                result["risk_alert"] = "⚠️ **紧急避险提示**: SKEW突破135，警惕极端尾部风险！"
            
            opt_msg += tnx_str + vix_str + skew_str
            result["macro"].append(opt_msg)
            
            csi300 = close_df['000300.SS'].dropna()
            gspc = close_df['^GSPC'].dropna()
            
            csi300_rsi = MacroJudgmentEngine.calc_rsi(csi300).iloc[-1] if not csi300.empty else 50
            gspc_rsi = MacroJudgmentEngine.calc_rsi(gspc).iloc[-1] if not gspc.empty else 50
            csi300_mtm = get_mtm('000300.SS')
            csi300_mtm_pct = get_mtm_pct('000300.SS')
            gspc_mtm = get_mtm('^GSPC')
            gspc_mtm_pct = get_mtm_pct('^GSPC')

            csi300_trend_name, csi300_trend_desc = get_ma_trend('000300.SS')
            gspc_trend_name, gspc_trend_desc = get_ma_trend('^GSPC')

            us_msg = f"> 📊 **技术动能**: 标普500 5日MTM **{gspc_mtm:+.2f}** ({gspc_mtm_pct:+.2f}%, RSI: {gspc_rsi:.1f})\n> 📈 **大盘趋势**: **{gspc_trend_name}** - {gspc_trend_desc}"
            result["us_tech"] = us_msg
            
            a_msg = f"> 📊 **技术动能**: 沪深300 5日MTM **{csi300_mtm:+.2f}** ({csi300_mtm_pct:+.2f}%, RSI: {csi300_rsi:.1f})\n> 📈 **大盘趋势**: **{csi300_trend_name}** - {csi300_trend_desc}"
            result["cn_tech"] = a_msg

        except ImportError:
            log.warning("yfinance 未安装，跳过高阶宏观研判")
            result["macro"].append("> <font color=\"#8c8c8c\">yfinance未安装，宏观研判已降级</font>")
        except Exception as e:
            log.warning(f"高阶研判引擎运行失败: {e}", exc_info=True)
            result["macro"].append("> <font color=\"#8c8c8c\">引擎数据抓取异常，研判熔断</font>")

        return result

class NewsDigest:
    @staticmethod
    def score_news(title):
        score = 0
        
        # 负面关键词（直接丢弃）
        block_words = [
            "辞职", "离职", "减持", "亏损", "立案", "退市", "违规", 
            "跌停", "暴跌", "不及预期", "恶化"
        ]
        if any(w in title for w in block_words):
            return -1
        
        # 低价值关键词（扣分但不丢弃）
        LOW_WEIGHT_WORDS = {
            "早报": -2, "必读": -2, "提示性公告": -1,
            "互动平台": -3, "董事": -2, "股东大会": -2,
            "高管": -1, "聘任": -1, "例行": -2
        }
        for word, penalty in LOW_WEIGHT_WORDS.items():
            if word in title:
                score += penalty
        
        # T1 宏观与顶层政策（最高权重）
        t1_words = ["发改委", "工信部", "央行", "国务院", "新规", "印发", 
                    "降准", "降息", "证监会", "政治局", "重磅", "刺激", "利好", "支持"]
        for w in t1_words:
            if w in title:
                score += 10
        
        # T2 行业前瞻与业绩指引
        t2_words = ["超预期", "指引", "订单", "需求爆发", "上调", "产能", 
                    "供不应求", "扭亏", "净利", "商业化", "突破", "暴增", 
                    "中标", "合作", "发布", "研发", "获批", "新高"]
        for w in t2_words:
            if w in title:
                score += 5
        
        # T3 热门赛道词（增强产业新闻权重）
        t3_words = ["新能源", "人工智能", "AI", "算力", "半导体", "芯片", 
                    "光伏", "储能", "锂电", "数据中心", "云计算", "数字经济",
                    "国产替代", "高端制造", "一带一路", "国企改革"]
        for w in t3_words:
            if w in title:
                score += 3
        
        # 负面趋势词（进一步扣分）
        neg_trend_words = ["下降", "走低", "回落", "下滑", "大跌", "收跌", "低迷"]
        for w in neg_trend_words:
            if w in title:
                score -= 3
        
        return score

    @staticmethod
    def get_news(limit=5):
        news_list = []
        scored_news = []
        
        token = os.environ.get("TUSHARE_TOKEN", "").strip()
        if token:
            try:
                import tushare as ts
                pro = ts.pro_api(token)
                df = pro.news(src='cls', limit=limit+80)
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        try:
                            ts = pd.to_datetime(row['datetime'])
                            time_str = ts.strftime('%H:%M')
                        except Exception:
                            time_str = ''
                        title = row['title'] if row['title'] else row['content'][:50]+"..."
                        score = NewsDigest.score_news(title)
                        if score > 0:
                            scored_news.append((score, row['datetime'], time_str, title))
                            
                    if scored_news:
                        scored_news.sort(key=lambda x: (x[0], x[1]), reverse=True)
                        for _, _, time_str, title in scored_news[:limit]:
                            if time_str:
                                news_list.append(f"> **[{time_str}]** {title}")
                            else:
                                news_list.append(f"> {title}")
                        log.info(f"Tushare财联社新闻获取成功，筛选出 {len(news_list)} 条高价值资讯")
                        return news_list
                else:
                    log.warning("Tushare返回空数据，尝试备用源")
            except Exception as e:
                log.warning(f"Tushare机构新闻获取失败，降级至新浪源: {e}")

        try:
            url = "https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2509&k=&num=80&page=1"
            res = safe_request_get(url, headers=SINA_HEADERS, timeout=8)
            if res:
                try:
                    data = res.json().get('result', {}).get('data', [])
                    for doc in data:
                        title = doc.get('title', '')
                        score = NewsDigest.score_news(title)
                        if score > 0:
                            dt = datetime.fromtimestamp(int(doc['ctime']))
                            scored_news.append((score, doc['ctime'], dt.strftime('%H:%M'), title))
                    
                    if scored_news:
                        scored_news.sort(key=lambda x: (x[0], x[1]), reverse=True)
                        for _, _, time_str, title in scored_news[:limit]:
                            news_list.append(f"> **[{time_str}]** {title}")
                        log.info(f"新浪新闻获取成功，筛选出 {len(news_list)} 条高价值资讯")
                except Exception as e:
                    log.warning(f"解析新浪新闻失败: {e}")
        except Exception as e:
            log.warning(f"获取新浪新闻兜底失败: {e}")
            
        return news_list

class BriefingRenderer:
    @staticmethod
    def render() -> str:
        date_str = _today_str()
        
        def fetch_all_data():
            results = {}
            
            def fetch_judgments():
                try:
                    return MacroJudgmentEngine.get_judgments()
                except Exception as e:
                    log.warning(f"获取宏观研判失败: {e}")
                    return {"macro": [], "us_tech": "", "cn_tech": "", "risk_alert": ""}
            
            def fetch_ashare():
                try:
                    return MacroBrain.get_ashare_indices()
                except Exception as e:
                    log.warning(f"获取A股指数失败: {e}")
                    return {}
            
            def fetch_global():
                try:
                    return MacroBrain.get_global_indices()
                except Exception as e:
                    log.warning(f"获取全球指数失败: {e}")
                    return {}
            
            def fetch_northbound():
                try:
                    return fetch_northbound_flow()
                except Exception as e:
                    log.warning(f"获取北向资金失败: {e}")
                    return (0.0, "")
            
            def fetch_news():
                try:
                    return NewsDigest.get_news(limit=12)
                except Exception as e:
                    log.warning(f"获取新闻失败: {e}")
                    return []
            
            def fetch_hot_sectors_data():
                try:
                    return fetch_hot_sectors()
                except Exception as e:
                    log.warning(f"获取热点板块失败: {e}")
                    return {}
            
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = {
                    'judgments': executor.submit(fetch_judgments),
                    'ashare': executor.submit(fetch_ashare),
                    'global': executor.submit(fetch_global),
                    'northbound': executor.submit(fetch_northbound),
                    'news': executor.submit(fetch_news),
                    'hot_sectors': executor.submit(fetch_hot_sectors_data),
                }
                
                for name, future in futures.items():
                    try:
                        results[name] = future.result(timeout=30)
                    except FuturesTimeoutError:
                        log.warning(f"{name} 获取超时")
                        results[name] = {} if name in ['ashare', 'global', 'hot_sectors'] else []
            
            return results
        
        data = fetch_all_data()
        judgments = data.get('judgments', {})
        ashare_idx = data.get('ashare', {})
        global_idx = data.get('global', {})
        flow_amt, flow_msg = data.get('northbound', (0.0, ""))
        news = data.get('news', [])
        hot_sectors = data.get('hot_sectors', {})
        
        lines = []
        lines.append(f"## 🤖 AI 每日市场简报\n*{date_str}*\n")
        
        if judgments.get("risk_alert"):
            lines.append(f"### ⚠️ {judgments['risk_alert']}\n")
        
        lines.append("---\n### 🌍 大类资产与衍生品\n")
        if judgments.get("macro"):
            lines.append("\n\n".join(judgments["macro"]))
        else:
            lines.append("> <font color=\"#8c8c8c\">研判引擎暂无数据输出</font>")
        
        lines.append("\n---\n### 🇺🇸 美股大盘体检")
        us_strs = []
        for name, data_item in global_idx.items():
            if name == "恒生指数":
                continue
            if "price" in data_item and "pct" in data_item:
                pct = data_item['pct']
                is_us = name in ["纳斯达克", "标普500", "道琼斯"]
                us_strs.append(f"- **{name}**: {data_item['price']:.2f} {format_dingtalk_pct(pct, is_us)}")
        
        if us_strs:
            lines.extend(us_strs)
        else:
            lines.append("- <font color=\"#8c8c8c\">暂无美股数据</font>")
        
        if judgments.get("us_tech"):
            lines.append(judgments["us_tech"])
        
        lines.append("\n---\n### 🇭🇰 港股大盘")
        if "恒生指数" in global_idx:
            hsi_data = global_idx["恒生指数"]
            if "price" in hsi_data and "pct" in hsi_data:
                lines.append(f"- **恒生指数**: {hsi_data['price']:.2f} {format_dingtalk_pct(hsi_data['pct'], False)}")
        else:
            lines.append("- <font color=\"#8c8c8c\">暂无港股数据</font>")
        
        lines.append("\n---\n### 🇨🇳 A股大盘体检")
        ashare_strs = []
        for name, data_item in ashare_idx.items():
            pct = data_item.get('pct', 0.0)
            price = data_item.get('price')
            if price is not None:
                ashare_strs.append(f"- **{name}**: {price:.2f} {format_dingtalk_pct(pct, False)}")
            else:
                ashare_strs.append(f"- **{name}**: {format_dingtalk_pct(pct, False)}")
        
        if ashare_strs:
            lines.extend(ashare_strs)
        else:
            lines.append("- <font color=\"#8c8c8c\">暂无A股数据</font>")
        
        if judgments.get("cn_tech"):
            lines.append(judgments["cn_tech"])
        
        if flow_msg:
            lines.append(f"\n> 💰 **北向资金** ({flow_amt:+.1f}亿元): {flow_msg.strip().replace('北向资金: ', '')}")
        
        if hot_sectors:
            top_sectors = list(set(hot_sectors.values()))[:3]
            lines.append(f"\n> 🔥 **热点板块**: {', '.join(top_sectors)}")
        
        lines.append("\n---\n### 📰 核心投研资讯\n")
        if news:
            lines.append("\n\n".join(news))
        else:
            lines.append("> <font color=\"#8c8c8c\">暂无重大新闻</font>")
        
        lines.append("\n---\n*<font color=\"#8c8c8c\">Antigravity 机构级量化引擎自动生成</font>*")
        
        content = "\n".join(lines)
        
        if len(content) > DINGTALK_MAX_LENGTH:
            log.warning(f"简报长度 {len(content)} 超过限制，需要截断")
            # 使用更精确的截断策略
            trunc_marker = "\n---\n*<font color=\"#8c8c8c\">"
            idx = content.rfind(trunc_marker, 0, DINGTALK_MAX_LENGTH)
            if idx != -1:
                content = content[:idx] + "\n\n*⚠️ 内容过长已截断，请检查完整日志*"
            else:
                content = content[:DINGTALK_MAX_LENGTH] + "\n\n*⚠️ 内容过长已截断，请检查完整日志*"
        
        return content

def send_dingtalk(content: str):
    webhook = os.environ.get('DINGTALK_WEBHOOK')
    if not webhook:
        log.warning("未配置 DINGTALK_WEBHOOK，仅在控制台输出：\n" + content)
        return
        
    try:
        payload = {
            'msgtype': 'markdown',
            'markdown': {
                'title': '🤖 每日市场简报',
                'text': content
            }
        }
        res = requests.post(webhook, json=payload, timeout=10)
        res.raise_for_status()
        res_dict = res.json()
        if res_dict.get('errcode', 0) != 0:
            log.error(f"❌ 钉钉推送失败: {res_dict}")
        else:
            log.info("✅ 简报推送成功")
    except requests.exceptions.RequestException as e:
        log.error(f"❌ 推送网络请求失败: {e}")
    except Exception as e:
        log.error(f"❌ 推送处理失败: {e}")

if __name__ == '__main__':
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 添加文件日志处理器
    try:
        file_handler = logging.FileHandler(f'briefing_{_today_str()}.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        log.info(f"日志文件: briefing_{_today_str()}.log")
    except Exception as e:
        log.warning(f"文件日志初始化失败: {e}")
    
    # 健康检查
    health_issues = health_check()
    if health_issues:
        log.warning("健康检查发现问题:")
        for issue in health_issues:
            log.warning(f"  - {issue}")
    
    try:
        report = BriefingRenderer.render()
        log.info(f"生成简报如下:\n{report}")
        send_dingtalk(report)
    except Exception as e:
        log.critical(f"简报生成崩溃: {e}", exc_info=True)
