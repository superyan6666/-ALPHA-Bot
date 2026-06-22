import os
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
os.environ['NO_PROXY'] = '*'

import time
import json
import requests
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

TZ_BJS = __import__('pytz').timezone('Asia/Shanghai')

class Config:
    DINGTALK_WEBHOOK = os.environ.get('DINGTALK_WEBHOOK', '')
    RUN_MODE = os.environ.get('RUN_MODE', 'normal')

def _patched_request(self, method, url, **kwargs):
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    whitelist_domains = ('eastmoney.com', 'dfcfw.com', 'sinajs.cn', 'money.163.com', '126.net', 'gtimg.cn', '10jqka.com.cn', 'csindex.com.cn', 'szse.cn')
    needs_patch = any(hostname == d or hostname.endswith('.' + d) for d in whitelist_domains)
    
    if needs_patch:
        headers = kwargs.get('headers', {})
        if not isinstance(headers, dict):
            headers = dict(headers)
        headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
        
        if 'eastmoney.com' in hostname or 'dfcfw.com' in hostname:
            headers['Referer'] = 'https://quote.eastmoney.com/'
        elif 'sina.com.cn' in hostname or 'sinajs.cn' in hostname:
            headers['Referer'] = 'https://finance.sina.com.cn/'
        elif '126.net' in hostname:
            headers['Referer'] = 'http://quotes.money.163.com/'
        elif 'gtimg.cn' in hostname:
            headers['Referer'] = 'https://finance.qq.com/'
        
        kwargs['headers'] = headers
    
    kwargs['timeout'] = kwargs.get('timeout', 15.0)
    return requests.Session.request.__wrapped__(self, method, url, **kwargs) if hasattr(requests.Session.request, '__wrapped__') else requests.Session.request(self, method, url, **kwargs)

requests.Session.request = _patched_request

def fetch_spot_akshare():
    try:
        import akshare as ak
        return ak.stock_zh_a_spot_em()
    except Exception as e:
        log.warning(f"akshare failed: {e}")
        return None

def fetch_spot_tencent():
    log.info("Using Tencent fallback")
    url = "http://qt.gtimg.cn/q=sh000001,sz399001"
    try:
        resp = requests.get(url, timeout=5)
        resp.encoding = 'gbk'
        lines = resp.text.strip().split('\n')
        results = []
        for line in lines:
            if '=' not in line: continue
            var, data = line.split('=', 1)
            parts = data.replace('"', '').replace(';', '').split('~')
            if len(parts) < 45: continue
            results.append({
                '代码': parts[2],
                '名称': parts[1],
                '最新价': float(parts[3]),
                '涨跌幅': float(parts[32]),
                '成交量': float(parts[36]),
            })
        return pd.DataFrame(results)
    except Exception as e:
        log.warning(f"Tencent failed: {e}")
        return None

def fetch_index(symbol):
    try:
        import akshare as ak
        df = ak.stock_zh_index_daily_tx(symbol=symbol)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            return df
    except Exception as e:
        log.warning(f"Index fetch failed: {e}")
    return None

def fetch_hot_sectors():
    try:
        import akshare as ak
        df = ak.stock_board_industry_name_em()
        if df is not None and not df.empty:
            return df.nlargest(5, '涨跌幅')['板块名称'].tolist()
    except Exception as e:
        log.warning(f"Hot sectors failed: {e}")
    return []

def fetch_northbound_flow():
    try:
        import akshare as ak
        df = ak.stock_em_hsgt_north_net_flow_in(indicator="沪深港通")
        if df is not None and not df.empty:
            col = 'value' if 'value' in df.columns else df.columns[-1]
            today_flow = float(df.iloc[-1][col]) / 1e8
            return today_flow
    except Exception as e:
        log.warning(f"Northbound flow failed: {e}")
    return 0.0

def generate_macro_section():
    try:
        import yfinance as yf
        tickers = yf.Tickers("^TNX ^VIX ^GSPC")
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
        sp500_l, sp500_p = get_last_pct('^GSPC')

        msg = (
            f"### 🌍 隔夜外围与宏观风控快报\n"
            f"- **标普500 (^GSPC)**: `{sp500_l:.2f}` ({sp500_p:+.2f}%)\n"
            f"- **恐慌指数 (^VIX)**: `{vix_l:.2f}` ({vix_p:+.2f}%) " + ("⚠️ **极度恐慌**" if vix_l > 25 else "✅ 情绪稳定") + "\n"
            f"- **美债10年期 (^TNX)**: `{tnx_l:.2f}%` ({tnx_p:+.2f}%)\n\n"
            f"> *数据源: Yahoo Finance*"
        )
        return msg
    except Exception as e:
        log.warning(f"Macro data failed: {e}")
        return f"### 🌍 隔夜外围与宏观指标快报\n⚠️ 外围数据获取失败 ({e})"

def generate_market_analysis():
    now = datetime.now(TZ_BJS)
    msg = f"## 📊 {now.strftime('%Y-%m-%d')} A股市场日报\n\n"
    
    msg += generate_macro_section() + "\n\n"
    
    idx_df = fetch_index('sh000001')
    if idx_df is not None and not idx_df.empty:
        cl = idx_df['close']
        if len(cl) >= 2:
            pct = (cl.iloc[-1] - cl.iloc[-2]) / cl.iloc[-2] * 100
            ma20 = cl.rolling(20).mean().iloc[-1] if len(cl) >= 20 else cl.iloc[-1]
            trend_status = "🟢 多头趋势" if cl.iloc[-1] > ma20 else "🔴 空头趋势"
            
            msg += f"### 📈 A股大盘行情\n"
            msg += f"- **上证指数**: `{cl.iloc[-1]:.2f}` ({pct:+.2f}%)\n"
            msg += f"- **20日均线**: `{ma20:.2f}` {trend_status}\n\n"
    
    north_flow = fetch_northbound_flow()
    if north_flow != 0.0:
        if north_flow > 30:
            msg += f"- 🌊 **北向资金**: 大举流入 **+{north_flow:.0f}亿**\n"
        elif north_flow < -30:
            msg += f"- ❄️ **北向资金**: 大幅流出 **{north_flow:.0f}亿**\n"
        else:
            msg += f"- ⚖️ **北向资金**: 温和流动 ({north_flow:+.0f}亿)\n"
    
    hot_sectors = fetch_hot_sectors()
    if hot_sectors:
        msg += f"\n### 🔥 行业热点\n"
        msg += f"- **领涨板块**: {', '.join(hot_sectors)}\n"
    
    msg += "\n---\n\n"
    msg += "**💡 今日策略建议**: 根据市场状态调整仓位，关注热点板块轮动机会。\n"
    
    return msg

def send_to_dingtalk(title, content):
    if not Config.DINGTALK_WEBHOOK:
        log.warning("未配置钉钉Webhook，仅打印输出")
        print("="*50)
        print(title)
        print("="*50)
        print(content)
        print("="*50)
        return
    
    headers = {"Content-Type": "application/json"}
    payload = {
        'msgtype': 'markdown',
        'markdown': {
            'title': title,
            'text': content
        }
    }
    
    try:
        res = requests.post(Config.DINGTALK_WEBHOOK, json=payload, headers=headers, timeout=10)
        res.raise_for_status()
        log.info("✅ 钉钉推送成功")
    except Exception as e:
        log.error(f"❌ 钉钉推送失败: {e}")

def main():
    log.info("🚀 每日投研简报生成器启动")
    
    try:
        content = generate_market_analysis()
        send_to_dingtalk("📊 每日投研简报", content)
        log.info("✅ 每日投研简报生成完成")
    except Exception as e:
        log.error(f"❌ 生成简报失败: {e}", exc_info=True)
        send_to_dingtalk("🚨 简报生成失败", f"生成每日投研简报时发生错误: {str(e)}")

if __name__ == '__main__':
    main()
