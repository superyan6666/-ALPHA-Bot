import os
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
os.environ['NO_PROXY'] = '*'

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import pytz
import requests

requests.adapters.DEFAULT_TIMEOUT = 10

TZ_BJS = pytz.timezone('Asia/Shanghai')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

DINGTALK_WEBHOOK = os.environ.get('DINGTALK_WEBHOOK', '')
NOTIFY_SEC_KEYWORD = os.environ.get('NOTIFY_SEC_KEYWORD', 'Hermes').strip()
DEMO_MODE = os.environ.get('DEMO_MODE', 'false').lower() == 'true'

def _today_str() -> str:
    return datetime.now(TZ_BJS).strftime('%Y-%m-%d')

def _now_str() -> str:
    return datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M:%S')

def send_dingtalk(title: str, content: str):
    if not DINGTALK_WEBHOOK:
        log.warning("未配置 DINGTALK_WEBHOOK，跳过推送")
        print(f"【钉钉通知】\n标题: {title}\n\n{content}")
        return
    
    sec_keyword = NOTIFY_SEC_KEYWORD
    if sec_keyword and sec_keyword not in title:
        title = f"{sec_keyword} | {title}"
    if sec_keyword and sec_keyword not in content:
        content = f"### {sec_keyword}\n\n{content}"
    
    payload = {
        'msgtype': 'markdown',
        'markdown': {
            'title': title,
            'text': content
        }
    }
    
    try:
        res = requests.post(DINGTALK_WEBHOOK, json=payload, headers={'Content-Type': 'application/json'}, timeout=10)
        res.raise_for_status()
        res_dict = res.json()
        if res_dict.get('errcode', 0) != 0:
            log.error(f"钉钉推送失败: {res_dict}")
        else:
            log.info("钉钉推送成功")
    except Exception as e:
        log.error(f"钉钉推送异常: {e}")

def fetch_index_data(symbol: str) -> pd.DataFrame:
    if DEMO_MODE:
        dates = pd.date_range(end=_today_str(), periods=120, freq='D')
        closes = np.cumsum(np.random.randn(120) * 0.5) + 3500
        return pd.DataFrame({'date': dates, 'close': closes})
    
    try:
        url = f"http://qt.gtimg.cn/q={symbol}"
        resp = requests.get(url, timeout=5)
        resp.encoding = 'gbk'
        parts = resp.text.replace('"', '').replace(';', '').split('~')
        if len(parts) >= 30:
            url_kline = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.{symbol[2:]}&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=1&end=20990101&limit=120"
            kline_resp = requests.get(url_kline, timeout=5)
            kline_data = kline_resp.json()
            if kline_data.get('data') and kline_data['data'].get('klines'):
                df = pd.DataFrame([line.split(',') for line in kline_data['data']['klines']],
                                columns=['date', 'open', 'close', 'high', 'low', 'vol', 'amount'])
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                return df
    except Exception as e:
        log.debug(f"获取指数 {symbol} 数据失败: {e}")
    
    return pd.DataFrame()

def fetch_market_overview() -> Dict:
    if DEMO_MODE:
        return {
            'sh_close': 3528.67,
            'sh_pct': 1.23,
            'sh_high': 3545.21,
            'sh_low': 3502.45,
            'sh_vol': 45210000,
            'sz_close': 11234.56,
            'sz_pct': 1.56,
            'market_data': {}
        }
    
    try:
        url = "https://push2.eastmoney.com/api/qt/stock/get?secid=1.000001&fields=f57,f58,f116,f117,f118,f119,f120,f107,f108,f109,f110"
        resp = requests.get(url, timeout=5)
        data = resp.json().get('data', {})
        
        url_sz = "https://push2.eastmoney.com/api/qt/stock/get?secid=0.399001&fields=f57,f58,f116,f117,f118,f119,f120,f107,f108,f109,f110"
        resp_sz = requests.get(url_sz, timeout=5)
        data_sz = resp_sz.json().get('data', {})
        
        return {
            'sh_close': float(data.get('f57', 0)),
            'sh_pct': float(data.get('f116', 0)),
            'sh_high': float(data.get('f118', 0)),
            'sh_low': float(data.get('f119', 0)),
            'sh_vol': float(data.get('f120', 0)),
            'sz_close': float(data_sz.get('f57', 0)),
            'sz_pct': float(data_sz.get('f116', 0)),
            'market_data': {}
        }
    except Exception as e:
        log.error(f"获取市场概况失败: {e}")
        return {}

def fetch_hot_sectors() -> List[Dict]:
    if DEMO_MODE:
        return [
            {'name': '半导体', 'pct': 5.23, 'volume': 12500000, 'amount': 450000},
            {'name': '人工智能', 'pct': 4.12, 'volume': 9800000, 'amount': 380000},
            {'name': '新能源', 'pct': 3.45, 'volume': 8200000, 'amount': 320000},
            {'name': '消费电子', 'pct': 2.87, 'volume': 6500000, 'amount': 250000},
            {'name': '军工', 'pct': 2.34, 'volume': 5800000, 'amount': 220000},
            {'name': '医药', 'pct': 1.98, 'volume': 7200000, 'amount': 280000},
            {'name': '券商', 'pct': 1.65, 'volume': 6100000, 'amount': 240000},
            {'name': '银行', 'pct': 0.87, 'volume': 4500000, 'amount': 180000},
        ]
    
    try:
        url = "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=20&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:90+t:2+f:!50&fields=f12,f14,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f22,f33,f11,f62,f128,f136,f115,f152,f124,f107,f104,f105,f140,f141,f207,f208,f209,f222"
        resp = requests.get(url, timeout=5)
        data = resp.json().get('data', {})
        items = data.get('diff', [])
        
        hotspots = []
        for item in items[:10]:
            hotspots.append({
                'name': item.get('f14', ''),
                'pct': float(item.get('f3', 0)),
                'volume': float(item.get('f5', 0)),
                'amount': float(item.get('f6', 0))
            })
        return hotspots
    except Exception as e:
        log.error(f"获取热点板块失败: {e}")
        return []

def fetch_etf_data() -> List[Dict]:
    if DEMO_MODE:
        return [
            {'code': '510300', 'name': '沪深300ETF', 'price': 4.25, 'pct': 1.12, 'pct_5d': 3.45, 'pct_20d': 8.23, 'above_ma5': True, 'above_ma20': True, 'vol': 25000000},
            {'code': '510500', 'name': '中证500ETF', 'price': 6.12, 'pct': 1.87, 'pct_5d': 4.56, 'pct_20d': 12.34, 'above_ma5': True, 'above_ma20': True, 'vol': 18000000},
            {'code': '512880', 'name': '券商ETF', 'price': 1.25, 'pct': 2.34, 'pct_5d': 5.67, 'pct_20d': 15.67, 'above_ma5': True, 'above_ma20': True, 'vol': 35000000},
            {'code': '512480', 'name': '医药ETF', 'price': 0.87, 'pct': 1.56, 'pct_5d': 2.34, 'pct_20d': 4.56, 'above_ma5': True, 'above_ma20': False, 'vol': 12000000},
            {'code': '510880', 'name': '红利ETF', 'price': 2.34, 'pct': 0.67, 'pct_5d': 1.23, 'pct_20d': 2.34, 'above_ma5': False, 'above_ma20': False, 'vol': 8000000},
            {'code': '159915', 'name': '创业板ETF', 'price': 2.12, 'pct': 2.45, 'pct_5d': 5.67, 'pct_20d': 14.23, 'above_ma5': True, 'above_ma20': True, 'vol': 22000000},
            {'code': '159905', 'name': '深证100ETF', 'price': 1.87, 'pct': 1.78, 'pct_5d': 4.32, 'pct_20d': 9.87, 'above_ma5': True, 'above_ma20': True, 'vol': 15000000},
            {'code': '512690', 'name': '军工ETF', 'price': 1.45, 'pct': 1.23, 'pct_5d': 3.45, 'pct_20d': 7.65, 'above_ma5': True, 'above_ma20': True, 'vol': 16000000},
            {'code': '513050', 'name': '中概互联', 'price': 1.67, 'pct': 0.89, 'pct_5d': 2.12, 'pct_20d': 3.45, 'above_ma5': False, 'above_ma20': False, 'vol': 9000000},
            {'code': '513100', 'name': '纳指ETF', 'price': 2.89, 'pct': 1.34, 'pct_5d': 3.56, 'pct_20d': 8.90, 'above_ma5': True, 'above_ma20': True, 'vol': 6000000},
        ]
    
    etf_codes = {
        '510300': '沪深300ETF', '510500': '中证500ETF', '512880': '券商ETF',
        '512480': '医药ETF', '510880': '红利ETF', '159915': '创业板ETF',
        '159905': '深证100ETF', '512690': '军工ETF', '513050': '中概互联',
        '513100': '纳指ETF', '513500': '日经ETF'
    }
    
    results = []
    for code, name in etf_codes.items():
        try:
            market = 'sh' if code.startswith('5') else 'sz'
            url = f"http://qt.gtimg.cn/q={market}{code}"
            resp = requests.get(url, timeout=5)
            resp.encoding = 'gbk'
            parts = resp.text.replace('"', '').replace(';', '').split('~')
            if len(parts) >= 50:
                close = float(parts[3])
                pct = float(parts[32])
                
                url_kline = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={'1' if code.startswith('5') else '0'}.{code}&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=1&end=20990101&limit=60"
                kline_resp = requests.get(url_kline, timeout=5)
                kline_data = kline_resp.json()
                
                pct_5d = 0
                pct_20d = 0
                above_ma5 = False
                above_ma20 = False
                
                if kline_data.get('data') and kline_data['data'].get('klines'):
                    klines = kline_data['data']['klines']
                    if len(klines) >= 20:
                        closes = [float(line.split(',')[2]) for line in klines]
                        pct_5d = (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 else 0
                        pct_20d = (closes[-1] / closes[-21] - 1) * 100 if len(closes) >= 21 else 0
                        ma5 = np.mean(closes[-5:])
                        ma20 = np.mean(closes[-20:])
                        above_ma5 = closes[-1] > ma5
                        above_ma20 = closes[-1] > ma20
                
                results.append({
                    'code': code,
                    'name': name,
                    'price': close,
                    'pct': pct,
                    'pct_5d': pct_5d,
                    'pct_20d': pct_20d,
                    'above_ma5': above_ma5,
                    'above_ma20': above_ma20,
                    'vol': 0
                })
        except Exception as e:
            log.debug(f"获取ETF {code} 数据失败: {e}")
    
    return results

def fetch_northbound_flow() -> Tuple[float, str]:
    if DEMO_MODE:
        return 45.67, "- 🌊 **北向资金**: +46亿 (大举流入)"
    
    try:
        url = "https://push2.eastmoney.com/api/qt/stock/get?secid=1.000001&fields=f107,f108"
        resp = requests.get(url, timeout=5)
        data = resp.json().get('data', {})
        
        north_in = float(data.get('f107', 0))
        
        if north_in > 30:
            return north_in, f"- 🌊 **北向资金**: +{north_in:.0f}亿 (大举流入)"
        elif north_in < -30:
            return north_in, f"- ❄️ **北向资金**: {north_in:.0f}亿 (大幅流出)"
        else:
            return north_in, f"- ⚖️ **北向资金**: {north_in:+.0f}亿"
    except Exception as e:
        log.debug(f"获取北向资金失败: {e}")
        return 0, ""

def generate_market_section() -> str:
    overview = fetch_market_overview()
    idx_df = fetch_index_data('sh000001')
    
    sh_close = overview.get('sh_close', 0)
    sh_pct = overview.get('sh_pct', 0)
    
    trend = "⚖️ 震荡整理"
    trend_desc = "多空博弈"
    
    if len(idx_df) >= 60:
        closes = idx_df['close'].dropna()
        ma5 = closes.rolling(5).mean().iloc[-1]
        ma20 = closes.rolling(20).mean().iloc[-1]
        ma60 = closes.rolling(60).mean().iloc[-1]
        
        if ma5 > ma20 > ma60:
            trend = "🔥 多头排列"
            trend_desc = "均线多头排列，趋势强劲"
        elif ma5 < ma20 < ma60:
            trend = "🧊 空头排列"
            trend_desc = "均线空头排列，趋势走弱"
        elif ma5 > ma20:
            trend = "⚡ 短期偏强"
            trend_desc = "短期均线拐头向上"
    
    north_flow, north_msg = fetch_northbound_flow()
    
    msg = (
        f"### 📊 市场全景分析\n"
        f"- **上证指数**: `{sh_close:.2f}` (今日 **{sh_pct:+.2f}%**)\n"
        f"- **均线趋势**: {trend} - {trend_desc}\n"
        f"{north_msg}\n"
    )
    
    return msg

def generate_sector_section() -> str:
    hotspots = fetch_hot_sectors()
    
    if not hotspots:
        return "### 🌋 行业热点排行\n⚠️ 数据获取失败\n"
    
    msg = "### 🌋 行业热点排行\n"
    msg += "| 板块 | 涨幅 |\n"
    msg += "|------|------|\n"
    
    for h in hotspots[:8]:
        msg += f"| **{h['name']}** | {h['pct']:+.2f}% |\n"
    
    top = hotspots[0]
    if top['pct'] > 3:
        msg += f"\n🔥 **领涨主线**: {top['name']} 涨幅 {top['pct']:+.2f}%\n"
    
    return msg

def generate_etf_section() -> str:
    etf_data = fetch_etf_data()
    
    if not etf_data:
        return "### 📈 ETF轮动策略\n⚠️ 数据获取失败\n"
    
    etf_data.sort(key=lambda x: x['pct_20d'], reverse=True)
    
    msg = "### 📈 ETF轮动策略\n"
    msg += "| ETF | 现价 | 5日涨幅 | 20日涨幅 | 均线状态 |\n"
    msg += "|------|------|---------|----------|----------|\n"
    
    for etf in etf_data[:8]:
        ma_status = '✅' if (etf['above_ma5'] and etf['above_ma20']) else '⚠️' if (etf['above_ma5'] or etf['above_ma20']) else '❌'
        msg += (f"| {etf['name']} (`{etf['code']}`) | ¥{etf['price']:.2f} | "
                f"{etf['pct_5d']:+.2f}% | {etf['pct_20d']:+.2f}% | {ma_status} |\n")
    
    top_etf = etf_data[0]
    if top_etf['above_ma20'] and top_etf['pct_20d'] > 3:
        msg += f"\n🔥 **轮动信号**: {top_etf['name']} 表现最强，建议关注！\n"
    
    return msg

def generate_macro_section() -> str:
    if DEMO_MODE:
        return (
            f"### 🌍 隔夜外围快报\n"
            f"- **标普500**: `5234.56` (+1.23%)\n"
            f"- **恐慌指数VIX**: `14.56` (-2.34%) ✅ 稳定\n"
            f"- **美债10年期收益率**: `4.25%` (+0.12%)\n"
        )
    return "### 🌍 隔夜外围快报\n⚠️ 外部数据源暂不可用，已跳过\n"

def generate_stock_signals_section() -> str:
    if DEMO_MODE:
        return (
            f"### 🎯 今日选股信号\n"
            f"#### 🔥 核心主力池\n"
            f"- **贵州茅台** (`600519`): ¥1680.00 | 得分 `87.5` | ⭐⭐⭐⭐⭐ 🐯 [S级·老虎机]\n"
            f"- **宁德时代** (`300750`): ¥285.50 | 得分 `82.3` | ⭐⭐⭐⭐ 🐕 [A级·看门狗]\n"
            f"\n#### 🛰️ 卫星观察池\n"
            f"- **比亚迪** (`002594`): ¥268.00 | 得分 `76.8`\n"
            f"- **隆基绿能** (`601012`): ¥45.60 | 得分 `74.2`\n"
            f"- **阳光电源** (`300274`): ¥128.50 | 得分 `72.1`\n"
        )
    return "### 🎯 今日选股信号\n⚠️ 信号获取模块暂不可用，已跳过\n"

def main():
    log.info("🚀 启动每日投研简报生成引擎...")
    if DEMO_MODE:
        log.info("📢 运行在演示模式，使用模拟数据")
    
    now = datetime.now(TZ_BJS)
    title = f"📋 A股每日投研简报 ({now.strftime('%Y-%m-%d')})"
    
    sections = []
    
    sections.append(generate_macro_section())
    sections.append(generate_market_section())
    sections.append(generate_sector_section())
    sections.append(generate_etf_section())
    sections.append(generate_stock_signals_section())
    
    disclaimer = (
        "\n---\n"
        "> ⚠️ **风险提示**: 以上内容仅供参考，不构成投资建议。\n"
        f"> 📅 生成时间: {_now_str()}"
        f"{(' | 📢 演示模式' if DEMO_MODE else '')}"
    )
    sections.append(disclaimer)
    
    full_content = "\n\n".join(sections)
    
    log.info(f"简报生成完成，共 {len(full_content)} 字符")
    
    send_dingtalk(title, full_content)
    
    log.info("✅ 每日投研简报任务完成")

if __name__ == '__main__':
    main()