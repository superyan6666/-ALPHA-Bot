import os
import sys
import time
import logging
import hashlib
import base64
import hmac
import urllib.parse
from datetime import datetime, timedelta
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import pytz
import requests

TZ_BJS = pytz.timezone('Asia/Shanghai')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

DINGTALK_WEBHOOK = os.environ.get('DINGTALK_WEBHOOK', '')
DINGTALK_SECRET = os.environ.get('DINGTALK_SECRET', '')
NOTIFY_SEC_KEYWORD = os.environ.get('NOTIFY_SEC_KEYWORD', '投研简报')

REQUEST_TIMEOUT = 8

ETF_UNIVERSE = [
    ('sh510300', '510300', '沪深300ETF', '宽基'),
    ('sh510500', '510500', '中证500ETF', '宽基'),
    ('sh588000', '588000', '科创50ETF', '宽基'),
    ('sz159915', '159915', '创业板ETF', '宽基'),
    ('sh510050', '510050', '上证50ETF', '宽基'),
    ('sz159901', '159901', '深证100ETF', '宽基'),
    ('sh512880', '512880', '证券ETF', '行业'),
    ('sh512690', '512690', '酒ETF', '行业'),
    ('sh512010', '512010', '医药ETF', '行业'),
    ('sh512660', '512660', '军工ETF', '行业'),
    ('sh515030', '515030', '新能源车ETF', '行业'),
    ('sh512400', '512400', '有色金属ETF', '行业'),
    ('sh512760', '512760', '芯片ETF', '行业'),
    ('sh515050', '515050', '5G ETF', '行业'),
    ('sh512980', '512980', '传媒ETF', '行业'),
    ('sh512600', '512600', '红利ETF', '主题'),
    ('sh513100', '513100', '纳指ETF', '海外'),
]

INDEX_LIST = [
    ('sh000001', '上证指数'),
    ('sh000300', '沪深300'),
    ('sh000905', '中证500'),
    ('sz399006', '创业板指'),
    ('sh000688', '科创50'),
]


def _sign_dingtalk_url(webhook_url: str, secret: str) -> str:
    if not secret:
        return webhook_url
    timestamp = str(round(time.time() * 1000))
    string_to_sign = f"{timestamp}\n{secret}"
    hmac_code = hmac.new(
        secret.encode('utf-8'),
        string_to_sign.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
    sep = '&' if '?' in webhook_url else '?'
    return f"{webhook_url}{sep}timestamp={timestamp}&sign={sign}"


def send_dingtalk_notification(title: str, content: str) -> bool:
    if not DINGTALK_WEBHOOK:
        log.warning("未配置 DINGTALK_WEBHOOK，跳过钉钉推送")
        return False

    try:
        url = _sign_dingtalk_url(DINGTALK_WEBHOOK, DINGTALK_SECRET)

        CHUNK_SIZE = 18000
        chunks = [content[i:i + CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]

        if len(chunks) > 3:
            log.warning(f"消息过长({len(chunks)}段)，截断前3段")
            chunks = chunks[:3]
            chunks[-1] += "\n\n> ⚠️ *(内容超出限制，已截断)*"

        for idx, chunk in enumerate(chunks):
            text = chunk if idx == 0 else f"_(续上条)_\n\n{chunk}"
            msg_title = title if len(chunks) == 1 else f"{title} (Part {idx+1}/{len(chunks)})"

            if NOTIFY_SEC_KEYWORD and NOTIFY_SEC_KEYWORD not in text:
                text = f"### {NOTIFY_SEC_KEYWORD}\n\n{text}"

            payload = {
                'msgtype': 'markdown',
                'markdown': {
                    'title': msg_title,
                    'text': text
                }
            }

            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            if result.get('errcode', 0) != 0:
                log.error(f"钉钉推送失败: {result}")
                return False
            log.info(f"✅ 钉钉推送成功 ({idx+1}/{len(chunks)})")

            if idx < len(chunks) - 1:
                time.sleep(1)

        return True
    except Exception as e:
        log.error(f"❌ 钉钉推送异常: {e}")
        return False


def get_sina_kline(symbol: str, scale: int = 240, datalen: int = 300) -> pd.DataFrame:
    url = f"https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData?symbol={symbol}&scale={scale}&datalen={datalen}"
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['day'] = pd.to_datetime(df['day'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('day').reset_index(drop=True)
        return df
    except Exception as e:
        log.debug(f"新浪K线获取失败 {symbol}: {e}")
        return pd.DataFrame()


def get_sina_realtime(symbols: list[str]) -> dict:
    if not symbols:
        return {}
    url = f"https://hq.sinajs.cn/list={','.join(symbols)}"
    try:
        headers = {
            'Referer': 'https://finance.sina.com.cn/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
        resp.encoding = 'gbk'
        text = resp.text
        result = {}
        for line in text.strip().split('\n'):
            if '=' not in line:
                continue
            var_part, data_part = line.split('=', 1)
            symbol = var_part.replace('var hq_str_', '').strip()
            data = data_part.strip().strip(';').strip('"')
            if not data:
                continue
            parts = data.split(',')
            if len(parts) >= 32:
                result[symbol] = {
                    'name': parts[0],
                    'open': float(parts[1]) if parts[1] else 0,
                    'pre_close': float(parts[2]) if parts[2] else 0,
                    'price': float(parts[3]) if parts[3] else 0,
                    'high': float(parts[4]) if parts[4] else 0,
                    'low': float(parts[5]) if parts[5] else 0,
                    'volume': float(parts[8]) if parts[8] else 0,
                    'amount': float(parts[9]) if parts[9] else 0,
                    'date': parts[30] if len(parts) > 30 else '',
                    'time': parts[31] if len(parts) > 31 else '',
                }
        return result
    except Exception as e:
        log.debug(f"新浪实时行情获取失败: {e}")
        return {}


def get_ma_trend(close_series: pd.Series) -> tuple[str, str]:
    if len(close_series) < 60:
        return "数据不足", ""
    ma5 = close_series.rolling(5).mean().iloc[-1]
    ma20 = close_series.rolling(20).mean().iloc[-1]
    ma60 = close_series.rolling(60).mean().iloc[-1]
    close = close_series.iloc[-1]

    mas = [ma5, ma20, ma60]
    max_ma, min_ma = max(mas), min(mas)
    spread = (max_ma - min_ma) / min_ma if min_ma > 0 else 0

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


def analyze_market() -> str:
    log.info("📊 分析A股市场...")
    content = "### 📊 A股市场诊断\n\n"

    index_data = {}
    for symbol, name in INDEX_LIST:
        df = get_sina_kline(symbol, scale=240, datalen=120)
        if not df.empty:
            index_data[symbol] = {'name': name, 'df': df}

    if not index_data:
        content += "⚠️ 指数数据获取失败\n\n"
        return content

    sh_symbol = 'sh000001'
    sh_df = index_data.get(sh_symbol, {}).get('df')
    if sh_df is not None and len(sh_df) >= 2:
        close = sh_df['close']
        latest = close.iloc[-1]
        prev = close.iloc[-2]
        pct = (latest - prev) / prev * 100

        vol = sh_df['volume']
        if len(vol) >= 20:
            vol_ma5 = vol.rolling(5).mean().iloc[-1]
            vol_ma20 = vol.rolling(20).mean().iloc[-1]
            vol_surge = vol_ma5 > vol_ma20 * 1.25
        else:
            vol_surge = False

        ma20 = close.rolling(20).mean().iloc[-1]
        market_trend_ok = latest > ma20
        trend_name, trend_desc = get_ma_trend(close)

        if market_trend_ok:
            regime = "🔥 **强势多头 (BULL)**"
            advice = "仓位 60%-80%。赚钱效应佳，资金活跃，跟随主线积极做多。"
        else:
            regime = "🐻 **弱势震荡 (BEAR)**"
            advice = "仓位 30%-50%。均线压制，控制仓位，防守为主。"

        content += "**🌡️ 大盘温度**\n"
        content += f"- **上证指数**：`{latest:.2f}` ({pct:+.2f}%)\n"
        content += f"- **均线形态**：`{trend_name}` - {trend_desc}\n"
        content += f"- **综合判定**：{regime}\n"
        if vol_surge:
            content += f"- **量能状态**：🌊 大盘放量\n"
        content += f"- **💡 仓位建议**：{advice}\n\n"

    content += "**📉 主要指数表现**\n"
    for symbol, info in index_data.items():
        name = info['name']
        df = info['df']
        if len(df) < 2:
            continue
        close = df['close']
        latest = close.iloc[-1]
        prev = close.iloc[-2]
        pct = (latest - prev) / prev * 100

        ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else latest
        above_ma20 = latest > ma20
        trend_name, _ = get_ma_trend(close)

        icon = "🔴" if pct > 0 else "🟢" if pct < 0 else "⚪"
        content += (
            f"- {icon} **{name}**：`{latest:.2f}` ({pct:+.2f}%)\n"
            f"  趋势：`{trend_name}` | 站上MA20：{'✅' if above_ma20 else '❌'}\n"
        )

    content += "\n"
    return content


def calc_etf_score(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 60:
        return {}

    close = df['close']
    price = close.iloc[-1]
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]

    ret_5d = (price / close.iloc[-6] - 1) * 100 if len(close) > 6 else 0
    ret_20d = (price / close.iloc[-21] - 1) * 100 if len(close) > 21 else 0
    ret_60d = (price / close.iloc[-61] - 1) * 100 if len(close) > 61 else 0

    vol = df['volume']
    vol_ma5 = vol.rolling(5).mean().iloc[-1] if len(vol) >= 5 else vol.iloc[-1]
    vol_ma20 = vol.rolling(20).mean().iloc[-1] if len(vol) >= 20 else vol_ma5
    vol_ratio = vol_ma5 / vol_ma20 if vol_ma20 > 0 else 1.0

    above_ma20 = price > ma20
    ma_bullish = ma5 > ma20 > ma60

    score = 0.0
    score += 20 if above_ma20 else 0
    score += 20 if ma_bullish else 0
    score += min(20, max(0, ret_20d * 2))
    score += min(20, max(0, ret_60d))
    score += min(20, vol_ratio * 10) if vol_ratio > 1 else max(0, 10 + vol_ratio * 10)

    return {
        'price': round(price, 3),
        'ma5': round(ma5, 3),
        'ma20': round(ma20, 3),
        'ma60': round(ma60, 3),
        'ret_5d': round(ret_5d, 2),
        'ret_20d': round(ret_20d, 2),
        'ret_60d': round(ret_60d, 2),
        'vol_ratio': round(vol_ratio, 2),
        'above_ma20': above_ma20,
        'ma_bullish': ma_bullish,
        'score': round(score, 1),
    }


def analyze_etf_rotation() -> str:
    log.info("📊 分析ETF轮动...")
    results = []

    for full_code, short_code, name, category in ETF_UNIVERSE:
        try:
            df = get_sina_kline(full_code, scale=240, datalen=120)
            if df.empty:
                continue
            metrics = calc_etf_score(df)
            if not metrics:
                continue
            metrics['code'] = short_code
            metrics['name'] = name
            metrics['category'] = category
            results.append(metrics)
        except Exception as e:
            log.debug(f"ETF {short_code}({name}) 分析失败: {e}")

    if not results:
        return "### 📊 ETF轮动策略\n\n⚠️ ETF数据获取失败，暂无轮动分析数据。\n"

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('score', ascending=False)

    content = "### 📊 ETF轮动策略\n\n"

    cats = {
        '宽基': '🏦 宽基ETF强度排名',
        '行业': '🔥 行业ETF热度排名',
        '主题': '🎯 主题ETF排名',
        '海外': '🌍 海外ETF排名',
    }

    for cat, cat_title in cats.items():
        cat_df = df_results[df_results['category'] == cat]
        if cat_df.empty:
            continue
        content += f"**{cat_title}**\n"

        for i, (_, row) in enumerate(cat_df.head(5).iterrows()):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            trend_icon = "🚀" if row['ma_bullish'] else "📉" if not row['above_ma20'] else "⚖️"
            content += (
                f"{medal} **{row['name']}** (`{row['code']}`) {trend_icon}\n"
                f"   价格: `¥{row['price']}` | 强度分: `{row['score']}`\n"
                f"   5日: `{row['ret_5d']:+.2f}%` | 20日: `{row['ret_20d']:+.2f}%` | 60日: `{row['ret_60d']:+.2f}%`\n"
                f"   量比: `{row['vol_ratio']:.2f}` | 站上MA20: {'✅' if row['above_ma20'] else '❌'}\n"
            )
        content += "\n"

    top_etf = df_results.iloc[0]
    bottom_etf = df_results.iloc[-1]

    content += "**💡 轮动策略建议**\n"
    content += f"- 🏆 **强势龙头**：{top_etf['name']}({top_etf['code']}) 综合强度最高，趋势向上且量能配合，可作为核心配置。\n"
    content += f"- 🔻 **弱势规避**：{bottom_etf['name']}({bottom_etf['code']}) 处于弱势区间，建议暂时规避或等待右侧信号。\n"

    strong_count = len(df_results[df_results['ma_bullish']])
    total_count = len(df_results)
    if strong_count >= total_count * 0.7:
        content += "- 📈 **市场环境**：超过七成ETF处于多头排列，市场整体偏强，可适度提高权益仓位。\n"
    elif strong_count <= total_count * 0.3:
        content += "- 📉 **市场环境**：不足三成ETF处于多头排列，市场整体偏弱，建议防御为主，控制仓位。\n"
    else:
        content += "- ⚖️ **市场环境**：ETF强弱分化，结构性行情为主，建议均衡配置，择优布局。\n"

    return content


def analyze_hot_sectors() -> str:
    log.info("🌋 分析行业热点...")
    content = "### 🌋 行业热点追踪\n\n"

    sector_etfs = [
        ('sh512880', '证券'),
        ('sh512690', '白酒'),
        ('sh512010', '医药'),
        ('sh512660', '军工'),
        ('sh515030', '新能源车'),
        ('sh512400', '有色金属'),
        ('sh512760', '芯片半导体'),
        ('sh515050', '5G通信'),
        ('sh512980', '传媒'),
        ('sh512600', '红利'),
        ('sh513100', '纳斯达克'),
    ]

    results = []
    for symbol, name in sector_etfs:
        df = get_sina_kline(symbol, scale=240, datalen=30)
        if df.empty or len(df) < 2:
            continue
        close = df['close']
        pct = (close.iloc[-1] / close.iloc[-2] - 1) * 100
        ret_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 6 else 0
        results.append({'name': name, 'pct': pct, 'ret_5d': ret_5d})

    if not results:
        content += "⚠️ 行业ETF数据获取失败。\n"
        return content

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('pct', ascending=False)

    content += "**🔥 今日行业涨跌（ETF口径）**\n"
    for i, (_, row) in enumerate(df_results.iterrows()):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        icon = "🔴" if row['pct'] > 0 else "🟢"
        content += f"{medal} {icon} **{row['name']}**：`{row['pct']:+.2f}%` (5日 `{row['ret_5d']:+.2f}%`)\n"

    content += "\n> 💡 基于行业ETF表现的热点追踪，仅供参考。\n"
    return content


def analyze_stock_signals_simple() -> str:
    log.info("🎯 扫描强势股票...")
    content = "### 🎯 强势股扫描\n\n"

    index_df = get_sina_kline('sh000001', scale=240, datalen=10)
    if index_df.empty or len(index_df) < 2:
        sh_pct = 0
    else:
        close = index_df['close']
        sh_pct = (close.iloc[-1] / close.iloc[-2] - 1) * 100

    content += "**📊 大盘参考**\n"
    content += f"- 上证指数今日涨跌幅：`{sh_pct:+.2f}%`\n\n"

    sample_stocks = [
        ('sh600519', '600519', '贵州茅台'),
        ('sh601318', '601318', '中国平安'),
        ('sz000858', '000858', '五粮液'),
        ('sz002594', '002594', '比亚迪'),
        ('sz000333', '000333', '美的集团'),
        ('sh600036', '600036', '招商银行'),
        ('sz300750', '300750', '宁德时代'),
        ('sh601899', '601899', '紫金矿业'),
        ('sz002415', '002415', '海康威视'),
        ('sh600276', '600276', '恒瑞医药'),
    ]

    realtime = get_sina_realtime([s[0] for s in sample_stocks])
    if not realtime:
        content += "⚠️ 个股行情获取失败。\n"
        return content

    content += "**⚡ 核心蓝筹表现**\n\n"
    stock_data = []
    for full_code, short_code, name in sample_stocks:
        if full_code not in realtime:
            continue
        info = realtime[full_code]
        price = info['price']
        pre_close = info['pre_close']
        if pre_close > 0:
            pct = (price - pre_close) / pre_close * 100
        else:
            pct = 0
        stock_data.append({'code': short_code, 'name': name, 'price': price, 'pct': pct})

    if not stock_data:
        content += "暂无数据\n"
        return content

    df_stocks = pd.DataFrame(stock_data).sort_values('pct', ascending=False)

    for i, (_, row) in enumerate(df_stocks.iterrows(), 1):
        icon = "🔴" if row['pct'] > 0 else "🟢" if row['pct'] < 0 else "⚪"
        content += f"{i}. {icon} **{row['name']}** (`{row['code']}`): `¥{row['price']:.2f}` ({row['pct']:+.2f}%)\n"

    content += "\n> ⚠️ 以上仅为核心蓝筹样本展示，不构成投资建议。\n"
    return content


def generate_macro_section() -> str:
    log.info("🌍 生成宏观快报...")
    content = "### 🌍 隔夜外围与宏观指标\n\n"

    macro_etfs = [
        ('gb_$dji', '道琼斯'),
        ('gb_$ixic', '纳斯达克'),
        ('gb_$spx', '标普500'),
        ('hf_GC', 'COMEX黄金'),
        ('hf_CL', 'WTI原油'),
    ]

    realtime = get_sina_realtime([s[0] for s in macro_etfs])

    if realtime:
        for symbol, name in macro_etfs:
            if symbol in realtime:
                info = realtime[symbol]
                price = info.get('price', 0)
                pre_close = info.get('pre_close', 0)
                if pre_close > 0:
                    pct = (price - pre_close) / pre_close * 100
                else:
                    pct = 0
                icon = "🔴" if pct > 0 else "🟢" if pct < 0 else "⚪"
                content += f"- {icon} **{name}**：`{price:.2f}` ({pct:+.2f}%)\n"
        content += "\n> *数据源: 新浪财经*\n"
    else:
        content += "⚠️ 外围数据获取中...\n"

    return content


def generate_briefing() -> str:
    now = datetime.now(TZ_BJS)
    now_str = now.strftime('%Y-%m-%d %H:%M')
    weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]

    log.info("=" * 60)
    log.info(f"🚀 每日投研简报生成器启动 - {now_str} {weekday}")
    log.info("=" * 60)

    content = f"## 📈 每日投研简报\n> **{now_str} {weekday}**\n\n"
    content += "---\n\n"

    try:
        content += generate_macro_section()
        content += "\n---\n\n"
    except Exception as e:
        log.error(f"宏观快报失败: {e}")
        content += "### 🌍 隔夜外围\n⚠️ 获取失败\n\n---\n\n"

    try:
        content += analyze_market()
        content += "\n---\n\n"
    except Exception as e:
        log.error(f"市场分析失败: {e}")
        content += f"### 📊 A股市场诊断\n\n⚠️ 分析失败: {e}\n\n---\n\n"

    try:
        content += analyze_stock_signals_simple()
        content += "\n---\n\n"
    except Exception as e:
        log.error(f"股票信号失败: {e}")

    try:
        content += analyze_etf_rotation()
        content += "\n---\n\n"
    except Exception as e:
        log.error(f"ETF轮动失败: {e}")
        content += f"### 📊 ETF轮动策略\n\n⚠️ 分析失败: {e}\n\n---\n\n"

    try:
        content += analyze_hot_sectors()
        content += "\n"
    except Exception as e:
        log.error(f"行业热点失败: {e}")
        content += f"### 🌋 行业热点\n\n⚠️ 分析失败: {e}\n\n"

    content += "\n---\n\n"
    content += "> ⚠️ **免责声明**：本简报由AI量化系统自动生成，仅供参考学习，不构成任何投资建议。投资有风险，入市需谨慎。\n"
    content += f"> 🤖 生成时间：{now_str} | 数据来源：新浪财经等公开数据源\n"

    log.info("✅ 简报生成完成！")
    return content


def main():
    try:
        content = generate_briefing()

        if DINGTALK_WEBHOOK:
            log.info("📤 正在推送至钉钉...")
            success = send_dingtalk_notification('📈 每日投研简报', content)
            if success:
                log.info("✅ 钉钉推送完成！")
            else:
                log.warning("⚠️ 钉钉推送失败，将在本地输出...")
                print("\n" + "=" * 60)
                print(content)
                print("=" * 60 + "\n")
        else:
            log.warning("⚠️ 未配置 DINGTALK_WEBHOOK，在本地输出简报内容...")
            print("\n" + "=" * 60)
            print(content)
            print("=" * 60 + "\n")

    except Exception as e:
        log.critical(f"💥 简报生成失败: {e}", exc_info=True)
        error_msg = (
            f"🚨 **每日投研简报生成失败**\n\n"
            f"**时间**: {datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"**异常信息**: {str(e)[:300]}..."
        )
        if DINGTALK_WEBHOOK:
            send_dingtalk_notification('🚨 投研简报生成失败', error_msg)
        else:
            print(error_msg)
        sys.exit(1)


if __name__ == '__main__':
    main()
