#!/usr/bin/env python3
"""
每日投研简报生成器 (Daily Investment Research Briefing)
=====================================================
整合市场分析、股票信号、ETF轮动、行业热点，生成结构化每日简报并发送钉钉通知。

数据源策略：
- 优先使用本地 parquet 缓存数据 (沙箱/离线模式)
- 尝试实时 API 获取 (正常模式，有超时降级)
- 钉钉推送使用沙箱代理
"""

import os
import sys
import time
import json
import logging
import traceback
import socket
from datetime import datetime, timedelta
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# 沙箱代理配置：数据获取不需要代理（直接连不通），钉钉推送需要
_DINGTALK_PROXY = os.environ.get('HTTPS_PROXY', os.environ.get('https_proxy', ''))

# 清除全局代理，防止 akshare 请求走沙箱代理而被拒
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
os.environ['NO_PROXY'] = '*'

import numpy as np
import pandas as pd
import requests
import pytz

socket.setdefaulttimeout(10.0)

# ─────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────

TZ_BJS = pytz.timezone('Asia/Shanghai')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.quantbot_data')
DINGTALK_WEBHOOK = os.environ.get('DINGTALK_WEBHOOK', '')
FEISHU_WEBHOOK = os.environ.get('FEISHU_WEBHOOK', '')
NOTIFY_SEC_KEYWORD = os.environ.get('NOTIFY_SEC_KEYWORD', 'Hermes')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('daily_briefing')

# ─────────────────────────────────────────────────────────
# 本地数据源
# ─────────────────────────────────────────────────────────

def load_local_ashare() -> pd.DataFrame | None:
    """加载本地 A 股日线数据"""
    path = os.path.join(DATA_DIR, 'ashare_daily.parquet')
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        log.warning(f"加载本地 A 股数据失败: {e}")
        return None


def load_local_macro() -> pd.DataFrame | None:
    """加载本地宏观数据"""
    path = os.path.join(DATA_DIR, 'macro_daily.parquet')
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        log.warning(f"加载本地宏观数据失败: {e}")
        return None


# ─────────────────────────────────────────────────────────
# 1. 市场指数快报
# ─────────────────────────────────────────────────────────

def generate_market_index_report(ashare_df: pd.DataFrame) -> str:
    """基于本地数据生成市场广度快报 (本地缓存不含指数和 ETF，仅个股)"""
    if ashare_df is None or ashare_df.empty:
        return "### 🌍 市场广度快报\n⚠️ 无本地数据可用。\n\n"

    last_date = ashare_df['date'].max()
    last_cross = ashare_df[ashare_df['date'] == last_date].copy()

    if len(last_cross) < 20:
        return "### 🌍 市场广度快报\n⚠️ 截面数据不足。\n\n"

    last_cross['pctChg'] = pd.to_numeric(last_cross['pctChg'], errors='coerce')
    last_cross['close'] = pd.to_numeric(last_cross['close'], errors='coerce')
    last_cross['amount'] = pd.to_numeric(last_cross['amount'], errors='coerce')

    # 市场广度统计
    total = len(last_cross)
    up = int((last_cross['pctChg'] > 0).sum())
    down = int((last_cross['pctChg'] < 0).sum())
    flat = int((last_cross['pctChg'] == 0).sum())
    zt = int((last_cross['pctChg'] >= 9.0).sum())
    dt = int((last_cross['pctChg'] <= -9.0).sum())
    strong = int((last_cross['pctChg'] >= 5.0).sum())
    weak = int((last_cross['pctChg'] <= -5.0).sum())
    total_amt = last_cross['amount'].sum() / 1e8
    avg_pct = last_cross['pctChg'].mean()
    median_pct = last_cross['pctChg'].median()
    breadth = up / max(total, 1) * 100

    regime = "🔥 多头" if breadth > 60 else ("🐻 空头" if breadth < 40 else "⚖️ 震荡")
    sentiment = "🟢 积极" if avg_pct > 1 else ("🔴 恐慌" if avg_pct < -1 else "⚪ 中性")

    content = (
        f"### 🌍 市场广度快报 ({last_date.strftime('%Y-%m-%d')})\n\n"
        f"| 指标 | 数值 |\n|:---|:---:|\n"
        f"| 样本个股数 | `{total}` |\n"
        f"| 红盘 / 绿盘 / 平盘 | `{up}` / `{down}` / `{flat}` |\n"
        f"| 广度指标 | `{breadth:.1f}%` → {regime} |\n"
        f"| 涨停 / 跌停 | `{zt}` / `{dt}` |\n"
        f"| 强势(>5%) / 弱势(<-5%) | `{strong}` / `{weak}` |\n"
        f"| 平均涨跌幅 | `{avg_pct:+.2f}%` |\n"
        f"| 中位数涨跌幅 | `{median_pct:+.2f}%` → {sentiment} |\n"
        f"| 两市成交额 | 约 `{total_amt:.0f}` 亿元 |\n\n"
    )

    # 计算近5日广度趋势
    if len(ashare_df) > 0:
        dates = sorted(ashare_df['date'].unique())[-5:]
        breadth_history = []
        for d in dates:
            day = ashare_df[ashare_df['date'] == d]
            day_pct = pd.to_numeric(day['pctChg'], errors='coerce')
            b = int((day_pct > 0).sum()) / max(len(day), 1) * 100
            breadth_history.append(f"{d.strftime('%m-%d')}: `{b:.0f}%`")
        if len(breadth_history) >= 3:
            content += f"**近5日广度趋势**: {', '.join(breadth_history)}\n\n"

    content += "> *数据源: 本地缓存 (ashare_daily.parquet)*\n"
    return content


# ─────────────────────────────────────────────────────────
# 2. 宏观分析
# ─────────────────────────────────────────────────────────

def generate_macro_report(macro_df: pd.DataFrame) -> str:
    """基于本地宏观数据生成宏观分析"""
    if macro_df is None or macro_df.empty:
        return "### 🌍 宏观分析\n⚠️ 无本地宏观数据。\n\n"

    last = macro_df.iloc[-1]
    last_date = macro_df.index[-1].strftime('%Y-%m-%d')

    # 利差分析
    cn_10y = last.get('cn_10y', 0)
    us_10y = last.get('us_10y', 0)
    spread = last.get('us_cn_spread', 0)
    inversion = last.get('us_yield_curve_inversion', 0)
    cn_trend = last.get('cn_10y_trend', 0)

    spread_emoji = "⚠️" if spread > 2.0 else "✅"
    inv_emoji = "🚨" if inversion > 0 else "✅"
    cn_trend_emoji = "📈" if cn_trend > 0 else "📉"

    content = (
        f"### 🌍 宏观利率环境 ({last_date})\n"
        f"- **中国10年期国债**: `{cn_10y:.4f}%` ({cn_trend_emoji} 趋势 {cn_trend:+.4f})\n"
        f"- **美国10年期国债**: `{us_10y:.2f}%`\n"
        f"- **中美利差**: `{spread:.2f}%` {spread_emoji}\n"
        f"- **美债曲线倒挂**: `{inversion}` {inv_emoji} "
        + ("⚠️ 美债倒挂信号，衰退风险加大！" if inversion > 0 else "曲线正常") + "\n"
    )

    # 趋势历史
    if len(macro_df) >= 20:
        cn_20d = macro_df['cn_10y'].iloc[-20:].mean()
        us_20d = macro_df['us_10y'].iloc[-20:].mean()
        content += f"- **20日均值**: 中债 `{cn_20d:.4f}%` / 美债 `{us_20d:.2f}%`\n"

    staleness = last.get('macro_staleness_days', 0)
    if staleness > 2:
        content += f"\n> ⚠️ 宏观数据滞后 {int(staleness)} 天，请注意时效性。\n"

    content += "\n> *数据源: 本地缓存 (macro_daily.parquet)*\n"
    return content


# ─────────────────────────────────────────────────────────
# 3. 行业热点分析 (基于本地截面数据)
# ─────────────────────────────────────────────────────────

def generate_industry_hotspots(ashare_df: pd.DataFrame) -> str:
    """基于本地数据统计行业涨跌分布"""
    if ashare_df is None or ashare_df.empty:
        return "### 🌋 行业热点\n⚠️ 无本地数据。\n\n---\n\n"

    last_date = ashare_df['date'].max()
    cross = ashare_df[ashare_df['date'] == last_date].copy()

    if len(cross) < 50:
        return "### 🌋 行业热点\n⚠️ 截面数据不足。\n\n---\n\n"

    # 按涨跌幅排序
    cross['pctChg'] = pd.to_numeric(cross['pctChg'], errors='coerce')

    top10 = cross.nlargest(10, 'pctChg')
    bottom10 = cross.nsmallest(10, 'pctChg')

    # 涨幅梯队
    zt_count = (cross['pctChg'] >= 9.0).sum()
    dt_count = (cross['pctChg'] <= -9.0).sum()
    strong_count = (cross['pctChg'] >= 5.0).sum()
    weak_count = (cross['pctChg'] <= -5.0).sum()

    top_lines = []
    for _, row in top10.iterrows():
        emoji = "🔥" if row['pctChg'] > 5 else "📈"
        # 简化 code 显示 (去掉交易所前缀)
        display_code = row['code'].replace('sh.', '').replace('sz.', '')
        top_lines.append(f"- {emoji} **{display_code}** ({row['pctChg']:+.2f}%)")

    bottom_lines = []
    for _, row in bottom10.iterrows():
        emoji = "💀" if row['pctChg'] < -5 else "📉"
        display_code = row['code'].replace('sh.', '').replace('sz.', '')
        bottom_lines.append(f"- {emoji} **{display_code}** ({row['pctChg']:+.2f}%)")

    content = (
        f"### 🌋 行业热点分析 ({last_date.strftime('%Y-%m-%d')})\n\n"
        f"#### 📊 涨跌梯队统计\n"
        f"- 涨停 `{zt_count}` 只 | 跌停 `{dt_count}` 只\n"
        f"- 强势(涨幅>5%) `{strong_count}` 只 | 弱势(跌幅>5%) `{weak_count}` 只\n\n"
        f"#### 📈 领涨 Top 10\n" + "\n".join(top_lines) + "\n\n"
        f"#### 📉 领跌 Bottom 10\n" + "\n".join(bottom_lines) + "\n\n"
        f"> *数据源: 本地缓存 (ashare_daily.parquet)*\n\n---\n\n"
    )
    return content


# ─────────────────────────────────────────────────────────
# 4. ETF 轮动分析 (基于本地数据中的 ETF)
# ─────────────────────────────────────────────────────────

INDUSTRY_ETF_MAP = {
    'sh.510300': '沪深300ETF', 'sh.510500': '中证500ETF', 'sh.510050': '上证50ETF',
    'sz.159949': '创业板ETF', 'sh.512100': '中证1000ETF',
    'sh.512660': '军工ETF', 'sh.512880': '证券ETF', 'sh.512010': '医药ETF',
    'sz.512170': '医疗ETF', 'sh.512480': '半导体ETF', 'sz.515030': '新能源车ETF',
    'sz.516790': '光伏ETF', 'sz.515880': '通信ETF', 'sh.512800': '银行ETF',
    'sh.518880': '黄金ETF', 'sh.512400': '有色金属ETF', 'sz.512690': '酒ETF',
    'sz.159928': '消费ETF', 'sh.512200': '房地产ETF', 'sh.513100': '纳指ETF',
    'sh.512090': '红利ETF', 'sh.515220': '煤炭ETF', 'sz.159825': '农业ETF',
}


def generate_etf_rotation(ashare_df: pd.DataFrame) -> str:
    """基于个股动量推算板块轮动方向 (本地缓存不含 ETF，按个股涨跌分布推算行业强弱)"""
    if ashare_df is None or ashare_df.empty:
        return "### 🔄 板块轮动方向\n\n⚠️ 无本地数据可用。\n\n---\n\n"

    last_date = ashare_df['date'].max()
    cross = ashare_df[ashare_df['date'] == last_date].copy()
    cross['pctChg'] = pd.to_numeric(cross['pctChg'], errors='coerce')
    cross['close'] = pd.to_numeric(cross['close'], errors='coerce')
    cross['amount'] = pd.to_numeric(cross['amount'], errors='coerce')

    # 按代码前缀分类板块 (简化版行业分类)
    # sh.600xxx / sh.601xxx / sh.603xxx → 上交所主板
    # sz.000xxx → 深交所主板
    # sz.002xxx → 中小板
    # sz.300xxx → 创业板

    sector_map = {
        '创业板': cross[cross['code'].str.startswith('sz.300')],
        '中小板': cross[cross['code'].str.startswith('sz.002')],
        '沪主板大盘': cross[cross['code'].str.match(r'^sh\.60[01]\d{3}$')],
        '沪主板中盘': cross[cross['code'].str.startswith('sh.603')],
    }

    lines = []
    for sector, stocks in sector_map.items():
        if len(stocks) < 5:
            continue
        avg_pct = stocks['pctChg'].mean()
        up_ratio = int((stocks['pctChg'] > 0).sum()) / max(len(stocks), 1) * 100
        total_amt = stocks['amount'].sum() / 1e8
        avg_close = stocks['close'].mean()

        # 5日动量 (需要对每只股票回溯)
        codes = stocks['code'].tolist()[:30]  # 取前30只计算
        ret_5d_list = []
        for code in codes:
            hist = ashare_df[ashare_df['code'] == code].sort_values('date')
            if len(hist) >= 6:
                closes = pd.to_numeric(hist['close'], errors='coerce')
                ret_5d_list.append((closes.iloc[-1] / closes.iloc[-6] - 1) * 100)

        avg_ret_5d = np.mean(ret_5d_list) if ret_5d_list else avg_pct

        emoji = "🟢" if avg_pct > 0 else "🔴"
        trend = "多头" if avg_ret_5d > 0 else "空头"
        lines.append(
            f"| {emoji} **{sector}** | `{len(stocks)}` 只 | "
            f"`{avg_pct:+.2f}%` | `{avg_ret_5d:+.2f}%` | "
            f"`{up_ratio:.0f}%` | `{total_amt:.0f}`亿 | "
            f"{'🟢' if trend == '多头' else '🔴'} {trend} |"
        )

    if not lines:
        return "### 🔄 板块轮动方向\n\n⚠️ 截面数据不足以计算板块动量。\n\n---\n\n"

    # 按平均涨跌幅排序
    lines.sort(key=lambda x: float(x.split("`")[1].replace("%", "")), reverse=True)

    content = (
        "### 🔄 板块轮动方向 (按板块平均涨幅排序)\n\n"
        "| 板块 | 个股数 | 日均涨幅 | 5日均涨幅 | 红盘率 | 成交额 | 动量 |\n"
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n"
        + "\n".join(lines) + "\n\n"
    )

    # 轮动建议
    strongest = [l for l in lines if "🟢" in l][:2]
    weakest = [l for l in lines if "🔴" in l][:2]
    if strongest:
        names = [l.split("**")[1] for l in strongest]
        content += f"> 💡 **轮动建议**：当前 `{', '.join(names)}` 板块动量较强，可关注趋势延续。\n\n"
    if weakest:
        names = [l.split("**")[1] for l in weakest]
        content += f"> ⚠️ **风险提示**：`{', '.join(names)}` 板块动量偏弱，建议回避。\n\n"

    content += "> *数据源: 本地缓存 (ashare_daily.parquet)，板块按代码前缀简化分类*\n\n---\n\n"
    return content


# ─────────────────────────────────────────────────────────
# 5. 股票信号 (基于本地数据量化筛选)
# ─────────────────────────────────────────────────────────

def generate_stock_signals(ashare_df: pd.DataFrame) -> str:
    """基于本地数据筛选具有动量优势的个股"""
    if ashare_df is None or ashare_df.empty:
        return "### 🎯 量化选股信号\n⚠️ 无本地数据。\n\n"

    last_date = ashare_df['date'].max()
    # 排除 ST 和停牌
    cross = ashare_df[ashare_df['date'] == last_date].copy()
    cross = cross[(cross['isST'] == 0) & (cross['tradestatus'] == 1)]
    cross['pctChg'] = pd.to_numeric(cross['pctChg'], errors='coerce')
    cross['close'] = pd.to_numeric(cross['close'], errors='coerce')
    cross['amount'] = pd.to_numeric(cross['amount'], errors='coerce')
    cross['turn'] = pd.to_numeric(cross['turn'], errors='coerce')

    # 排除科创板、北交所
    cross = cross[~cross['code'].str.startswith(('sh.688', 'bj.', 'sz.8', 'sz.4', 'sz.9'))]

    # 量能过滤 (成交额 > 1亿)
    cross = cross[cross['amount'] > 1e8]

    if len(cross) < 20:
        return "### 🎯 量化选股信号\n⚠️ 截面数据不足以筛选。\n\n"

    # 对每只股票计算多日动量
    top_candidates = []

    for code in cross['code'].unique():
        hist = ashare_df[ashare_df['code'] == code].sort_values('date')
        if len(hist) < 30:
            continue

        closes = pd.to_numeric(hist['close'], errors='coerce')
        pcts = pd.to_numeric(hist['pctChg'], errors='coerce')

        # 动量指标
        ret_5d = (closes.iloc[-1] / closes.iloc[-6] - 1) * 100 if len(closes) >= 6 else 0
        ret_20d = (closes.iloc[-1] / closes.iloc[-21] - 1) * 100 if len(closes) >= 21 else 0

        # 均线排列
        ma5 = closes.rolling(5).mean().iloc[-1]
        ma20 = closes.rolling(20).mean().iloc[-1]
        ma_align = ma5 > ma20

        # 量价配合 (近5日平均成交额 vs 20日)
        amt = pd.to_numeric(hist['amount'], errors='coerce')
        vol_ratio = amt.iloc[-5:].mean() / amt.iloc[-20:].mean() if amt.iloc[-20:].mean() > 0 else 1.0

        # 综合评分
        score_raw = (
            ret_5d * 0.3 +          # 短期动量
            ret_20d * 0.3 +          # 中期动量
            (10 if ma_align else -10) +  # 均线排列
            (vol_ratio - 1) * 15 * 0.2   # 量能配合
        )
        score = float(f"{score_raw:.1f}")

        latest = hist.iloc[-1]
        top_candidates.append({
            'code': code,
            'close': float(latest['close']),
            'pctChg': float(latest['pctChg']),
            'amount': float(latest['amount']),
            'turn': float(latest.get('turn', 0)),
            'score': score,
            'ret_5d': round(ret_5d, 2),
            'ret_20d': round(ret_20d, 2),
            'trend': '多头' if ma_align else '空头',
        })

    top_candidates.sort(key=lambda x: x['score'], reverse=True)

    # 分级
    core = [c for c in top_candidates if c['score'] >= 15][:5]
    satellite = [c for c in top_candidates if 5 <= c['score'] < 15][:10]

    total_pool = len(cross)
    total_mkt = len(ashare_df[ashare_df['date'] == last_date])

    content = f"### 🎯 量化选股信号\n\n"
    content += f"**漏斗**：全市场 `{total_mkt}` → 量能筛选 `{total_pool}` → 高动量 `{len(core) + len(satellite)}`\n\n"

    if core:
        content += "#### 🔥 核心主力池 (高动量 + 均线多头)\n\n"
        for c in core:
            content += f"- **{c['code']}** | ¥{c['close']:.2f} ({c['pctChg']:+.2f}%) | "
            content += f"5日 {c['ret_5d']:+.2f}% | 20日 {c['ret_20d']:+.2f}% | "
            content += f"评分 `{c['score']}` | {c['trend']}\n"
        content += "\n"

    if satellite:
        content += "#### 🛰️ 卫星观察池 (动量尚可)\n\n"
        for c in satellite:
            content += f"- **{c['code']}** | ¥{c['close']:.2f} ({c['pctChg']:+.2f}%) | "
            content += f"5日 {c['ret_5d']:+.2f}% | 评分 `{c['score']}`\n"
        content += "\n"

    if not core and not satellite:
        content += "✅ 今日未发现强动量标的，建议**空仓防守**。\n\n"

    content += "> *数据源: 本地缓存 (ashare_daily.parquet)*\n\n"
    return content


# ─────────────────────────────────────────────────────────
# 通知推送
# ─────────────────────────────────────────────────────────

def send_to_dingtalk(title: str, content: str) -> None:
    """发送简报到钉钉 (使用沙箱代理)"""
    if not DINGTALK_WEBHOOK:
        log.warning("⚠️ DINGTALK_WEBHOOK 未配置，简报仅本地输出。")
        print("\n" + "=" * 60)
        print(content[:5000])
        if len(content) > 5000:
            print(f"\n... (共 {len(content)} 字符)")
        print("=" * 60)
        return

    # 注入关键词
    if NOTIFY_SEC_KEYWORD and NOTIFY_SEC_KEYWORD not in title:
        title = f"{NOTIFY_SEC_KEYWORD} | {title}"
    if NOTIFY_SEC_KEYWORD and NOTIFY_SEC_KEYWORD not in content:
        content = f"### {NOTIFY_SEC_KEYWORD}\n\n{content}"

    # 钉钉 markdown 消息
    payload = {
        'msgtype': 'markdown',
        'markdown': {
            'title': title,
            'text': content,
        }
    }

    headers = {"Content-Type": "application/json"}

    # 使用沙箱代理推送钉钉
    proxies = {}
    if _DINGTALK_PROXY:
        proxies = {'http': _DINGTALK_PROXY, 'https': _DINGTALK_PROXY}

    CHUNK_SIZE = 18000
    chunks = [content[i:i+CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]

    for idx, chunk in enumerate(chunks[:3]):
        text = chunk if idx == 0 else f"_(续上条)_\n\n{chunk}"
        msg_title = title if len(chunks) == 1 else f"{title} (Part {idx+1}/{len(chunks)})"
        payload['markdown']['title'] = msg_title
        payload['markdown']['text'] = text

        for attempt in range(2):
            try:
                res = requests.post(DINGTALK_WEBHOOK, json=payload, headers=headers,
                                    timeout=10, proxies=proxies)
                res.raise_for_status()
                res_dict = res.json()
                if res_dict.get('errcode', 0) != 0:
                    log.error(f"❌ 钉钉推送接口拒绝: {res_dict}")
                else:
                    log.info(f"✅ 钉钉推送成功 (Part {idx+1})")
                break
            except Exception as e:
                if attempt == 1:
                    log.error(f"❌ 钉钉推送失败: {e}")
                else:
                    time.sleep(1)

        if idx < len(chunks) - 1:
            time.sleep(1)


def send_to_feishu(title: str, content: str) -> None:
    """发送简报到飞书 (使用沙箱代理)"""
    if not FEISHU_WEBHOOK:
        return

    sec_keyword = NOTIFY_SEC_KEYWORD
    if sec_keyword and sec_keyword not in title:
        title = f"{sec_keyword} | {title}"

    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": "blue"
            },
            "elements": [{"tag": "markdown", "content": content}]
        }
    }

    headers = {"Content-Type": "application/json"}
    proxies = {}
    if _DINGTALK_PROXY:
        proxies = {'http': _DINGTALK_PROXY, 'https': _DINGTALK_PROXY}

    try:
        res = requests.post(FEISHU_WEBHOOK, json=payload, headers=headers,
                            timeout=10, proxies=proxies)
        res.raise_for_status()
        res_dict = res.json()
        if res_dict.get('code', 0) != 0:
            log.error(f"❌ 飞书推送接口拒绝: {res_dict}")
        else:
            log.info("✅ 飞书推送成功")
    except Exception as e:
        log.error(f"❌ 飞书推送失败: {e}")


# ─────────────────────────────────────────────────────────
# 简报组装
# ─────────────────────────────────────────────────────────

def generate_daily_briefing() -> str:
    """生成完整每日投研简报"""

    now_ts = datetime.now(TZ_BJS)
    date_str = now_ts.strftime('%Y年%m月%d日')
    now_str = now_ts.strftime('%Y-%m-%d %H:%M')

    # 加载本地数据
    log.info("📦 加载本地缓存数据...")
    ashare_df = load_local_ashare()
    macro_df = load_local_macro()

    if ashare_df is not None and not ashare_df.empty:
        data_date = ashare_df['date'].max().strftime('%Y-%m-%d')
        log.info(f"📊 A股数据: {len(ashare_df)} 条, 最新日期 {data_date}")
    else:
        log.warning("⚠️ 无本地 A 股数据!")

    if macro_df is not None and not macro_df.empty:
        macro_date = macro_df.index[-1].strftime('%Y-%m-%d')
        log.info(f"🌍 宏观数据: {len(macro_df)} 条, 最新日期 {macro_date}")
    else:
        log.warning("⚠️ 无本地宏观数据!")

    parts = []

    # ── 头部 ──
    parts.append(f"## 📋 每日投研简报\n> **{date_str}** | 生成于 {now_str}\n\n---\n\n")

    # ── 1. 市场指数 ──
    log.info("🌍 [1/5] 生成市场指数快报...")
    parts.append(generate_market_index_report(ashare_df) + "\n\n---\n\n")

    # ── 2. 宏观分析 ──
    log.info("📊 [2/5] 生成宏观分析...")
    parts.append(generate_macro_report(macro_df) + "\n\n---\n\n")

    # ── 3. 行业热点 ──
    log.info("🌋 [3/5] 生成行业热点分析...")
    parts.append(generate_industry_hotspots(ashare_df))

    # ── 4. ETF 轮动 ──
    log.info("🔄 [4/5] 生成ETF轮动排行...")
    parts.append(generate_etf_rotation(ashare_df))

    # ── 5. 股票信号 ──
    log.info("🎯 [5/5] 生成股票信号...")
    parts.append(generate_stock_signals(ashare_df))

    # ── 尾部 ──
    parts.append(
        "---\n\n"
        "> ⚠️ **免责声明**：本简报由 AI 量化系统基于本地缓存数据自动生成，仅供投资研究参考，不构成任何投资建议。"
        "数据时效性受限于本地缓存更新频率，请结合实时行情综合判断。\n"
    )

    return "".join(parts)


def main():
    log.info("=" * 50)
    log.info("📋 每日投研简报生成器启动")
    log.info("=" * 50)

    try:
        content = generate_daily_briefing()
        log.info(f"📝 简报生成完成，共 {len(content)} 字符")

        # 保存本地
        date_str = datetime.now(TZ_BJS).strftime('%Y%m%d')
        filepath = f"daily_briefing_{date_str}.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        log.info(f"💾 简报已保存至 {filepath}")

        # 推送钉钉
        now_ts = datetime.now(TZ_BJS)
        title = f"每日投研简报 {now_ts.strftime('%m-%d')}"
        send_to_dingtalk(title, content)
        send_to_feishu(title, content)

        log.info("✅ 每日投研简报生成与推送完成！")

    except Exception as e:
        log.critical(f"🚨 简报生成崩溃: {e}", exc_info=True)


if __name__ == '__main__':
    main()
