#!/usr/bin/env python3
"""
每日投研简报生成器 - 离线版本
基于本地缓存数据生成简报（不依赖网络）
"""

import os
import sys
import glob
import pickle
import logging
from datetime import datetime
import json

import pandas as pd
import numpy as np
import pytz

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

TZ_BJS = pytz.timezone('Asia/Shanghai')


def load_hist_cache():
    """加载本地历史缓存数据"""
    cache_dir = 'hist_cache'
    files = glob.glob(os.path.join(cache_dir, '*.parquet'))

    data_list = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            data_list.append(df)
        except Exception as e:
            log.warning(f"读取缓存失败 {f}: {e}")

    if not data_list:
        return pd.DataFrame()

    # 标准化列名
    normalized = []
    for df in data_list:
        col_map = {}
        for col in df.columns:
            if '日期' in col:
                col_map[col] = 'date'
            elif '开盘' in col:
                col_map[col] = 'open'
            elif '收盘' in col:
                col_map[col] = 'close'
            elif '最高' in col:
                col_map[col] = 'high'
            elif '最低' in col:
                col_map[col] = 'low'
            elif '成交量' in col:
                col_map[col] = 'volume'
        df = df.rename(columns=col_map)
        normalized.append(df)

    combined = pd.concat(normalized, ignore_index=True)

    # 从文件名提取股票代码
    def extract_code_from_filename(fname):
        # 文件名格式: hist_000027_20260601.parquet
        basename = os.path.basename(fname)
        parts = basename.replace('.parquet', '').split('_')
        if len(parts) >= 2:
            return parts[1]  # 如 000027
        return 'unknown'

    # 为没有code列的数据添加code
    if 'code' not in combined.columns:
        combined['code'] = combined.index.map(lambda i: 'unknown')

    return combined


def load_ashare_data():
    """加载全市场日线数据"""
    try:
        df = pd.read_parquet('.quantbot_data/ashare_daily.parquet')
        return df
    except Exception as e:
        log.warning(f"加载ashare_daily失败: {e}")
        return pd.DataFrame()


def load_macro_data():
    """加载宏观数据"""
    try:
        df = pd.read_parquet('.quantbot_data/macro_daily.parquet')
        return df
    except Exception as e:
        log.warning(f"加载macro_daily失败: {e}")
        return pd.DataFrame()


def load_core_pool():
    """加载核心股票池"""
    try:
        with open('hist_cache/core_pool.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        log.warning(f"加载core_pool失败: {e}")
        return set()


def get_latest_date(df):
    """获取数据的最新日期"""
    if df.empty:
        return None
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        return df['date'].max()
    return None


def generate_market_overview(ashare_df, macro_df):
    """生成市场概览"""
    lines = ["### 📊 市场概览\n"]

    # A股数据概览
    if not ashare_df.empty:
        latest_date = get_latest_date(ashare_df)
        if latest_date:
            lines.append(f"**数据日期**: {latest_date.strftime('%Y-%m-%d')} (缓存数据)\n")

            # 当日涨跌统计
            latest_data = ashare_df[ashare_df['date'] == latest_date]
            if not latest_data.empty:
                total = len(latest_data)
                up = (latest_data['pctChg'] > 0).sum()
                down = (latest_data['pctChg'] < 0).sum()
                zt = (latest_data['pctChg'] >= 9.5).sum()
                dt = (latest_data['pctChg'] <= -9.5).sum()

                up_pct = up / total * 100 if total > 0 else 0

                lines.append(f"- **市场广度**: 红盘 `{up}` 家 / 绿盘 `{down}` 家")
                lines.append(f"- **涨跌停**: 涨停 `{zt}` / 跌停 `{dt}`")
                lines.append(f"- **上涨占比**: {up_pct:.1f}%\n")

    # 宏观数据
    if not macro_df.empty:
        latest_macro = macro_df.iloc[-1] if not macro_df.empty else None
        if latest_macro is not None:
            lines.append("**宏观指标**:\n")
            lines.append(f"- 中国10年期国债: `{latest_macro.get('cn_10y', 'N/A'):.4f}`%")
            lines.append(f"- 美国10年期国债: `{latest_macro.get('us_10y', 'N/A'):.4f}`%")
            lines.append(f"- 中美利差: `{latest_macro.get('us_cn_spread', 'N/A'):.4f}`%")
            lines.append(f"- 美债收益率曲线: {'倒挂' if latest_macro.get('us_yield_curve_inversion', 0) == 1 else '正常'}")
            lines.append(f"- 曲线利差: `{latest_macro.get('us_yield_curve_spread', 'N/A'):.4f}`\n")

    return "\n".join(lines)


def generate_top_movers(ashare_df, top_n=10):
    """生成涨跌停TOP榜"""
    lines = ["### 🏆 涨跌停TOP榜\n"]

    if ashare_df.empty:
        return "\n".join(lines) + "> 无数据\n"

    latest_date = get_latest_date(ashare_df)
    if latest_date is None:
        return "\n".join(lines) + "> 无数据\n"

    latest_data = ashare_df[ashare_df['date'] == latest_date].copy()

    if latest_data.empty:
        return "\n".join(lines) + "> 无当日数据\n"

    # 涨幅榜
    top_gainers = latest_data.nlargest(top_n, 'pctChg')
    lines.append("\n**涨幅榜**:\n")
    lines.append("| 排名 | 代码 | 收盘价 | 涨跌幅 |")
    lines.append("|------|------|--------|--------|")
    for i, (_, row) in enumerate(top_gainers.iterrows(), 1):
        code = str(row.get('code', 'N/A')).replace('sh.', '').replace('sz.', '')
        pct = row.get('pctChg', 0)
        close = row.get('close', 0)
        lines.append(f"| {i} | {code} | {close:.2f} | {pct:+.2f}% |")

    # 跌幅榜
    top_losers = latest_data.nsmallest(top_n, 'pctChg')
    lines.append("\n**跌幅榜**:\n")
    lines.append("| 排名 | 代码 | 收盘价 | 涨跌幅 |")
    lines.append("|------|------|--------|--------|")
    for i, (_, row) in enumerate(top_losers.iterrows(), 1):
        code = str(row.get('code', 'N/A')).replace('sh.', '').replace('sz.', '')
        pct = row.get('pctChg', 0)
        close = row.get('close', 0)
        lines.append(f"| {i} | {code} | {close:.2f} | {pct:+.2f}% |")

    return "\n".join(lines)


def generate_etf_analysis(ashare_df):
    """生成ETF分析"""
    lines = ["### 📊 ETF基金分析\n"]

    if ashare_df.empty:
        return "\n".join(lines) + "> 无数据\n"

    # 筛选ETF（代码以51/15/56/588开头）
    latest_date = get_latest_date(ashare_df)
    if latest_date is None:
        return "\n".join(lines) + "> 无数据\n"

    latest_data = ashare_df[ashare_df['date'] == latest_date].copy()

    # ETF代码判断 - 匹配 .51 或 .15 或 .56 或 .588 结尾
    etf_mask = latest_data['code'].astype(str).str.contains(r'\.(51|15|56|588)', regex=True)
    etfs = latest_data[etf_mask].copy()

    if etfs.empty:
        return "\n".join(lines) + "> 缓存数据中无ETF记录\n"

    # 按成交量排序取前10
    top_etfs = etfs.nlargest(10, 'volume')

    lines.append("| 代码 | 收盘价 | 涨跌幅 | 成交量 |")
    lines.append("|------|--------|--------|--------|")
    for _, row in top_etfs.iterrows():
        code = str(row.get('code', 'N/A')).replace('sh.', '').replace('sz.', '')
        pct = row.get('pctChg', 0)
        close = row.get('close', 0)
        vol = row.get('volume', 0) / 1e6  # 转换为百万
        lines.append(f"| {code} | {close:.3f} | {pct:+.2f}% | {vol:.1f}M |")

    return "\n".join(lines)


def generate_sector_hotspots(hist_cache_df):
    """生成行业热点"""
    lines = ["### 🔥 行业热点\n"]

    if hist_cache_df.empty:
        return "\n".join(lines) + "> 无数据\n"

    # 从hist_cache文件名提取的股票代码对应的板块信息
    # 由于缓存中无板块信息，这里使用成交量最大的股票作为代理

    latest_date = get_latest_date(hist_cache_df)
    if latest_date is None:
        return "\n".join(lines) + "> 无数据\n"

    latest_data = hist_cache_df[hist_cache_df['date'] == latest_date].copy()

    if latest_data.empty:
        return "\n".join(lines) + "> 无当日数据\n"

    # 按成交量排序
    top_vol = latest_data.nlargest(20, 'volume')

    lines.append("**今日活跃股（按成交量）**:\n")
    lines.append("| 代码 | 收盘价 | 涨跌幅 | 成交量 |")
    lines.append("|------|--------|--------|--------|")

    for _, row in top_vol.head(10).iterrows():
        code = str(row.get('code', 'N/A'))
        close = row.get('close', 0)
        pct = row.get('pctChg', 0) if 'pctChg' in row else 0
        vol = row.get('volume', 0) / 1e6
        lines.append(f"| {code} | {close:.2f} | {pct:+.2f}% | {vol:.1f}M |")

    return "\n".join(lines)


def generate_ai_signals_summary():
    """生成AI量化信号摘要"""
    lines = ["### 📡 AI量化信号\n"]

    # 尝试读取历史推荐
    try:
        with open('advisory_tracker.json', 'r') as f:
            tracker = json.load(f) if 'json' in sys.modules else {}

        if tracker:
            lines.append(f"**跟踪中的标的**: {len(tracker)} 只\n")
            for code, info in list(tracker.items())[:5]:
                name = info.get('name', 'N/A')
                target = info.get('target', 0)
                stop = info.get('stop', 0)
                lines.append(f"- `{code}` {name} - 目标: ¥{target} / 止损: ¥{stop}")
        else:
            lines.append("> 暂无跟踪中的标的")
    except Exception as e:
        lines.append(f"> 无法读取信号跟踪数据: {e}")

    return "\n".join(lines)


def generate_daily_briefing():
    """生成完整简报"""
    now = datetime.now(TZ_BJS)
    now_str = now.strftime('%Y-%m-%d %H:%M')

    log.info("🚀 开始生成每日投研简报...")

    sections = []

    # 加载数据
    log.info("📂 加载本地缓存数据...")
    ashare_df = load_ashare_data()
    macro_df = load_macro_data()
    hist_cache_df = load_hist_cache()
    core_pool = load_core_pool()

    latest_data_date = get_latest_date(ashare_df) or get_latest_date(hist_cache_df)

    # 1. 市场概览
    try:
        sections.append(generate_market_overview(ashare_df, macro_df))
        log.info("✅ 市场概览完成")
    except Exception as e:
        log.warning(f"市场概览失败: {e}")
        sections.append("### 市场概览\n> 数据生成失败\n")

    # 2. 涨跌停TOP
    try:
        sections.append(generate_top_movers(ashare_df))
        log.info("✅ 涨跌停TOP完成")
    except Exception as e:
        log.warning(f"涨跌停TOP失败: {e}")
        sections.append("### 涨跌停TOP\n> 数据生成失败\n")

    # 3. ETF分析
    try:
        sections.append(generate_etf_analysis(ashare_df))
        log.info("✅ ETF分析完成")
    except Exception as e:
        log.warning(f"ETF分析失败: {e}")
        sections.append("### ETF分析\n> 数据生成失败\n")

    # 4. 行业热点
    try:
        sections.append(generate_sector_hotspots(hist_cache_df))
        log.info("✅ 行业热点完成")
    except Exception as e:
        log.warning(f"行业热点失败: {e}")
        sections.append("### 行业热点\n> 数据生成失败\n")

    # 5. AI信号
    try:
        sections.append(generate_ai_signals_summary())
        log.info("✅ AI信号完成")
    except Exception as e:
        log.warning(f"AI信号失败: {e}")
        sections.append("### AI信号\n> 数据生成失败\n")

    # 组装简报
    data_date_str = latest_data_date.strftime('%Y-%m-%d') if latest_data_date else '未知'

    title = f"📊 每日投研简报 | {data_date_str} (离线缓存)"

    header = f"""# {title}
> 生成时间: {now_str}
> ⚠️ **离线模式** - 数据截止至 {data_date_str}，可能非最新

---

"""

    content = header + "\n\n---\n\n".join(sections)

    footer = """

---

> 📌 **免责声明**: 本简报基于本地缓存数据生成，仅供参考，不构成投资建议。
> 🔔 如需获取最新实时数据，请在可访问外部网络的环境运行本系统。
"""
    content += footer

    return title, content


def main():
    """主函数"""
    log.info("=" * 50)
    log.info("📊 每日投研简报生成器 (离线版) 启动")
    log.info("=" * 50)

    try:
        title, content = generate_daily_briefing()

        # 打印简报
        print("\n" + "=" * 60)
        print("📋 每日投研简报:")
        print("=" * 60)
        print(content)
        print("=" * 60)

        # 保存到文件
        output_file = f"daily_briefing_{datetime.now(TZ_BJS).strftime('%Y%m%d_%H%M')}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        log.info(f"💾 简报已保存至: {output_file}")

        return 0

    except Exception as e:
        log.critical(f"💥 简报生成失败: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
