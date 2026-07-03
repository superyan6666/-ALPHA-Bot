import os
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
os.environ['NO_PROXY'] = '*'

import time
import json
import logging
from datetime import datetime, timedelta
from collections import Counter

import requests
import numpy as np
import pandas as pd
import pytz

from main import (
    AppConfig, DataProxy, LocalDataLake, NotificationGateway,
    Signal, Config, Cols, C, generate_macro_section,
    get_ma_trend, fetch_spot, fetch_hist, fetch_index,
    fetch_core_pool, fetch_hot_sectors, fetch_northbound_flow,
    TZ_BJS, _today_str, log
)

config = AppConfig()

ETF_ROTATION_POOL = {
    '510300': '沪深300ETF',
    '510500': '中证500ETF',
    '512100': '中证1000ETF',
    '159915': '创业板ETF',
    '588000': '科创50ETF',
    '512880': '证券ETF',
    '512690': '酒ETF',
    '512010': '医药ETF',
    '515030': '新能源车ETF',
    '512480': '半导体ETF',
    '512660': '军工ETF',
    '515790': '光伏ETF',
    '512400': '有色ETF',
    '515210': '钢铁ETF',
    '510410': '资源ETF',
    '512000': '券商ETF',
    '159928': '消费ETF',
    '515000': '科技ETF',
    '512760': '芯片ETF',
    '515050': '5GETF',
}

def get_etf_rotation_analysis() -> str:
    try:
        log.info("📊 开始计算ETF轮动信号...")
        etf_data = []
        end_date = datetime.now(TZ_BJS).strftime('%Y%m%d')
        start_date = (datetime.now(TZ_BJS) - timedelta(days=120)).strftime('%Y%m%d')

        for code, name in ETF_ROTATION_POOL.items():
            try:
                df = fetch_hist(code, start_date, end_date)
                if df is None or len(df) < 20:
                    continue
                close = df[C.H_CLOSE].astype(float)
                high = df[C.H_HIGH].astype(float)
                low = df[C.H_LOW].astype(float)
                vol = df[C.H_VOL].astype(float)

                ma5 = close.rolling(5).mean()
                ma10 = close.rolling(10).mean()
                ma20 = close.rolling(20).mean()
                ma60 = close.rolling(60).mean() if len(close) >= 60 else close.rolling(len(close)).mean()

                latest_close = float(close.iloc[-1])
                pct_5d = (latest_close / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 else 0
                pct_20d = (latest_close / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0
                pct_60d = (latest_close / float(close.iloc[-61]) - 1) * 100 if len(close) >= 61 else 0

                vol_ma5 = vol.rolling(5).mean()
                vol_ratio = float(vol.iloc[-1] / vol_ma5.iloc[-1]) if vol_ma5.iloc[-1] > 0 else 1.0

                ma_score = 0
                if latest_close > ma5.iloc[-1]: ma_score += 1
                if latest_close > ma10.iloc[-1]: ma_score += 1
                if latest_close > ma20.iloc[-1]: ma_score += 1
                if len(close) >= 60 and latest_close > ma60.iloc[-1]: ma_score += 1

                ma_trend = ""
                if ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1]:
                    ma_trend = "多头排列"
                elif ma5.iloc[-1] < ma10.iloc[-1] < ma20.iloc[-1]:
                    ma_trend = "空头排列"
                else:
                    ma_trend = "震荡整理"

                rsi = calculate_rsi(close, 14)

                etf_data.append({
                    'code': code,
                    'name': name,
                    'close': latest_close,
                    'pct_5d': pct_5d,
                    'pct_20d': pct_20d,
                    'pct_60d': pct_60d,
                    'vol_ratio': vol_ratio,
                    'ma_score': ma_score,
                    'ma_trend': ma_trend,
                    'rsi': rsi,
                })
            except Exception as e:
                log.debug(f"ETF {code} ({name}) 数据获取失败: {e}")
                continue

        if not etf_data:
            return "### 📊 ETF轮动分析\n⚠️ ETF数据获取失败，暂无轮动信号。\n"

        df_etf = pd.DataFrame(etf_data)

        df_etf['momentum_score'] = (
            df_etf['pct_5d'] * 0.4 +
            df_etf['pct_20d'] * 0.4 +
            df_etf['pct_60d'] * 0.2
        )
        df_etf['total_score'] = (
            df_etf['momentum_score'] * 0.5 +
            df_etf['ma_score'] * 10 * 0.3 +
            (df_etf['vol_ratio'] - 1) * 5 * 0.2
        )

        df_sorted = df_etf.sort_values('total_score', ascending=False)

        top_etfs = df_sorted.head(5)
        bottom_etfs = df_sorted.tail(3)

        strong_etfs = df_sorted[(df_sorted['ma_trend'] == '多头排列') & (df_sorted['pct_5d'] > 0)].head(5)

        msg = "### 📊 ETF轮动雷达\n\n"

        msg += "**🏆 强势ETF Top 5（动量+趋势综合评分）**\n"
        for _, row in top_etfs.iterrows():
            emoji = "🚀" if row['ma_trend'] == '多头排列' else "📈" if row['pct_5d'] > 0 else "⚖️"
            msg += (
                f"- {emoji} **{row['name']}** (`{row['code']}`) | "
                f"现价 `¥{row['close']:.3f}` | "
                f"5日 `{row['pct_5d']:+.2f}%` / 20日 `{row['pct_20d']:+.2f}%` / 60日 `{row['pct_60d']:+.2f}%` | "
                f"量比 `{row['vol_ratio']:.2f}` | "
                f"趋势: {row['ma_trend']} | "
                f"RSI: `{row['rsi']:.1f}`\n"
            )

        msg += "\n**💀 弱势ETF Bottom 3（规避风险区）**\n"
        for _, row in bottom_etfs.iterrows():
            msg += (
                f"- 📉 **{row['name']}** (`{row['code']}`) | "
                f"5日 `{row['pct_5d']:+.2f}%` / 20日 `{row['pct_20d']:+.2f}%` | "
                f"趋势: {row['ma_trend']}\n"
            )

        if len(strong_etfs) >= 2:
            msg += "\n**🎯 轮动建议**\n"
            msg += f"当前市场强势板块集中在 **{', '.join(strong_etfs['name'].head(3).tolist())}** 等方向，"
            msg += "建议关注多头排列且量能温和放大的品种，回避空头排列且持续缩量的弱势ETF。\n"

        msg += "\n> *ETF轮动基于动量+均线趋势+量能综合评分，仅供参考，不构成投资建议。*"

        log.info(f"✅ ETF轮动分析完成，共分析 {len(etf_data)} 只ETF")
        return msg

    except Exception as e:
        log.error(f"❌ ETF轮动分析失败: {e}")
        return f"### 📊 ETF轮动分析\n⚠️ ETF轮动分析异常: {e}\n"

def calculate_rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    val = float(rsi.iloc[-1])
    return val if not np.isnan(val) else 50.0

def get_industry_hotspot_analysis() -> str:
    try:
        log.info("🌋 开始分析行业热点...")
        hot_map = fetch_hot_sectors()
        if not hot_map:
            return "### 🌋 行业热点\n⚠️ 行业板块数据获取失败。\n"

        sec_counts = Counter(hot_map.values())
        top_sectors = sec_counts.most_common(10)

        msg = "### 🌋 行业热点雷达\n\n"
        msg += "**🔥 今日领涨行业 Top 10**\n"
        for i, (sector, count) in enumerate(top_sectors, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            msg += f"- {medal} **{sector}**：{count}只成分股活跃\n"

        msg += "\n**💡 热点解读**\n"
        if top_sectors:
            top3 = [s[0] for s in top_sectors[:3]]
            msg += f"今日市场主线聚焦于 **{', '.join(top3)}** 方向，"
            if len(top_sectors) > 5:
                msg += "板块轮动较快，建议关注持续性强的主线品种。\n"
            else:
                msg += "市场热点相对集中，可重点跟踪龙头标的表现。\n"

        msg += "\n> *行业热点数据基于东方财富/同花顺板块涨幅榜，成分股数量代表板块活跃度。*"
        return msg

    except Exception as e:
        log.error(f"❌ 行业热点分析失败: {e}")
        return f"### 🌋 行业热点\n⚠️ 行业热点分析异常: {e}\n"

def get_market_analysis() -> tuple[str, bool]:
    try:
        log.info("📈 开始大盘分析...")
        df_raw = fetch_spot()
        if len(df_raw) < 1000:
            return "### 📊 市场全景分析\n⚠️ 行情数据不足，无法进行完整市场分析。\n", False

        idx_df = fetch_index('sh000001')
        cl = idx_df['close'].astype(float)

        pct = (cl.iloc[-1] - cl.iloc[-2]) / cl.iloc[-2] * 100 if len(cl) >= 2 else 0.0
        trend_name, trend_desc = get_ma_trend(cl)

        up_count = (df_raw[C.S_PCT].astype(float) > 0).sum()
        down_count = (df_raw[C.S_PCT].astype(float) < 0).sum()
        zt_count = (df_raw[C.S_PCT].astype(float) >= 9.0).sum()
        dt_count = (df_raw[C.S_PCT].astype(float) <= -9.0).sum()
        total_amt = df_raw[C.S_AMT].astype(float).sum() / 1e8

        north_flow, north_msg = fetch_northbound_flow()

        idx_300 = fetch_index('sh000300')
        idx_500 = fetch_index('sh000905')
        idx_gem = fetch_index('sz399006')

        def get_idx_pct(idx_df):
            if idx_df is None or len(idx_df) < 2:
                return 0.0
            c = idx_df['close'].astype(float)
            return (c.iloc[-1] - c.iloc[-2]) / c.iloc[-2] * 100

        hs300_pct = get_idx_pct(idx_300)
        zz500_pct = get_idx_pct(idx_500)
        cyb_pct = get_idx_pct(idx_gem)

        def pct_emoji(p):
            if p > 1: return "🔴"
            elif p > 0: return "🟠"
            elif p > -1: return "🟢"
            else: return "🟢"

        msg = "### 📊 市场全景分析\n\n"

        msg += "**📈 主要指数表现**\n"
        msg += f"- 上证指数：`{cl.iloc[-1]:.2f}` ({pct:+.2f}%)\n"
        msg += f"- 沪深300：`{idx_300['close'].astype(float).iloc[-1]:.2f}` ({hs300_pct:+.2f}%)\n" if idx_300 is not None else ""
        msg += f"- 中证500：`{idx_500['close'].astype(float).iloc[-1]:.2f}` ({zz500_pct:+.2f}%)\n" if idx_500 is not None else ""
        msg += f"- 创业板指：`{idx_gem['close'].astype(float).iloc[-1]:.2f}` ({cyb_pct:+.2f}%)\n" if idx_gem is not None else ""

        msg += f"\n**🧭 大盘趋势诊断**\n"
        msg += f"- **均线系统**：{trend_name} - {trend_desc}\n"
        msg += f"- **市场广度**：上涨 {up_count} 家 / 下跌 {down_count} 家\n"
        msg += f"- **涨跌停**：涨停 {zt_count} 家 / 跌停 {dt_count} 家\n"
        msg += f"- **两市成交**：约 {total_amt:.0f} 亿元\n"
        msg += f"{north_msg}\n"

        market_temp = "偏暖" if up_count > down_count * 1.5 else "偏冷" if down_count > up_count * 1.5 else "均衡"
        msg += f"\n**🌡️ 市场情绪**：{market_temp}\n"

        log.info("✅ 大盘分析完成")
        return msg, True

    except Exception as e:
        log.error(f"❌ 市场分析失败: {e}")
        return f"### 📊 市场全景分析\n⚠️ 市场分析异常: {e}\n", False

def get_stock_signals_brief() -> str:
    try:
        log.info("🎯 开始股票信号筛选...")
        df_raw = fetch_spot()
        if len(df_raw) < 100:
            return "### 🎯 股票信号扫描\n⚠️ 行情数据不足。\n"

        df_raw[C.S_CODE] = df_raw[C.S_CODE].astype(str).str.zfill(6)
        df_raw[C.S_PCT] = pd.to_numeric(df_raw[C.S_PCT], errors='coerce')
        df_raw[C.S_PRICE] = pd.to_numeric(df_raw[C.S_PRICE], errors='coerce')
        df_raw[C.S_AMT] = pd.to_numeric(df_raw[C.S_AMT], errors='coerce')
        df_raw[C.S_VR] = pd.to_numeric(df_raw[C.S_VR], errors='coerce')
        df_raw[C.S_TURN] = pd.to_numeric(df_raw[C.S_TURN], errors='coerce')
        df_raw[C.S_MCAP] = pd.to_numeric(df_raw[C.S_MCAP], errors='coerce')

        mask = (
            (df_raw[C.S_PCT] > 2.0) &
            (df_raw[C.S_PCT] < 9.0) &
            (df_raw[C.S_AMT] > 5e8) &
            (df_raw[C.S_MCAP] > 50e8) &
            (df_raw[C.S_MCAP] < 2000e8) &
            (~df_raw[C.S_NAME].str.contains('ST|退')) &
            (~df_raw[C.S_CODE].str.startswith(('688', '8', '4', '9')))
        )

        candidates = df_raw[mask].copy()
        if candidates.empty:
            return "### 🎯 股票信号扫描\n✅ 今日未发现符合条件的异动标的，建议空仓观望。\n"

        candidates['vol_turn_score'] = (
            candidates[C.S_VR].fillna(1) * 0.5 +
            candidates[C.S_TURN].fillna(2) * 0.3 +
            candidates[C.S_PCT] * 0.2
        )
        top_candidates = candidates.nlargest(8, 'vol_turn_score')

        hot_map = fetch_hot_sectors()

        msg = "### 🎯 今日强势异动股（简版）\n\n"
        msg += "| 代码 | 名称 | 现价 | 涨幅 | 成交额 | 量比 | 所属板块 |\n"
        msg += "|------|------|------|------|--------|------|----------|\n"

        for _, row in top_candidates.iterrows():
            code = str(row[C.S_CODE]).zfill(6)
            sector = hot_map.get(code, '-')
            amt_yi = row[C.S_AMT] / 1e8
            msg += (
                f"| `{code}` | {row[C.S_NAME]} | ¥{row[C.S_PRICE]:.2f} | "
                f"`{row[C.S_PCT]:+.2f}%` | {amt_yi:.1f}亿 | "
                f"{row[C.S_VR]:.2f} | {sector} |\n"
            )

        msg += "\n> *以上为今日量价异动筛选结果，仅作观察参考，需结合技术形态和基本面进一步分析。完整信号请参阅盘后量化选股报告。*"

        log.info(f"✅ 股票信号筛选完成，共 {len(top_candidates)} 只候选")
        return msg

    except Exception as e:
        log.error(f"❌ 股票信号筛选失败: {e}")
        return f"### 🎯 股票信号扫描\n⚠️ 信号筛选异常: {e}\n"

def generate_daily_briefing() -> str:
    now = datetime.now(TZ_BJS)
    now_str = now.strftime('%Y-%m-%d %H:%M')
    weekday_cn = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'][now.weekday()]

    title = f"# 📮 每日投研简报\n> **{now.strftime('%Y年%m月%d日')} {weekday_cn} {now.strftime('%H:%M')}**\n\n"

    macro_section = generate_macro_section()
    market_section, market_ok = get_market_analysis()
    industry_section = get_industry_hotspot_analysis()
    etf_section = get_etf_rotation_analysis()
    stock_section = get_stock_signals_brief()

    content = (
        f"{title}"
        f"{macro_section}\n\n"
        f"---\n\n"
        f"{market_section}\n\n"
        f"---\n\n"
        f"{industry_section}\n\n"
        f"---\n\n"
        f"{etf_section}\n\n"
        f"---\n\n"
        f"{stock_section}\n\n"
        f"---\n\n"
        f"> 📌 **免责声明**：本简报由AI量化系统自动生成，仅供投研参考，不构成任何投资建议。股市有风险，入市需谨慎。\n"
        f"> 🔧 **数据来源**：东方财富、同花顺、Yahoo Finance、Tushare等公开数据源。\n"
    )

    return content

def send_briefing_to_dingtalk(content: str) -> None:
    now_str = datetime.now(TZ_BJS).strftime('%Y-%m-%d')
    NotificationGateway.send(
        f'📮 每日投研简报 - {now_str}',
        content,
        template='blue'
    )

def main():
    log.info("=" * 60)
    log.info("📮 每日投研简报系统启动")
    log.info("=" * 60)

    try:
        config.print_summary(log)

        content = generate_daily_briefing()

        if not config.DINGTALK_WEBHOOK and not config.FEISHU_WEBHOOK:
            log.warning("⚠️ 未配置 WEBHOOK，简报内容将输出到本地...")
            output_file = f"daily_briefing_{_today_str()}.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            log.info(f"✅ 简报已保存到 {output_file}")
            print("\n" + "=" * 60)
            print(content)
            print("=" * 60)
        else:
            send_briefing_to_dingtalk(content)
            log.info("✅ 每日投研简报已推送")

    except Exception as e:
        log.critical(f"❌ 每日投研简报生成失败: {e}", exc_info=True)
        error_msg = (
            f"🚨 **每日投研简报生成失败**\n\n"
            f"**时间**: {_today_str()}\n"
            f"**异常信息**: {str(e)[:300]}..."
        )
        try:
            NotificationGateway.send("🚨 简报生成失败", error_msg, template="red")
        except:
            pass
    finally:
        from main import _DATA_PROXY
        _DATA_PROXY.cleanup()
        log.info("📮 每日投研简报系统结束")

if __name__ == '__main__':
    main()
