#!/usr/bin/env python3
"""
每日投研简报生成器 - Daily Investment Research Briefing Generator

功能：
  1. 宏观快报（VIX/标普500/美债/黄金/原油等）
  2. A股深度诊断（趋势/广度/量能/北向资金）
  3. ETF轮动信号（筛选51xxxx/15xxxx/588xxx/56xxxx）
  4. 行业热点统计（基于hot_sectors缓存）
  5. 个股信号扫描（遍历hist_cache中的~93只历史K线数据）
  6. 钉钉推送 / stdout输出

使用方式：
  python daily_briefing.py

依赖：main.py 已完整配置好所有数据源和通知网关
"""

import os
# 强制设为手动模式，绕过时间检查（与 main.py 中 IS_MANUAL 逻辑一致）
os.environ['GITHUB_EVENT_NAME'] = 'workflow_dispatch'

# --- SANDBOX_MODE 代理保留逻辑（与 main.py 第1-7行一致） ---
if os.environ.get('SANDBOX_MODE', '') != '1':
    for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
        os.environ.pop(k, None)
    os.environ['NO_PROXY'] = '*'

import logging
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

# 导入 main.py 的核心组件
from main import (
    config,
    log,
    TZ_BJS,
    C,           # Cols 常量类实例
    Config,      # 配置 dataclass
    Signal,      # 信号 dataclass
    NotificationGateway,
    generate_macro_section,
    extract_market_context,
    fetch_spot,
    fetch_index,
    fetch_hot_sectors,
    fetch_northbound_flow,
    AShareTechnicals,
    process_stock,
    calc_target_price,
    format_money_risk_msg,
    generate_tranche_plan,
    generate_plan_b,
)

HIST_CACHE_DIR = Path('/workspace/hist_cache')

# ──────────────────────────────────────────────
# 简报生成核心函数
# ──────────────────────────────────────────────


def load_cached_hist_files():
    """从 hist_cache 目录加载所有 hist_*.parquet 个股历史K线文件"""
    hist_files = sorted(HIST_CACHE_DIR.glob('hist_*.parquet'))
    if not hist_files:
        log.warning("⚠️ hist_cache 目录下未找到任何 hist_*.parquet 文件")
        return {}

    loaded = {}
    for fp in hist_files:
        try:
            df = pd.read_parquet(fp)
            # 从文件名提取股票代码: hist_000001_20260709.parquet -> 000001
            code = fp.stem.split('_')[1] if '_' in fp.stem else fp.stem.replace('hist_', '')
            loaded[code] = df
        except Exception as e:
            log.warning(f"加载历史数据失败 {fp.name}: {e}")

    log.info(f"📦 成功加载 {len(loaded)} 只个股的历史K线数据")
    return loaded


def build_etf_section(spot_df: pd.DataFrame) -> str:
    """从现货行情中筛选 ETF（51/15/588/56开头），按涨跌幅排序展示Top 10"""
    if spot_df is None or spot_df.empty:
        return "### 📈 ETF轮动信号\n⚠️ 无法获取现货行情数据\n"

    code_col = C.S_CODE
    name_col = C.S_NAME
    pct_col = C.S_PCT
    price_col = C.S_PRICE

    etf_prefixes = ('51', '15', '588', '56')
    mask = spot_df[code_col].astype(str).str.startswith(etf_prefixes)
    etf_df = spot_df[mask].copy()
    if etf_df.empty:
        return "### 📈 ETF轮动信号\n⚠️ 今日无ETF交易数据\n"

    etf_df[pct_col] = pd.to_numeric(etf_df[pct_col], errors='coerce')
    etf_df = etf_df.sort_values(pct_col, ascending=False).head(10)

    lines = ["### 📈 ETF轮动信号", ""]
    for _, row in etf_df.iterrows():
        code = str(row[code_col])
        name = row[name_col]
        pct = row[pct_col]
        price = row[price_col]
        emoji = "🔥" if pct > 2 else "📊" if pct > 0 else "❄️"
        pct_str = f"{pct:+.2f}%" if pd.notna(pct) else "N/A"
        lines.append(f"- {emoji} **{name}** (`{code}`): ¥{price} ({pct_str})")

    return "\n".join(lines)


def build_sector_hotspot_section(hot_sectors_map: dict) -> str:
    """基于 hot_sectors 缓存构建行业热点板块"""
    if not hot_sectors_map:
        return "### 🔥 行业热点\n⚠️ 暂无行业热点数据\n"

    from collections import Counter
    sec_counts = Counter(hot_sectors_map.values())
    top_sectors = sec_counts.most_common(8)

    lines = ["### 🔥 行业热点", ""]
    for sector_name, count in top_sectors:
        emoji = "🔴" if count >= 6 else "🟡" if count >= 3 else "🟢"
        lines.append(f"- {emoji} **{sector_name}**: {count} 只个股异动")

    total_stocks = len(hot_sectors_map)
    lines.append(f"\n> 共涉及 `{total_stocks}` 只热门标的")

    return "\n".join(lines)


def build_stock_signal_section(hist_data: dict, spot_df: pd.DataFrame, now: datetime,
                               market_ok: bool, index_ret: float, hot_sectors_map: dict) -> str:
    """遍历缓存的历史K线数据，通过 process_stock 管道生成个股信号"""
    if not hist_data or spot_df is None or spot_df.empty:
        return "### 🎯 个股信号\n⚠️ 无可用个股数据或历史K线数据不足\n"

    signals_found = []
    processed_count = 0
    error_count = 0

    for code, raw_hist in hist_data.items():
        try:
            # 从现货行情中查找该股票的当日截面数据
            code_key = str(code).zfill(6)
            match_rows = spot_df[spot_df[C.S_CODE].astype(str).str.contains(code_key)]
            if match_rows.empty:
                continue

            row = match_rows.iloc[0]
            processed_count += 1

            # 调用 main.py 的 process_stock 进行技术面+基本面过滤
            result = process_stock(row, raw_hist, now, market_ok, index_ret, hot_sectors_map)
            if result is None:
                continue

            data, stop, risk_pct = result

            score, level, reasons = apply_scoring_lite(data, now)

            target1 = calc_target_price(row[C.S_PRICE], stop, data)
            money_msg = format_money_risk_msg(row[C.S_PRICE], stop, target1)
            tranche_msg = generate_tranche_plan(row[C.S_PRICE], score, market_ok, False)
            plan_b_msg = generate_plan_b(row[C.S_PRICE], stop, data['ma20_val'])

            sig = Signal(
                code=row[C.S_CODE],
                name=row[C.S_NAME],
                price=row[C.S_PRICE],
                pct_chg=f"{row[C.S_PCT]}%",
                score=score,
                level=level,
                trigger_time=now.strftime('%H:%M'),
                reasons=reasons,
                stop_loss=round(stop, 2),
                target1=target1,
                ma10=round(data['ma10_val'], 2),
                money_risk_msg=money_msg,
                tranche_plan_msg=tranche_msg,
                plan_b_msg=plan_b_msg,
            )
            signals_found.append(sig)

        except Exception as e:
            error_count += 1
            log.debug(f"处理 {code} 异常: {e}")
            continue

    # 格式化信号输出
    if not signals_found:
        return (
            f"### 🎯 个股信号\n\n"
            f"> 扫描了 `{processed_count}` 只股票的缓存历史数据，"
            f"未发现符合安全边际的信号。建议空仓防守。\n"
        )

    # 按分数降序排列
    signals_found.sort(key=lambda s: s.score, reverse=True)

    lines = [f"### 🎯 个股信号", ""]
    lines.append(f"> 共扫描 `{processed_count}` 只缓存个股，发现 `{len(signals_found)}` 只有效信号\n")

    for sig in signals_found[:10]:  # 展示Top 10
        warn = "> ⚡ 该股为创业板(波动±20%)，请务必缩减仓位。\n\n" if str(sig.code).startswith('300') else ""
        prefix = '1' if str(sig.code).startswith('6') else '0'
        sina_market = 'sh' if str(sig.code).startswith('6') else 'sz'
        kline_url = f"http://image.sinajs.cn/newchart/weekly/n/{sina_market}{sig.code}.gif"

        lines.append(
            f"#### 🎯 {sig.name} (`{sig.code}`)\n"
            f"{warn}"
            f"- **综合评级**: `{sig.score}` 分 {sig.level}\n"
            f"- **今日收盘**: ¥{sig.price} ({sig.pct_chg}) [📈 周K图]({kline_url})\n\n"
            f"**💡 核心逻辑**\n{sig.reasons}\n\n"
            f"**🛡️ 交易计划**\n"
            f"{sig.money_risk_msg}\n"
            f"{sig.tranche_plan_msg}\n"
            f"{sig.plan_b_msg}\n"
            f"> ⚠️ 纪律: 破防守线 ¥{sig.stop_loss} 止损; 高开>4%放弃\n\n"
            f"[🔗 东财App看盘](https://quote.eastmoney.com/unify/r/{prefix}.{sig.code})"
        )

    if len(signals_found) > 10:
        lines.append(f"\n> *另有 {len(signals_found) - 10} 只信号因篇幅限制未展示*")

    return "\n".join(lines)


def apply_scoring_lite(data: dict, now: datetime):
    """轻量级评分逻辑（不依赖 ML 模型，纯规则打分）"""
    adx = data.get('adx', 20)
    rsi = data.get('rsi', 50)
    dist_ma20 = data.get('dist_ma20', 0)
    has_obv_break = data.get('has_obv_break', False)
    macd_divergence = data.get('macd_divergence', False)
    is_true_vcp = data.get('is_true_vcp', False)
    has_chip_break = data.get('has_chip_break', False)
    is_first_dip = data.get('is_first_dip', False)
    red_days = data.get('red_days', 0)
    surge_5d = data.get('surge_5d', 0)
    close_val = data.get('close_val', 0)
    ma20_val = data.get('ma20_val', 0)

    base_score = 50

    # 动量加分
    if close_val > ma20_val:
        base_score += 8
    if has_obv_break:
        base_score += 6
    if macd_divergence:
        base_score += 8
    if is_first_dip:
        base_score += 12

    # 形态加分
    if is_true_vcp:
        base_score += 6
    if has_chip_break:
        base_score += 5
    if red_days >= 2:
        base_score += 3

    # RSI 区间调整
    if 40 <= rsi <= 65:
        base_score += 5
    elif rsi > 80:
        base_score -= 10

    # MA20 距离
    if 0 < dist_ma20 < 3:
        base_score += 3
    elif dist_ma20 < -5:
        base_score -= 8

    # ADX 趋势强度
    if adx > 25:
        base_score += 4
    elif adx < 15:
        base_score -= 3

    # 5日涨幅控制（防止追高）
    if surge_5d > 20:
        base_score -= 8
    elif 3 < surge_5d < 12:
        base_score += 3

    base_score = max(0, min(base_score, 100))

    # 评级判定
    if base_score >= 85:
        level = '⭐⭐⭐⭐⭐ 🐯 **[S级·老虎机]** (胜率极高，跌势有限)'
    elif base_score >= 75:
        level = '⭐⭐⭐⭐ 🐕 **[A级·看门狗]** (防守兼备，需耐心等涨)'
    elif base_score >= 70:
        level = '⭐⭐⭐ 🦊 **[B+级·小狐狸]** (次优机会，必须控制仓位)'
    else:
        level = '⭐⭐ 🐒 **[B级·小猕猴]** (上蹿下跳振幅大，新手回避)'

    # 构造理由文本
    reason_parts = [
        f"- 🧭 **趋势雷达**: {'处于强势主升浪' if adx > 25 else '正处于底部反转期' if adx < 15 else '平稳震荡蓄势'}",
    ]
    if has_obv_break:
        reason_parts.append("- 💧 **OBV突破**: 能量潮突破21日新高，资金持续进场")
    if macd_divergence:
        reason_parts.append("- 🔄 **MACD底背离**: 价格新低但MACD未创新低，反转概率提升")
    if is_first_dip:
        reason_parts.append("- 🎯 **龙头首阴**: 连板后首次回调至均线支撑，经典买点")
    if is_true_vcp:
        reason_parts.append("- 📐 **VCP结构**: 波动率逐级收敛，主力洗盘完毕")
    if has_chip_break:
        reason_parts.append("- 🔨 **筹码突破**: 放量突破长期筹码密集区")

    reasons = "\n".join(reason_parts)

    return int(base_score), level, reasons


def build_risk_control_section(market_msg: str) -> str:
    """从市场消息中提炼风控提示"""
    if not market_msg:
        return ""

    lines = ["### 📌 风控提示", ""]

    # 提取关键风控信息
    if "熔断" in market_msg:
        lines.append("- 🚨 大盘已触发熔断警报，建议空仓观望")
    if "走熊" in market_msg:
        lines.append("- ⚠️ 大盘处于熊市结构，严控仓位在30%以下")
    if "外资砸盘" in market_msg:
        lines.append("- ⚠️ 北向资金大幅流出，注意规避外资重仓股")
    if "情绪熔断" in market_msg or "极度狂欢" in market_msg:
        lines.append("- 🚨 市场过热，涨停破百，谨防获利盘踩踏")
    if "恐慌冰点" in market_msg:
        lines.append("- 🧊 市场进入冰点区域，可左侧轻仓试错但需严格止损")

    # 默认通用提示
    if len(lines) <= 2:
        lines.extend([
            "- 单只个股仓位不超过总资金的10%",
            "- 严格执行止损纪律，亏损达8%无条件出局",
            "- 避免追涨杀跌，耐心等待确定性机会",
        ])

    return "\n".join(lines)


def generate_daily_briefing() -> str:
    """组装完整的每日投研简报"""
    now = datetime.now(TZ_BJS)
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M')

    log.info(f"🚀 每日投研简报引擎启动... [{date_str} {time_str}]")

    parts = []

    # ── 1. 报头 ──
    header = (
        f"## 🤖 AI量化每日投研简报\n"
        f"> **{date_str}**\n"
    )
    parts.append(header)

    # ── 2 & 3. 宏观快报 + A股市场诊断 ──
    # extract_market_context 内部已包含 generate_macro_section()，
    # 因此直接复用其返回的 market_msg 即可避免宏观部分重复
    market_msg = ""
    market_ok = True
    index_ret = 0.0
    market_overheated = False
    market_regime = "NEUTRAL"
    vol_surge = False
    spot_df = None

    log.info("📊 正在进行A股深度诊断（含宏观指标）...")
    try:
        spot_df = fetch_spot()
        c_conf = Config()

        # 复用 main.py 的市场分析管道（内含宏观快报 + 行业热点 + 北向资金）
        _, m_ok, m_msg, idx_ret, m_overheat, m_regime, v_surge, _temp = \
            extract_market_context(spot_df, c_conf)

        market_ok = m_ok
        index_ret = idx_ret
        market_overheated = m_overheat
        market_regime = m_regime
        vol_surge = v_surge
        market_msg = m_msg
        parts.append(market_msg)
        parts.append("")

    except Exception as e:
        log.error(f"A股诊断失败: {e}\n{traceback.format_exc()}")
        # 如果市场诊断失败，单独尝试获取宏观快报
        log.info("📡 市场诊断失败，单独获取宏观指标...")
        try:
            macro_section = generate_macro_section()
            parts.append(macro_section)
            parts.append("")
        except Exception as me:
            log.warning(f"宏观数据也获取失败: {me}")
            parts.append("### 🌍 隔夜外围与宏观风控快报\n⚠️ 外围数据暂时不可用\n")
        parts.append("### 📊 A股深度诊断\n⚠️ 市场数据获取异常，请检查网络或数据源配置\n")

    # ── 4. 行业热点 ──
    log.info("🔥 正在统计行业热点...")
    hot_sectors_map = {}
    try:
        hot_sectors_map = fetch_hot_sectors()
        sector_section = build_sector_hotspot_section(hot_sectors_map)
        parts.append(sector_section)
        parts.append("")
    except Exception as e:
        log.warning(f"行业热点获取失败: {e}")
        parts.append("### 🔥 行业热点\n⚠️ 行业热点暂不可用\n\n")

    # ── 5. ETF轮动 ──
    log.info("📈 正在计算ETF轮动信号...")
    try:
        if spot_df is None:
            spot_df = fetch_spot()
        etf_section = build_etf_section(spot_df)
        parts.append(etf_section)
        parts.append("")
    except Exception as e:
        log.warning(f"ETF数据获取失败: {e}")
        parts.append("### 📈 ETF轮动信号\n⚠️ ETF数据暂不可用\n\n")

    # ── 6. 个股信号 ──
    log.info("🎯 正在扫描个股信号...")
    try:
        hist_data = load_cached_hist_files()
        if spot_df is None:
            spot_df = fetch_spot()
        if not hot_sectors_map:
            hot_sectors_map = fetch_hot_sectors()
        signal_section = build_stock_signal_section(
            hist_data, spot_df, now, market_ok, index_ret, hot_sectors_map
        )
        parts.append(signal_section)
        parts.append("")
    except Exception as e:
        log.error(f"个股信号扫描失败: {e}\n{traceback.format_exc()}")
        parts.append("### 🎯 个股信号\n⚠️ 个股信号扫描异常\n\n")

    # ── 7. 风控提示 ──
    risk_section = build_risk_control_section(market_msg)
    if risk_section:
        parts.append(risk_section)
        parts.append("")

    # ── 8. 页脚 ──
    footer = (
        "---\n\n"
        f"> *本报告由 AI 量化选股系统自动生成 | "
        f"生成时间: {time_str} | "
        f"数据来源: 东方财富/Tushare/Yahoo Finance*\n"
    )
    parts.append(footer)

    full_content = "\n".join(parts)
    log.info(f"✅ 简报生成完成，总长度: {len(full_content)} 字符")
    return full_content


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def main():
    """主入口函数：生成简报并推送/打印"""
    try:
        briefing_content = generate_daily_briefing()

        # 尝试通过 NotificationGateway 推送到钉钉/飞书
        if config.DINGTALK_WEBHOOK or config.FEISHU_WEBHOOK:
            log.info("📤 检测到 Webhook 配置，正在推送简报...")
            try:
                NotificationGateway.send(
                    title="🤖 AI量化每日投研简报",
                    content=briefing_content,
                    template="blue",
                )
                log.info("✅ 简报推送成功！")
            except Exception as push_err:
                log.error(f"❌ 推送失败: {push_err}")
                print("\n" + "=" * 60)
                print("推送失败，以下为简报内容：")
                print("=" * 60 + "\n")
                print(briefing_content)
        else:
            # 未配置 Webhook 时直接打印到标准输出
            print("\n" + "=" * 60)
            print("每日投研简报 (未检测到 Webhook，仅打印到控制台)")
            print("=" * 60 + "\n")
            print(briefing_content)

    except Exception as e:
        log.error(f"❌ 简报生成过程发生致命错误: {e}\n{traceback.format_exc()}")
        raise


if __name__ == '__main__':
    main()
