#!/usr/bin/env python3
"""
每日投研简报生成器
生成包含市场分析、股票信号、ETF轮动、行业热点等内容，并发送到钉钉通知
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    AppConfig, DataProxy, LocalDataLake, AShareTechnicals,
    NotificationGateway, Config, Cols, Signal,
    fetch_spot, fetch_index, fetch_hot_sectors, fetch_northbound_flow,
    extract_market_context, generate_macro_section,
    format_money_risk_msg, generate_tranche_plan, generate_plan_b,
    calc_target_price, MathUtils
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

TZ_BJS = pytz.timezone('Asia/Shanghai')
C = Cols()


class DailyBriefingGenerator:
    def __init__(self):
        self.config = AppConfig()
        self.data_proxy = DataProxy()
        self.data_lake = LocalDataLake(self.data_proxy)

    def generate_etf_rotation_analysis(self, df_raw: pd.DataFrame) -> str:
        """生成ETF轮动分析报告"""
        try:
            etf_df = df_raw[df_raw[C.S_CODE].astype(str).str.startswith(('51', '15', '588', '56'))]

            if etf_df.empty:
                return "### 📊 ETF轮动分析\n⚠️ 今日ETF数据获取异常\n"

            etf_df[C.S_PCT] = pd.to_numeric(etf_df[C.S_PCT], errors='coerce').fillna(0)

            top_gainers = etf_df.nlargest(5, C.S_PCT)[[C.S_NAME, C.S_CODE, C.S_PCT, C.S_AMT]]
            top_losers = etf_df.nsmallest(5, C.S_PCT)[[C.S_NAME, C.S_CODE, C.S_PCT, C.S_AMT]]

            gainer_lines = []
            for _, row in top_gainers.iterrows():
                code = str(row[C.S_CODE])
                amt = row.get(C.S_AMT, 0) / 1e8 if pd.notna(row.get(C.S_AMT)) else 0
                gainer_lines.append(
                    f"- `{code}` **{row[C.S_NAME]}** 涨跌幅 `{row[C.S_PCT]:+.2f}%` 成交额 `{amt:.1f}亿`"
                )

            loser_lines = []
            for _, row in top_losers.iterrows():
                code = str(row[C.S_CODE])
                amt = row.get(C.S_AMT, 0) / 1e8 if pd.notna(row.get(C.S_AMT)) else 0
                loser_lines.append(
                    f"- `{code}` **{row[C.S_NAME]}** 涨跌幅 `{row[C.S_PCT]:+.2f}%` 成交额 `{amt:.1f}亿`"
                )

            broad_etf = etf_df[etf_df[C.S_NAME].str.contains('沪深300|中证500|中证1000|创业板|科创板', na=False)]
            if not broad_etf.empty:
                broad_pct = broad_etf[C.S_PCT].mean()
                broad_msg = f"\n> **宽基ETF平均表现**: `{broad_pct:+.2f}%`"
            else:
                broad_msg = ""

            msg = (
                f"### 📊 ETF轮动分析\n"
                f"{broad_msg}\n\n"
                f"**📈 涨幅前五ETF**:\n" + "\n".join(gainer_lines) + "\n\n"
                f"**📉 跌幅前五ETF**:\n" + "\n".join(loser_lines) + "\n\n"
                f"> *注: ETF资金流向可反映机构对后市的预期判断*"
            )
            return msg
        except Exception as e:
            log.warning(f"ETF轮动分析生成失败: {e}")
            return f"### 📊 ETF轮动分析\n⚠️ 分析生成失败: {e}\n"

    def generate_sector_rotation_analysis(self, hot_sectors_map: dict) -> str:
        """生成行业轮动热力图分析"""
        try:
            if not hot_sectors_map:
                return "### 🔥 行业热点轮动\n⚠️ 今日行业热点数据获取异常\n"

            from collections import Counter
            sector_counts = Counter(hot_sectors_map.values())

            top_sectors = sector_counts.most_common(8)

            lines = []
            for sector, count in top_sectors:
                intensity = "🔥" * min(3, count // 5 + 1)
                lines.append(f"- {intensity} **{sector}** 成分股 {count} 只")

            msg = (
                f"### 🔥 行业热点轮动\n\n"
                + "\n".join(lines) + "\n\n"
                f"> *注: 成分股数量反映该板块的市场关注度与资金聚集程度*"
            )
            return msg
        except Exception as e:
            log.warning(f"行业轮动分析生成失败: {e}")
            return f"### 🔥 行业热点轮动\n⚠️ 分析生成失败: {e}\n"

    def generate_index_comparison(self) -> str:
        """生成主要指数对比分析"""
        try:
            indices = [
                ('sh000001', '上证指数'),
                ('sh000300', '沪深300'),
                ('sh000905', '中证500'),
                ('sh000852', '中证1000'),
                ('sz399006', '创业板指'),
            ]

            lines = []
            for symbol, name in indices:
                try:
                    df = fetch_index(symbol)
                    if df is not None and not df.empty and len(df) >= 2:
                        close = df['close'].iloc[-1]
                        prev_close = df['close'].iloc[-2]
                        pct = (close - prev_close) / prev_close * 100
                        lines.append(f"- **{name}**: `{close:.2f}` ({pct:+.2f}%)")
                except Exception as e:
                    log.debug(f"获取指数 {name} 失败: {e}")

            if not lines:
                return "### 📈 主要指数对比\n⚠️ 指数数据获取失败\n"

            msg = (
                f"### 📈 主要指数对比\n" + "\n".join(lines) + "\n"
            )
            return msg
        except Exception as e:
            log.warning(f"指数对比分析生成失败: {e}")
            return f"### 📈 主要指数对比\n⚠️ 分析生成失败: {e}\n"

    def generate_smart_money_analysis(self) -> str:
        """生成聪明钱流向分析"""
        try:
            north_flow, north_msg = fetch_northbound_flow()

            msg = f"### 💰 聪明钱流向\n{north_msg}\n"

            return msg
        except Exception as e:
            log.warning(f"聪明钱分析生成失败: {e}")
            return f"### 💰 聪明钱流向\n⚠️ 分析生成失败: {e}\n"

    def generate_risk_warning(self, market_overheated: bool, market_regime: str) -> str:
        """生成风险预警模块"""
        warnings = []

        if market_overheated:
            warnings.append("🚨 **市场极度过热警告**: 涨停家数破百，情绪极度亢奋，系统建议谨慎追高！")

        if market_regime == 'PANIC':
            warnings.append("🧊 **市场恐慌模式**: 系统检测到恐慌信号，建议降低仓位，防御为主！")

        if market_regime == 'BEAR':
            warnings.append("🐻 **空头主导**: 趋势偏空，建议控制仓位，等待企稳信号。")

        if not warnings:
            warnings.append("✅ **市场状态正常**: 未检测到极端风险信号。")

        return "### ⚠️ 风险预警\n" + "\n".join(warnings) + "\n"

    def run(self) -> Tuple[bool, str]:
        """执行每日投研简报生成"""
        now = datetime.now(TZ_BJS)
        now_str = now.strftime('%Y-%m-%d %H:%M')
        today_str = now.strftime('%Y%m%d')

        log.info(f"🚀 启动每日投研简报生成... 时间: {now_str}")

        try:
            df_raw = fetch_spot()
            if df_raw is None or df_raw.empty:
                return False, "❌ 数据源获取失败，无法生成简报"

            c_conf = Config()
            df_clean, m_ok, m_msg, idx_ret, m_overheated, m_regime, vol_surge = extract_market_context(df_raw, c_conf)

            hot_sectors_map = fetch_hot_sectors()

            header = (
                f"## 📊 每日投研简报\n"
                f"> **{now_str}** | AI量化系统自动生成\n>\n"
                f"> ⚠️ 本报告仅供参考，不构成投资建议\n\n"
                f"---\n\n"
            )

            content = header

            content += generate_macro_section() + "\n\n---\n\n"

            content += m_msg + "\n\n---\n\n"

            content += self.generate_index_comparison() + "\n\n---\n\n"

            content += self.generate_etf_rotation_analysis(df_raw) + "\n\n---\n\n"

            content += self.generate_sector_rotation_analysis(hot_sectors_map) + "\n\n---\n\n"

            content += self.generate_smart_money_analysis() + "\n\n---\n\n"

            content += self.generate_risk_warning(m_overheated, m_regime) + "\n\n---\n\n"

            footer = (
                f"---\n\n"
                f"### 📋 报告说明\n"
                f"- 数据来源: 东方财富、同花顺、腾讯财经\n"
                f"- 生成时间: {now_str}\n"
                f"- 系统版本: AI量化 v2.0\n\n"
                f"> **📌 免责声明**: 本报告由程序自动生成，股市有风险，投资需谨慎。\n"
            )
            content += footer

            if self.config.DINGTALK_WEBHOOK:
                try:
                    NotificationGateway.send(
                        f'📊 每日投研简报 {now.strftime("%m-%d")}',
                        content,
                        template='blue'
                    )
                    log.info("✅ 每日投研简报已成功发送到钉钉")
                    return True, "简报生成并发送成功"
                except Exception as e:
                    log.error(f"❌ 钉钉通知发送失败: {e}")
                    return False, f"简报生成成功但通知发送失败: {e}"
            else:
                log.warning("⚠️ 未配置钉钉Webhook，简报仅生成不发送")
                return True, "简报生成成功（未配置通知渠道）"

        except Exception as e:
            log.error(f"❌ 每日投研简报生成失败: {e}")
            return False, f"简报生成失败: {e}"


def main():
    print("=" * 60)
    print("🚀 每日投研简报生成器启动")
    print("=" * 60)

    generator = DailyBriefingGenerator()
    success, message = generator.run()

    print("\n" + "=" * 60)
    if success:
        print(f"✅ 执行结果: {message}")
    else:
        print(f"❌ 执行结果: {message}")
    print("=" * 60)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
