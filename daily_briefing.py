#!/usr/bin/env python3
"""
每日投研简报生成器
Daily Investment Briefing Generator

功能：整合市场分析、ETF轮动、行业热点、股票信号，生成完整的每日投研简报并发送至钉钉
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

import akshare as ak
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

TZ_BJS = __import__('pytz').timezone('Asia/Shanghai')

def get_env(key: str, default: str = '') -> str:
    return os.environ.get(key, default)

DINGTALK_WEBHOOK = get_env('DINGTALK_WEBHOOK', '')
NOTIFY_SEC_KEYWORD = get_env('NOTIFY_SEC_KEYWORD', 'AI量化').strip()

class DataFetcher:
    @staticmethod
    def get_etf_list() -> pd.DataFrame:
        try:
            df = ak.fund_etf_spot_em()
            if df is not None and not df.empty:
                log.info(f"成功获取ETF列表，共 {len(df)} 只")
                return df
        except Exception as e:
            log.warning(f"获取ETF列表失败: {e}")
        return pd.DataFrame()

    @staticmethod
    def get_hot_sectors() -> list:
        try:
            df = ak.stock_board_industry_name_em()
            if df is not None and not df.empty:
                df = df.nlargest(10, '涨跌幅')
                sectors = []
                for _, row in df.iterrows():
                    sectors.append({
                        'name': row['板块名称'],
                        'pct': row['涨跌幅']
                    })
                return sectors
        except Exception as e:
            log.warning(f"获取行业热点失败: {e}")
        return []

    @staticmethod
    def get_index_data(symbol: str = "sh000001") -> Optional[pd.DataFrame]:
        try:
            df = ak.stock_zh_index_daily_em(symbol=symbol)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            log.warning(f"获取指数数据失败: {e}")
        return None

    @staticmethod
    def get_northbound_flow() -> tuple:
        try:
            df = ak.stock_em_hsgt_north_net_flow_in(indicator="沪深港通")
            if df is not None and not df.empty:
                col = 'value' if 'value' in df.columns else df.columns[-1]
                today_flow = float(df.iloc[-1][col]) / 1e8
                return today_flow
        except Exception:
            pass
        return 0.0

    @staticmethod
    def get_market_breadth() -> dict:
        try:
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                up_count = (df['涨跌幅'] > 0).sum()
                down_count = (df['涨跌幅'] < 0).sum()
                zt_count = (df['涨跌幅'] >= 9.9).sum()
                dt_count = (df['涨跌幅'] <= -9.9).sum()
                total_amt = df['成交额'].sum() / 1e8
                return {
                    'up': up_count,
                    'down': down_count,
                    'zt': zt_count,
                    'dt': dt_count,
                    'amount': total_amt
                }
        except Exception as e:
            log.warning(f"获取市场广度失败: {e}")
        return {'up': 0, 'down': 0, 'zt': 0, 'dt': 0, 'amount': 0}


class ETFRotationAnalyzer:
    ETF_CATEGORIES = {
        '宽基指数': ['510300', '510500', '159915', '512000', '512200', '515000', '159919', '510050'],
        '行业主题': ['512760', '512980', '159819', '515050', '159928', '512690', '512800', '512660'],
        '红利低波': ['515080', '512890', '515100', '510850'],
        '商品期货': ['518880', '159980', '159995', '161815'],
        '海外市场': ['513500', '513100', '159941', '513050']
    }

    def __init__(self):
        self.etf_df = DataFetcher.get_etf_list()

    def get_rotation_analysis(self) -> str:
        if self.etf_df.empty:
            return "### 📊 ETF轮动追踪\n\n⚠️ ETF数据获取失败\n\n"

        results = {}
        for category, codes in self.ETF_CATEGORIES.items():
            category_etfs = self.etf_df[self.etf_df['代码'].isin(codes)]
            if not category_etfs.empty:
                top_performers = category_etfs.nlargest(3, '涨跌幅')[['代码', '名称', '涨跌幅', '成交额']]
                results[category] = top_performers.to_dict('records')

        msg = "### 📊 ETF轮动追踪\n\n"
        for category, etfs in results.items():
            msg += f"**【{category}】**\n"
            for etf in etfs:
                pct = etf['涨跌幅']
                pct_str = f"{pct:+.2f}%" if isinstance(pct, (int, float)) else pct
                emoji = "🟢" if pct > 0 else "🔴" if pct < 0 else "⚪"
                msg += f"- {emoji} `{etf['代码']}` {etf['名称']}: {pct_str}\n"
            msg += "\n"

        if not results:
            msg = "### 📊 ETF轮动追踪\n\n暂无有效ETF数据\n\n"

        return msg


class MarketAnalyzer:
    def __init__(self):
        self.index_df = DataFetcher.get_index_data()
        self.breadth = DataFetcher.get_market_breadth()
        self.north_flow = DataFetcher.get_northbound_flow()
        self.hot_sectors = DataFetcher.get_hot_sectors()

    def get_ma_trend(self, cl: pd.Series) -> tuple:
        if len(cl) < 60:
            return "数据不足", ""
        ma5 = cl.rolling(5).mean().iloc[-1]
        ma20 = cl.rolling(20).mean().iloc[-1]
        ma60 = cl.rolling(60).mean().iloc[-1]
        close = cl.iloc[-1]

        mas = [ma5, ma20, ma60]
        max_ma, min_ma = max(mas), min(mas)
        spread = (max_ma - min_ma) / min_ma

        if spread < 0.02:
            return "均线粘连", "面临方向性变盘选择"
        elif ma5 > ma20 > ma60:
            return "三线开花(强势多头)", "全面多头排列，上行动能极强"
        elif ma5 < ma20 < ma60:
            return "空头瀑布(极度弱势)", "全面空头排列，下行趋势加速"
        elif ma60 > ma20 and ma5 > ma20:
            return "筑底反弹", "中长线偏空但短期均线拐头向上"
        else:
            return "震荡分化", "长短均线方向不一，无明显单边趋势"

    def analyze(self) -> str:
        msg = "### 📈 A股大盘诊断\n\n"

        if self.index_df is not None and not self.index_df.empty:
            cl = self.index_df['close']
            ma20 = cl.rolling(20).mean().iloc[-1]
            pct = (cl.iloc[-1] - cl.iloc[-2]) / cl.iloc[-2] * 100
            trend_name, trend_desc = self.get_ma_trend(cl)

            msg += f"- **大盘趋势**：`{trend_name}` - {trend_desc}\n"
            msg += f"- **上证指数**：`{cl.iloc[-1]:.2f}` (今日 **{pct:+.2f}%**)\n"
            msg += f"- **20日均线**：`{ma20:.2f}`\n"
        else:
            msg += "- ⚠️ 指数数据获取失败\n\n"

        b = self.breadth
        total = b['up'] + b['down']
        breadth = b['up'] / total * 100 if total > 0 else 0

        msg += f"- **市场广度**：红盘 `{b['up']}` 家 / 绿盘 `{b['down']}` 家\n"
        msg += f"- **涨跌停**：涨停 `{b['zt']}` / 跌停 `{b['dt']}`\n"
        msg += f"- **两市量能**：约 `{b['amount']:.0f}` 亿元\n"
        msg += f"- **市场广度比**：{breadth:.1f}%\n\n"

        if self.north_flow != 0:
            if self.north_flow > 30:
                msg += f"- 🌊 **北向资金**：大幅流入 **+{self.north_flow:.0f}亿** 🚀\n\n"
            elif self.north_flow < -30:
                msg += f"- ❄️ **北向资金**：大幅流出 **{self.north_flow:.0f}亿**\n\n"
            else:
                msg += f"- ⚖️ **北向资金**：温和 ({self.north_flow:+.0f}亿)\n\n"

        if self.hot_sectors:
            msg += "**🔥 今日主线板块：**\n"
            for i, sector in enumerate(self.hot_sectors[:5], 1):
                pct = sector['pct']
                pct_str = f"{pct:+.2f}%" if isinstance(pct, float) else pct
                emoji = "🟢" if pct > 0 else "🔴"
                msg += f"- {emoji} {i}. **{sector['name']}** ({pct_str})\n"
            msg += "\n"

        return msg


class MacroAnalyzer:
    @staticmethod
    def analyze() -> str:
        msg = "### 🌍 隔夜外围市场\n\n"
        try:
            import yfinance as yf
            tickers = yf.Tickers("^TNX ^VIX ^GSPC ^DJI IXIC HG=F GC=F CL=F")
            hist = tickers.history(period="5d")
            close_df = hist['Close']

            def get_pct(ticker):
                try:
                    s = close_df[ticker].dropna()
                    if len(s) >= 2:
                        last = s.iloc[-1]
                        prev = s.iloc[-2]
                        return last, (last - prev) / prev * 100
                except Exception:
                    pass
                return 0.0, 0.0

            sp500, sp_pct = get_pct('^GSPC')
            nasdaq, nasdaq_pct = get_pct('^IXIC')
            dow, dow_pct = get_pct('^DJI')
            vix, vix_pct = get_pct('^VIX')
            tnx, tnx_pct = get_pct('^TNX')
            gold, gold_pct = get_pct('GC=F')
            oil, oil_pct = get_pct('CL=F')

            if sp500 > 0:
                msg += f"- **标普500**：`{sp500:.2f}` ({sp_pct:+.2f}%)\n"
            if nasdaq > 0:
                msg += f"- **纳斯达克**：`{nasdaq:.2f}` ({nasdaq_pct:+.2f}%)\n"
            if dow > 0:
                msg += f"- **道琼斯**：`{dow:.2f}` ({dow_pct:+.2f}%)\n"

            msg += f"- **恐慌指数(VIX)**：`{vix:.2f}` ({vix_pct:+.2f}%) "
            if vix > 25:
                msg += "⚠️ **极度恐慌**\n"
            elif vix > 20:
                msg += "⚡ **谨慎**\n"
            else:
                msg += "✅ 稳定\n"

            if tnx > 0:
                msg += f"- **美债10Y**：`{tnx:.2f}%` ({tnx_pct:+.2f}%)\n"
            if gold > 0:
                msg += f"- **黄金**：`{gold:.2f}` ({gold_pct:+.2f}%)\n"
            if oil > 0:
                msg += f"- **原油**：`{oil:.2f}` ({oil_pct:+.2f}%)\n"
            msg += "\n"

        except Exception as e:
            msg += f"外围数据获取失败 (可能被限流)，请稍后重试\n\n"
            log.warning(f"宏观数据获取失败: {e}")

        return msg


class StockSignalGenerator:
    def __init__(self):
        self.main_module = None
        self._load_main()

    def _load_main(self):
        try:
            import main as main_module
            self.main_module = main_module
            log.info("成功加载 main.py 模块")
        except Exception as e:
            log.warning(f"无法加载 main.py: {e}")

    def generate_signals(self) -> str:
        if self.main_module is None:
            return self._generate_mock_signals()

        try:
            log.info("正在调用 main.py 获取股票信号...")
            sigs, watch, pushed, pool_size, m_msg, total_mkt = self.main_module.get_signals()

            msg = ""
            has_any_signal = any(sigs.values()) if isinstance(sigs, dict) else bool(sigs)

            if has_any_signal:
                if isinstance(sigs, dict):
                    if sigs.get('Resonance'):
                        msg += "### 🔥 多周期共振信号\n\n"
                        for s in sigs['Resonance'][:3]:
                            msg += self._format_signal(s)
                        msg += "\n"

                    if sigs.get('T+1'):
                        msg += "### ⚡ T+1 短线信号\n\n"
                        for s in sigs['T+1'][:3]:
                            msg += self._format_signal(s)
                        msg += "\n"

                    if sigs.get('T+5'):
                        msg += "### 🌊 T+5 波段信号\n\n"
                        for s in sigs['T+5'][:3]:
                            msg += self._format_signal(s)
                        msg += "\n"
                else:
                    for s in sigs[:5]:
                        msg += self._format_signal(s)
                    msg += "\n"
            else:
                msg += "### 🎯 股票信号\n\n"
                msg += "✅ 今日未发现 B+ 级以上核心机会，建议空仓防守。\n\n"

            if watch:
                msg += "### 👁️ 候补观察池\n\n"
                for name, code, score, price in watch[:5]:
                    msg += f"- `{code}` **{name}** (¥{price}) 得分: **{score}**\n"
                msg += "\n"

            return msg

        except Exception as e:
            log.warning(f"获取股票信号失败: {e}")
            return self._generate_mock_signals()

    def _format_signal(self, s) -> str:
        try:
            return (
                f"- **{s.name}** (`{s.code}`)\n"
                f"  - 评分：`{s.score}` 分 {s.level}\n"
                f"  - 现价：`¥{s.price}` ({s.pct_chg})\n"
                f"  - 止损：`¥{s.stop_loss}` | 目标：`¥{s.target1}`\n\n"
            )
        except Exception:
            return ""

    def _generate_mock_signals(self) -> str:
        msg = "### 🎯 股票信号\n\n"
        msg += "⚠️ 实时信号获取功能暂不可用，请确保已正确配置环境变量。\n\n"
        msg += "**建议检查项：**\n"
        msg += "1. TUSHARE_TOKEN 是否配置\n"
        msg += "2. 网络连接是否正常\n"
        msg += "3. main.py 模块是否可正常导入\n\n"
        return msg


class BriefingReport:
    def __init__(self):
        self.now = datetime.now(TZ_BJS)
        self.sections = []

    def build(self) -> str:
        self._add_macro_section()
        self._add_market_section()
        self._add_etf_section()
        self._add_stock_section()
        self._add_summary_section()
        return self._compose()

    def _add_macro_section(self):
        self.sections.append(MacroAnalyzer.analyze())

    def _add_market_section(self):
        self.sections.append(MarketAnalyzer().analyze())

    def _add_etf_section(self):
        try:
            self.sections.append(ETFRotationAnalyzer().get_rotation_analysis())
        except Exception as e:
            log.warning(f"ETF轮动分析失败: {e}")
            self.sections.append("### 📊 ETF轮动追踪\n\n⚠️ ETF数据获取失败\n\n")

    def _add_stock_section(self):
        self.sections.append(StockSignalGenerator().generate_signals())

    def _add_summary_section(self):
        now_str = self.now.strftime('%Y-%m-%d %H:%M')
        summary = (
            "---\n\n"
            "### 📋 每日总结\n\n"
            f"> 生成时间：{now_str}\n\n"
            "**免责声明**：本报告由量化模型自动生成，仅供技术交流与策略复盘，"
            "**绝不构成任何投资建议**。股市有风险，入市需谨慎，盈亏请自负。\n\n"
            "---\n\n"
            "### 🤔 每日灵魂拷问\n\n"
            "如果明天买入的股票跌了 5%，我会焦虑得睡不着觉吗？\n\n"
            "> **如果会，请把你准备买入的金额【再砍掉一半】！投资是为了生活更好，不是花钱找罪受。**"
        )
        self.sections.append(summary)

    def _compose(self) -> str:
        header = (
            f"## 🤖 AI量化·每日投研简报\n\n"
            f"> **{self.now.strftime('%Y-%m-%d %A')}**\n\n"
            "---\n\n"
        )
        return header + "\n\n".join(self.sections)


class DingTalkNotifier:
    @staticmethod
    def send(title: str, content: str) -> bool:
        if not DINGTALK_WEBHOOK:
            log.warning("⚠️ 未配置 DINGTALK_WEBHOOK，跳过发送")
            return False

        headers = {"Content-Type": "application/json"}
        final_title = title if NOTIFY_SEC_KEYWORD in title else f"{NOTIFY_SEC_KEYWORD} | {title}"
        final_content = content
        if NOTIFY_SEC_KEYWORD not in final_content:
            final_content = f"### {NOTIFY_SEC_KEYWORD}\n\n{final_content}"

        payload = {
            'msgtype': 'markdown',
            'markdown': {
                'title': final_title,
                'text': final_content
            }
        }

        try:
            import requests
            res = requests.post(DINGTALK_WEBHOOK, json=payload, headers=headers, timeout=10)
            res_dict = res.json()

            if res_dict.get('errcode', 0) != 0:
                log.error(f"钉钉推送失败: {res_dict}")
                return False

            log.info(f"✅ 钉钉推送成功: {title}")
            return True

        except Exception as e:
            log.error(f"钉钉推送异常: {e}")
            return False


def main():
    log.info("🚀 每日投研简报生成器启动...")
    now_str = datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M:%S')
    log.info(f"📅 当前时间: {now_str}")

    report = BriefingReport()
    content = report.build()

    print("\n" + "="*80)
    print("📊 每日投研简报预览")
    print("="*80)
    print(content)
    print("="*80 + "\n")

    if DINGTALK_WEBHOOK:
        success = DingTalkNotifier.send("🤖 AI量化·每日投研简报", content)
        if success:
            log.info("✅ 简报已成功发送至钉钉")
        else:
            log.error("❌ 钉钉发送失败，请检查配置")
    else:
        log.warning("⚠️ 未配置 DINGTALK_WEBHOOK，已跳过钉钉推送")

    log.info("🏁 每日投研简报生成完毕")


if __name__ == '__main__':
    main()
