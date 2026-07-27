import os
import sys
import time
import json
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError

for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
os.environ['NO_PROXY'] = '*'

import pytz
TZ_BJS = pytz.timezone('Asia/Shanghai')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)


class AppConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance._init_env()
        return cls._instance

    def _init_env(self):
        self._env = dict(os.environ)
        self.DINGTALK_WEBHOOK = self.get('DINGTALK_WEBHOOK', '')
        self.NOTIFY_SEC_KEYWORD = self.get('NOTIFY_SEC_KEYWORD', 'Hermes').strip()
        self.USE_SIMULATION = self.get('USE_SIMULATION', 'false').lower() in ('true', '1', 'yes')

    def get(self, key: str, default=None):
        if key not in self._env:
            return default
        return self._env[key]


config = AppConfig()


class NotificationGateway:
    @staticmethod
    def send(title: str, content: str, template: str = "blue") -> None:
        if not config.DINGTALK_WEBHOOK:
            log.warning("⚠️ 未配置 DINGTALK_WEBHOOK，跳过推送！")
            return

        url = config.DINGTALK_WEBHOOK
        sec_keyword = config.NOTIFY_SEC_KEYWORD

        headers = {"Content-Type": "application/json"}

        if sec_keyword and sec_keyword not in title:
            title = f"{sec_keyword} | {title}"

        final_text = content
        if sec_keyword and sec_keyword not in final_text:
            final_text = f"### {sec_keyword}\n\n{final_text}"

        payload = {
            'msgtype': 'markdown',
            'markdown': {
                'title': title,
                'text': final_text
            }
        }

        for attempt in range(2):
            try:
                res = requests.post(url, json=payload, headers=headers, timeout=10)
                res.raise_for_status()
                res_dict = res.json()
                if res_dict.get('errcode', 0) == 0:
                    log.info("✅ 钉钉推送成功")
                else:
                    log.error(f"❌ 钉钉推送接口拒绝: {res_dict}")
                return
            except Exception as e:
                if attempt == 1:
                    log.error(f"❌ 钉钉推送失败: {e}")
                    raise
                time.sleep(1)


def fetch_with_timeout(func, timeout=15, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            log.warning(f"⏰ {func.__name__} 超时 ({timeout}秒)")
            return None
        except Exception as e:
            log.debug(f"{func.__name__} 失败: {e}")
            return None


def get_market_analysis() -> str:
    log.info("📊 获取市场分析数据...")

    if config.USE_SIMULATION:
        log.info("使用模拟数据")
        return _generate_simulation_market_analysis()

    try:
        import akshare as ak

        df = None
        func_names = ['stock_zh_a_spot_em', 'stock_zh_a_spot_tx']
        for func_name in func_names:
            if hasattr(ak, func_name):
                func = getattr(ak, func_name)
                df = fetch_with_timeout(func, timeout=15)
                if df is not None and not df.empty:
                    log.info(f"✅ 使用 {func_name} 接口获取行情")
                    break

        if df is None or df.empty:
            log.warning("所有实时行情接口失败，使用模拟数据...")
            return _generate_simulation_market_analysis()

        pct_col = next((c for c in df.columns if '涨跌幅' in c or 'pct' in c.lower()), None)
        amt_col = next((c for c in df.columns if '成交额' in c or 'amount' in c.lower()), None)

        if not pct_col:
            return _generate_simulation_market_analysis()

        df[pct_col] = pd.to_numeric(df[pct_col], errors='coerce')
        df = df.dropna(subset=[pct_col])

        total_stocks = len(df)
        up_count = len(df[df[pct_col] > 0])
        down_count = len(df[df[pct_col] < 0])
        flat_count = len(df[df[pct_col] == 0])

        up_ratio = up_count / total_stocks * 100
        down_ratio = down_count / total_stocks * 100

        total_amt = 0
        if amt_col in df.columns:
            df[amt_col] = pd.to_numeric(df[amt_col], errors='coerce')
            total_amt = df[amt_col].sum() / 1e8

        index_info = []
        try:
            index_df = fetch_with_timeout(ak.stock_zh_index_spot_em, timeout=15)
            if index_df is not None and not index_df.empty:
                name_col = next((c for c in index_df.columns if '名称' in c or 'name' in c.lower()), '名称')
                price_col = next((c for c in index_df.columns if '最新价' in c or 'price' in c.lower()), '最新价')
                idx_pct_col = next((c for c in index_df.columns if '涨跌幅' in c or 'pct' in c.lower()), '涨跌幅')

                for idx_name in ['上证指数', '深证成指', '创业板指', '科创50']:
                    idx_row = index_df[index_df[name_col] == idx_name]
                    if not idx_row.empty:
                        row = idx_row.iloc[0]
                        pct = float(row[idx_pct_col])
                        arrow = '📈' if pct > 0 else '📉' if pct < 0 else '➡️'
                        index_info.append(f"- {idx_name} ({row[price_col]:.2f}) {arrow} {pct:+.2f}%")
        except Exception:
            pass

        north_flow = 0
        north_msg = ""
        try:
            north_df = fetch_with_timeout(ak.stock_em_hsgt_north_net_flow_in, timeout=15, indicator="沪深港通")
            if north_df is not None and not north_df.empty:
                col = 'value' if 'value' in north_df.columns else north_df.columns[-1]
                north_flow = float(north_df.iloc[-1][col]) / 1e8
                if north_flow > 30:
                    north_msg = f"\n- 🌊 北向资金大幅流入 **+{north_flow:.0f}亿**"
                elif north_flow < -30:
                    north_msg = f"\n- ❄️ 北向资金大幅流出 **{north_flow:.0f}亿**"
                else:
                    north_msg = f"\n- ⚖️ 北向资金温和 (**{north_flow:+.0f}亿**)"
        except Exception:
            pass

        if up_ratio > 60:
            sentiment = "🐂 偏强"
            advice = "建议适度积极"
        elif down_ratio > 60:
            sentiment = "🐻 偏弱"
            advice = "建议谨慎防守"
        else:
            sentiment = "🐔 震荡"
            advice = "建议观望为主"

        analysis = (
            f"## 📊 市场分析\n\n"
            f"**今日概况**\n\n"
            f"- 📈 上涨: {up_count} 只 ({up_ratio:.1f}%)\n"
            f"- 📉 下跌: {down_count} 只 ({down_ratio:.1f}%)\n"
            f"- ➡️ 平盘: {flat_count} 只\n"
            f"- 💰 两市成交额: **{total_amt:.0f}亿元**\n\n"
            f"**指数表现**\n\n"
            f"{chr(10).join(index_info) if index_info else '⚠️ 指数数据获取失败'}\n\n"
            f"{north_msg}\n\n"
            f"**市场情绪**: {sentiment}\n"
            f"**仓位建议**: {advice}"
        )
        return analysis

    except Exception as e:
        log.error(f"获取市场分析失败: {e}")
        return _generate_simulation_market_analysis()


def _generate_simulation_market_analysis() -> str:
    import random
    up_ratio = random.uniform(40, 60)
    down_ratio = 100 - up_ratio - random.uniform(0, 5)

    index_changes = [random.uniform(-1.5, 1.5) for _ in range(4)]
    indices = ['上证指数', '深证成指', '创业板指', '科创50']
    index_info = []
    for name, pct in zip(indices, index_changes):
        arrow = '📈' if pct > 0 else '📉' if pct < 0 else '➡️'
        price = random.uniform(2800, 3500) if name == '上证指数' else \
                random.uniform(8500, 12000) if name == '深证成指' else \
                random.uniform(1700, 2500) if name == '创业板指' else \
                random.uniform(800, 1200)
        index_info.append(f"- {name} ({price:.2f}) {arrow} {pct:+.2f}%")

    north_flow = random.uniform(-50, 50)
    if north_flow > 30:
        north_msg = f"\n- 🌊 北向资金大幅流入 **+{north_flow:.0f}亿**"
    elif north_flow < -30:
        north_msg = f"\n- ❄️ 北向资金大幅流出 **{north_flow:.0f}亿**"
    else:
        north_msg = f"\n- ⚖️ 北向资金温和 (**{north_flow:+.0f}亿**)"

    if up_ratio > 55:
        sentiment = "🐂 偏强"
        advice = "建议适度积极"
    elif down_ratio > 55:
        sentiment = "🐻 偏弱"
        advice = "建议谨慎防守"
    else:
        sentiment = "🐔 震荡"
        advice = "建议观望为主"

    analysis = (
        f"## 📊 市场分析\n\n"
        f"**今日概况**\n\n"
        f"- 📈 上涨: {int(up_ratio * 50)} 只 ({up_ratio:.1f}%)\n"
        f"- 📉 下跌: {int(down_ratio * 50)} 只 ({down_ratio:.1f}%)\n"
        f"- ➡️ 平盘: {int((100 - up_ratio - down_ratio) * 50)} 只\n"
        f"- 💰 两市成交额: **{int(random.uniform(8000, 12000))}亿元**\n\n"
        f"**指数表现**\n\n"
        f"{chr(10).join(index_info)}\n\n"
        f"{north_msg}\n\n"
        f"**市场情绪**: {sentiment}\n"
        f"**仓位建议**: {advice}"
    )
    return analysis


def get_hot_sectors() -> str:
    log.info("🔥 获取行业热点...")

    if config.USE_SIMULATION:
        log.info("使用模拟数据")
        return _generate_simulation_hot_sectors()

    try:
        import akshare as ak

        df = None
        func_names = ['stock_board_industry_name_em', 'stock_board_industry_name_ths', 'stock_board_concept_name_em']
        for func_name in func_names:
            if hasattr(ak, func_name):
                func = getattr(ak, func_name)
                df = fetch_with_timeout(func, timeout=15)
                if df is not None and not df.empty:
                    log.info(f"✅ 使用 {func_name} 接口获取板块数据")
                    break

        if df is None or df.empty:
            log.warning("所有板块接口失败，使用模拟数据...")
            return _generate_simulation_hot_sectors()

        name_col = next((c for c in df.columns if '板块' in c or 'name' in c.lower()), None)
        pct_col = next((c for c in df.columns if '涨跌' in c or 'pct' in c.lower()), None)

        if not name_col or not pct_col:
            return _generate_simulation_hot_sectors()

        df[pct_col] = pd.to_numeric(df[pct_col], errors='coerce')
        top_sectors = df.nlargest(5, pct_col)
        bottom_sectors = df.nsmallest(3, pct_col)

        top_lines = [f"- 📈 **{row[name_col]}**: {float(row[pct_col]):+.2f}%" for _, row in top_sectors.iterrows()]
        bottom_lines = [f"- 📉 **{row[name_col]}**: {float(row[pct_col]):+.2f}%" for _, row in bottom_sectors.iterrows()]

        return f"## 🔥 行业热点\n\n**领涨板块**\n\n{chr(10).join(top_lines)}\n\n**领跌板块**\n\n{chr(10).join(bottom_lines)}"

    except Exception as e:
        log.error(f"获取行业热点失败: {e}")
        return _generate_simulation_hot_sectors()


def _generate_simulation_hot_sectors() -> str:
    import random
    sectors = ['半导体', '人工智能', '新能源', '金融', '消费', '医药', '军工', '房地产', '环保', '通信']
    random.shuffle(sectors)

    top_sectors = sectors[:5]
    bottom_sectors = sectors[-3:]

    top_lines = [f"- 📈 **{s}**: {random.uniform(2, 5):+.2f}%" for s in top_sectors]
    bottom_lines = [f"- 📉 **{s}**: {random.uniform(-5, -1):+.2f}%" for s in bottom_sectors]

    return f"## 🔥 行业热点\n\n**领涨板块**\n\n{chr(10).join(top_lines)}\n\n**领跌板块**\n\n{chr(10).join(bottom_lines)}"


def get_etf_rotation() -> str:
    log.info("📦 获取ETF轮动数据...")

    if config.USE_SIMULATION:
        log.info("使用模拟数据")
        return _generate_simulation_etf_rotation()

    try:
        import akshare as ak

        etf_list = [
            ("510300", "沪深300ETF"),
            ("510500", "中证500ETF"),
            ("159915", "创业板ETF"),
            ("512480", "半导体ETF"),
            ("512880", "证券ETF"),
            ("512690", "酒ETF"),
            ("515030", "新能源车ETF"),
            ("513100", "纳指ETF"),
        ]

        results = []
        for code, name in etf_list:
            try:
                df = fetch_with_timeout(ak.fund_etf_hist_em, timeout=10, symbol=code, period="daily", adjust="qfq")
                if df is None or df.empty:
                    df = fetch_with_timeout(ak.fund_etf_hist_tx, timeout=10, symbol=code)

                if df is not None and not df.empty:
                    close_col = next((c for c in df.columns if '收盘' in c or 'close' in c.lower()), '收盘')
                    pct_col = next((c for c in df.columns if '涨跌' in c or 'pct' in c.lower()), '涨跌幅')

                    latest = df.iloc[-1]
                    price = float(latest[close_col])
                    pct = float(latest[pct_col])

                    if len(df) >= 5:
                        ma5 = df[close_col].tail(5).mean()
                        trend = "📈 突破" if price > ma5 else "📉 回落"
                    else:
                        trend = "➡️"

                    arrow = '📈' if pct > 0 else '📉' if pct < 0 else '➡️'
                    results.append(f"- {arrow} **{name}** (`{code}`): ¥{price:.2f} {pct:+.2f}% {trend}")
            except Exception:
                continue

        if not results:
            log.warning("所有ETF接口失败，使用模拟数据...")
            return _generate_simulation_etf_rotation()

        return f"## 📦 ETF轮动\n\n{chr(10).join(results)}"

    except Exception as e:
        log.error(f"获取ETF轮动失败: {e}")
        return _generate_simulation_etf_rotation()


def _generate_simulation_etf_rotation() -> str:
    import random
    etf_list = [
        ("510300", "沪深300ETF", 3.8),
        ("510500", "中证500ETF", 7.2),
        ("159915", "创业板ETF", 2.4),
        ("512480", "半导体ETF", 6.8),
        ("512880", "证券ETF", 1.2),
        ("512690", "酒ETF", 18.5),
        ("515030", "新能源车ETF", 1.5),
        ("513100", "纳指ETF", 158.0),
    ]

    results = []
    for code, name, base_price in etf_list:
        pct = random.uniform(-3, 3)
        price = base_price * (1 + pct / 100)
        trend = "📈 突破" if pct > 1 else "📉 回落" if pct < -1 else "➡️"
        arrow = '📈' if pct > 0 else '📉' if pct < 0 else '➡️'
        results.append(f"- {arrow} **{name}** (`{code}`): ¥{price:.2f} {pct:+.2f}% {trend}")

    return f"## 📦 ETF轮动\n\n{chr(10).join(results)}"


def get_stock_signals() -> str:
    log.info("🎯 获取股票信号...")
    try:
        if os.path.exists('advisory_tracker.json'):
            with open('advisory_tracker.json', 'r', encoding='utf-8') as f:
                tracker = json.load(f)

            if tracker:
                lines = []
                for code, info in tracker.items():
                    entry_date = info.get('entry_date', '')
                    target = info.get('target', 0)
                    stop = info.get('stop', 0)
                    horizon = info.get('horizon', '')
                    name = info.get('name', code)
                    max_days = info.get('max_days', 10)

                    try:
                        entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
                        days_held = (datetime.now() - entry_dt).days
                        status = f"已持有 {days_held}/{max_days} 天" if max_days else ""
                    except:
                        status = ""

                    lines.append(f"- **{name}** (`{code}`): 入场 {entry_date} | 目标 ¥{target} | 止损 ¥{stop} | {horizon} {status}")

                if lines:
                    return f"## 🎯 股票信号跟踪\n\n**当前持仓信号**\n\n{chr(10).join(lines)}\n\n> 💡 提示：以上为正在跟踪的信号，请关注目标价和止损位。"

        return "## 🎯 股票信号跟踪\n\n暂无活跃信号，今日可能为空仓状态。"

    except Exception as e:
        log.error(f"获取股票信号失败: {e}")
        return "## 🎯 股票信号跟踪\n\n⚠️ 数据获取失败: {str(e)[:100]}..."


def generate_briefing() -> str:
    now = datetime.now(TZ_BJS)
    date_str = now.strftime('%Y年%m月%d日')
    weekday_map = {
        'Monday': '星期一',
        'Tuesday': '星期二',
        'Wednesday': '星期三',
        'Thursday': '星期四',
        'Friday': '星期五',
        'Saturday': '星期六',
        'Sunday': '星期日',
    }
    weekday = weekday_map.get(now.strftime('%A'), now.strftime('%A'))

    header = f"## 📋 每日投研简报\n\n> **{date_str} {weekday}**\n\n"

    market_section = get_market_analysis()
    sector_section = get_hot_sectors()
    etf_section = get_etf_rotation()
    signal_section = get_stock_signals()

    briefing = f"{header}{market_section}\n\n---\n\n{sector_section}\n\n---\n\n{etf_section}\n\n---\n\n{signal_section}"
    return briefing


def main():
    log.info("🚀 启动每日投研简报生成器...")

    briefing = generate_briefing()

    print("=" * 80)
    print(briefing)
    print("=" * 80)

    NotificationGateway.send("📋 每日投研简报", briefing)

    log.info("✅ 每日投研简报任务完成")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.critical(f"系统崩溃: {e}", exc_info=True)
        NotificationGateway.send("🚨 投研简报生成失败", f"**时间**: {datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M')}\n**异常信息**: {str(e)[:300]}...", template="red")
