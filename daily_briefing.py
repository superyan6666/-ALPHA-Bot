"""
daily_briefing.py
==================

每日投研简报生成器
功能清单：
  1. A股市场深度诊断 (大盘趋势 / 涨跌家数 / 涨停跌停 / 外资流向 / 情绪估值)
  2. 个股量化信号 (来自 main.py 的全周期共振精选)
  3. ETF 轮动雷达 (宽基 / 行业 / 主题 / 跨境 ETF 的动量与趋势排名)
  4. 行业热点主线 (东方财富 / 同花顺 板块榜 + 板块成分股强度分析)
  5. 隔夜外围宏观快报 (美股 / VIX / 黄金 / 原油 / 美债)

依赖：main.py 中的 DataProxy / NotificationGateway / extract_market_context / generate_macro_section
      feature_engine.py 中的 ML 特征工程 (可选)

运行方式：
    python daily_briefing.py

环境变量：
    DINGTALK_WEBHOOK  - 钉钉机器人 webhook (必填，若要推送)
    FEISHU_WEBHOOK    - 飞书机器人 webhook (可选)
    RUN_MODE          - 'normal' (默认) 或 'market_only' / 'morning'
    NOTIFY_SEC_KEYWORD - 消息安全词 (默认 'AI量化')
"""

import os
import sys
import time
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO'), logging.INFO),
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger("daily_briefing")

# ---------------------------------------------------------------------------
# 0. 挂载 main.py 的核心能力 (复用 DataProxy / 通知网关 / 已有算法)
# ---------------------------------------------------------------------------
try:
    import pytz
    TZ_BJS = pytz.timezone('Asia/Shanghai')
except Exception:
    TZ_BJS = None

# ---------------------------------------------------------------------------
# 0.1 为缺失的重量级 ML 依赖注入占位符 (defensive import)
#     main.py / ml_engine.py 顶层 import 了 xgboost / torch 等模块，
#     但每日简报实际上并不走 ML 推理路径，仅复用数据层和通知网关。
#     若用户环境缺少这些 ML 库，我们用 stub 占位，保证主流程仍能跑通。
# ---------------------------------------------------------------------------
def _inject_missing_module_stubs():
    """
    为 main.py -> ml_engine.py/feature_engine.py 链路中涉及的重量级 ML 库注入占位模块。
    这些库在每日简报场景下并不会真正被调用（简报不走模型推理路径），
    但 main.py 顶层 import 会加载它们，缺少时直接 import 失败。
    """
    heavy_deps = [
        "xgboost",
        "torch", "torch.nn", "torch.optim", "torch.utils",
        "torch.utils.data",
        "vectorbt",
    ]
    import sys as _sys
    import types as _types

    class _StubClass:
        """可被任意继承 / 实例化 / 调用的通用占位类"""
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, name):
            return self

        @classmethod
        def __mro_entries__(cls, bases):
            return (cls,)

    class _StubModule(_types.ModuleType):
        def __getattr__(self, name):
            # 第一次访问某个属性时，动态挂一个可被继承的 StubClass
            obj = type(name, (_StubClass,), {})()
            setattr(self, name, obj)
            return obj

        def __call__(self, *args, **kwargs):
            return self

    for dep in heavy_deps:
        if dep not in _sys.modules:
            _sys.modules[dep] = _StubModule(dep)

_inject_missing_module_stubs()

try:
    from main import (
        _DATA_PROXY,
        _DATA_LAKE,
        config,
        fetch_index,
        fetch_hot_sectors,
        fetch_northbound_flow,
        extract_market_context,
        generate_macro_section,
        get_ma_trend,
        get_signals,
        NotificationGateway,
        C,
        Config,
        Signal,
    )
    log.info("✅ 已成功挂载 main.py 的核心能力")
except Exception as e:
    log.error(f"❌ 挂载 main.py 失败: {e}，将尝试以最小离线模式运行")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 1. ETF 轮动雷达 (独立模块)
# ---------------------------------------------------------------------------

# 核心 ETF 白名单 (宽基 / 行业 / 主题 / 跨境 / 债券 / 商品)
ETF_UNIVERSE = [
    # 宽基
    ("510300", "沪深300ETF"),
    ("510050", "上证50ETF"),
    ("510500", "中证500ETF"),
    ("512100", "中证1000ETF"),
    ("159915", "创业板ETF"),
    ("588000", "科创50ETF"),
    ("510900", "恒生ETF"),
    # 行业主题
    ("512170", "医疗ETF"),
    ("512010", "医药ETF"),
    ("159995", "芯片ETF"),
    ("512760", "半导体ETF"),
    ("515030", "新能源车ETF"),
    ("515050", "5G ETF"),
    ("512690", "酒ETF"),
    ("512880", "证券ETF"),
    ("512800", "银行ETF"),
    ("512660", "军工ETF"),
    ("510410", "消费ETF"),
    ("560610", "红利ETF"),
    ("512580", "环保ETF"),
    # 商品 / 跨境
    ("518880", "黄金ETF"),
    ("162411", "华宝油气"),
    ("513100", "纳指ETF"),
    ("513500", "标普500ETF"),
    ("513030", "德国30ETF"),
    ("513600", "恒生科技ETF"),
]


def fetch_etf_hist(code: str, days: int = 120) -> Optional[pd.DataFrame]:
    """获取 ETF 历史行情 (使用 DataProxy 的统一取数层)"""
    try:
        start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        end = datetime.now().strftime("%Y%m%d")
        # 走底层 _DATA_PROXY (可能是 LocalDataLake 或 DataProxy)
        if hasattr(_DATA_PROXY, 'get_hist'):
            return _DATA_PROXY.get_hist(code, start, end)
        return None
    except Exception as e:
        log.debug(f"  ⚠️ {code} 历史数据获取失败: {e}")
        return None


def rank_etf_on_momentum(hist: pd.DataFrame) -> Dict[str, float]:
    """对单一 ETF 计算多周期动量与趋势得分"""
    if hist is None or len(hist) < 20:
        return {}

    try:
        # 兼容不同列名
        close_col = C.H_CLOSE if C.H_CLOSE in hist.columns else 'close'
        high_col = C.H_HIGH if C.H_HIGH in hist.columns else 'high'
        low_col = C.H_LOW if C.H_LOW in hist.columns else 'low'
        vol_col = C.H_VOL if C.H_VOL in hist.columns else 'volume'

        close = pd.to_numeric(hist[close_col], errors='coerce').dropna()
        if len(close) < 20:
            return {}

        now_price = float(close.iloc[-1])
        ma5 = close.rolling(5).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20
        ma120 = close.rolling(120).mean().iloc[-1] if len(close) >= 120 else ma60

        # 涨跌幅
        r5 = (now_price / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0.0
        r20 = (now_price / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0.0
        r60 = (now_price / close.iloc[-61] - 1) * 100 if len(close) >= 61 else 0.0
        r120 = (now_price / close.iloc[-121] - 1) * 100 if len(close) >= 121 else 0.0

        # 趋势强弱：价格相对 MA 位置
        trend_score = 0.0
        if now_price > ma5 > ma20 > ma60:
            trend_score += 30.0  # 多头排列
        elif now_price < ma5 < ma20 < ma60:
            trend_score -= 30.0  # 空头排列
        else:
            trend_score += 10.0  # 震荡

        # 距年线位置 (反映长期趋势)
        if ma120 > 0:
            rel_120 = (now_price - ma120) / ma120 * 100
            trend_score += rel_120 * 0.8

        # 动量得分 = 短中长期动量加权
        mom_score = r5 * 0.2 + r20 * 0.3 + r60 * 0.3 + r120 * 0.2

        # 波动率惩罚 (避免追高暴涨品种)
        vol_20 = close.pct_change().tail(20).std() * 100
        vol_penalty = max(0.0, (vol_20 - 3.0)) * 2.0

        total_score = trend_score + mom_score - vol_penalty

        return {
            "price": now_price,
            "ma5": round(ma5, 3),
            "ma20": round(ma20, 3),
            "ma60": round(ma60, 3),
            "r5": round(r5, 2),
            "r20": round(r20, 2),
            "r60": round(r60, 2),
            "r120": round(r120, 2),
            "trend": round(trend_score, 2),
            "momentum": round(mom_score, 2),
            "volatility": round(vol_20, 2),
            "score": round(total_score, 2),
        }
    except Exception as e:
        log.debug(f"  ⚠️ ETF 评分计算异常: {e}")
        return {}


def generate_etf_rotation_report() -> str:
    """生成 ETF 轮动雷达报告区块"""
    log.info("📡 [3/5] 正在扫描 ETF 轮动雷达 ...")

    results = []
    for code, name in ETF_UNIVERSE:
        try:
            hist = fetch_etf_hist(code, days=140)
            metrics = rank_etf_on_momentum(hist)
            if not metrics:
                continue
            metrics["code"] = code
            metrics["name"] = name
            results.append(metrics)
            time.sleep(0.15)  # 温柔请求
        except Exception as e:
            log.debug(f"  ⚠️ {code}({name}) 处理失败: {e}")
            continue

    if not results:
        return (
            f"### 📡 ETF 轮动雷达\n"
            f"> ⚠️ 所有 ETF 数据源均不可用 (可能网络受限或接口熔断)\n"
        )

    df = pd.DataFrame(results)

    # ---- TOP5 / BOTTOM5 ----
    top5 = df.nlargest(5, "score")
    bottom5 = df.nsmallest(5, "score")

    def _format_row(row):
        code = row["code"]
        prefix = '1' if code.startswith('5') or code.startswith('6') else '0'
        tdx_link = f"https://quote.eastmoney.com/unify/r/{prefix}.{code}"
        return (
            f"- **{row['name']}** (`{code}`) ¥{row['price']:.2f} | "
            f"5D `{row['r5']:+.1f}%` / 20D `{row['r20']:+.1f}%` / 60D `{row['r60']:+.1f}%` "
            f"| 综合得分 **`{row['score']:.1f}`** "
            f"[📈]({tdx_link})"
        )

    # ---- 宽基强弱对比 ----
    broad_bases = ["510300", "510050", "510500", "512100", "159915", "588000", "510900"]
    bb_df = df[df["code"].isin(broad_bases)].copy()

    broad_lines = []
    if not bb_df.empty:
        bb_sorted = bb_df.sort_values("score", ascending=False)
        for _, row in bb_sorted.iterrows():
            icon = "🟢" if row["r5"] > 0 else "🔴" if row["r5"] < -1 else "🟡"
            broad_lines.append(
                f"  {icon} **{row['name']}** (`{row['code']}`) | "
                f"5D `{row['r5']:+.1f}%` / 20D `{row['r20']:+.1f}%` / 60D `{row['r60']:+.1f}%` "
                f"| 得分 **`{row['score']:.1f}`**"
            )

    # ---- 主线板块强度 ----
    sector_codes = ["512170", "512010", "159995", "512760", "515030", "515050",
                    "512690", "512880", "512800", "512660", "510410", "560610",
                    "512580", "518880", "513100", "513600"]
    sec_df = df[df["code"].isin(sector_codes)].copy()

    sector_lines = []
    if not sec_df.empty:
        sec_sorted = sec_df.sort_values("score", ascending=False)
        for _, row in sec_sorted.head(10).iterrows():
            icon = "🔥" if row["r5"] > 2 else "🌱" if row["r5"] < -2 else "➰"
            sector_lines.append(
                f"  {icon} **{row['name']}** (`{row['code']}`) | "
                f"5D `{row['r5']:+.1f}%` / 20D `{row['r20']:+.1f}%` | 得分 **`{row['score']:.1f}`**"
            )

    # ---- 顶部 / 底部榜 ----
    top_lines = [_format_row(row) for _, row in top5.iterrows()]
    bottom_lines = [_format_row(row) for _, row in bottom5.iterrows()]

    report = (
        f"### 📡 ETF 轮动雷达\n"
        f"> 扫描时间: {datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M') if TZ_BJS else datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"> 覆盖品种: 宽基 × 7 | 行业/主题 × 16 | 商品/跨境 × 5\n\n"
    )

    if broad_lines:
        report += f"**🗂 宽基强弱**\n" + "\n".join(broad_lines) + "\n\n"

    if sector_lines:
        report += f"**🎯 行业/主题强度 Top10**\n" + "\n".join(sector_lines) + "\n\n"

    report += f"**🚀 强势榜 Top5 (追涨需谨慎)**\n" + "\n".join(top_lines) + "\n\n"
    report += f"**🛟 弱势榜 Top5 (抄底有风险)**\n" + "\n".join(bottom_lines) + "\n\n"

    # 策略提示
    try:
        if not bb_df.empty:
            best_bb = bb_df.nlargest(1, "score").iloc[0]
            worst_bb = bb_df.nsmallest(1, "score").iloc[0]
            report += (
                f"> 💡 **轮动建议**: 当前 **{best_bb['name']}** 综合动量最强 (`{best_bb['score']:.1f}`)，"
                f"**{worst_bb['name']}** 最弱 (`{worst_bb['score']:.1f}`)。"
                f"若强势品种距 20 日均线偏离 > 5%，建议等待回踩均线再介入，避免单边追高。\n"
            )
    except Exception:
        pass

    return report


# ---------------------------------------------------------------------------
# 2. 行业热点主线深度分析 (基于 main.fetch_hot_sectors + 成分股强度)
# ---------------------------------------------------------------------------

def generate_hot_sector_deep_report() -> str:
    """深度行业热点报告：板块榜 + 板块成分股涨停/高涨幅统计"""
    log.info("🔥 [4/5] 正在解析行业热点主线 ...")

    try:
        hot_map = fetch_hot_sectors()  # {code: sector_name}
    except Exception as e:
        log.warning(f"  ⚠️ 热点板块抓取失败: {e}")
        hot_map = {}

    if not hot_map:
        return (
            f"### 🔥 行业热点主线\n"
            f"> ⚠️ 板块数据源不可用 (可能被云端 WAF 限流)，请稍后再试。\n"
        )

    # 反转: 板块 -> 成分股代码列表
    sector_members: Dict[str, List[str]] = {}
    for code, sector in hot_map.items():
        sector_members.setdefault(sector, []).append(code)

    # 尝试拉取今日成分股截面涨跌幅 (若 spot 已在外部缓存则走缓存)
    try:
        spot_df = _DATA_PROXY.get_spot()  # 复用 main 全局 spot
        spot_code_col = C.S_CODE
        spot_pct_col = C.S_PCT
        spot_name_col = C.S_NAME
        spot_price_col = C.S_PRICE

        spot_ok = (spot_df is not None and not spot_df.empty and
                   spot_code_col in spot_df.columns and
                   spot_pct_col in spot_df.columns)
    except Exception:
        spot_ok = False

    sector_stats = []
    for sector, codes in sector_members.items():
        members = codes
        if not members:
            continue

        pcts = []
        leaders = []
        if spot_ok:
            try:
                sub = spot_df[spot_df[spot_code_col].astype(str).str.zfill(6).isin(
                    [str(c).zfill(6) for c in members])].copy()
                sub[spot_pct_col] = pd.to_numeric(sub[spot_pct_col], errors='coerce')
                if not sub.empty:
                    pcts = sub[spot_pct_col].dropna().tolist()
                    top3 = sub.nlargest(3, spot_pct_col)
                    for _, r in top3.iterrows():
                        try:
                            nm = r.get(spot_name_col, '')
                            pc = float(r.get(spot_pct_col, 0))
                            pr = float(r.get(spot_price_col, 0))
                            leaders.append(f"{nm} `{pc:+.1f}%`(¥{pr:.2f})")
                        except Exception:
                            pass
            except Exception:
                pass

        if pcts:
            avg_pct = float(np.mean(pcts))
            median_pct = float(np.median(pcts))
            up_ratio = sum(1 for p in pcts if p > 0) / len(pcts)
            big_up = sum(1 for p in pcts if p >= 5.0)
            limit_up = sum(1 for p in pcts if p >= 9.0)
        else:
            avg_pct = 0.0
            median_pct = 0.0
            up_ratio = 0.0
            big_up = 0
            limit_up = 0

        sector_stats.append({
            "sector": sector,
            "count": len(members),
            "avg_pct": avg_pct,
            "median_pct": median_pct,
            "up_ratio": up_ratio,
            "big_up": big_up,
            "limit_up": limit_up,
            "leaders": leaders,
        })

    # 按平均涨幅排名
    sector_stats.sort(key=lambda x: x["avg_pct"], reverse=True)

    report_lines = []
    for s in sector_stats[:8]:
        icon = "🔥" if s["avg_pct"] >= 3.0 else "🟢" if s["avg_pct"] >= 1.0 else "🟡" if s["avg_pct"] >= -1.0 else "🔴"
        leader_str = " / ".join(s["leaders"][:3]) if s["leaders"] else "(成分股数据暂缺)"
        report_lines.append(
            f"- {icon} **{s['sector']}** (覆盖约 {s['count']} 只成分股) — "
            f"板块平均 `{s['avg_pct']:+.2f}%` / 中位 `{s['median_pct']:+.2f}%` / "
            f"红盘占比 `{s['up_ratio']*100:.0f}%` / ≥5% `{s['big_up']}` 家 / 涨停 `{s['limit_up']}` 家\n"
            f"  > **龙一/龙二**: {leader_str}"
        )

    report = (
        f"### 🔥 行业热点主线深度榜\n"
        f"> 数据源: 东方财富 / 同花顺 实时板块榜 + 成分股截面强度回灌\n\n"
    )
    if report_lines:
        report += "\n\n".join(report_lines) + "\n\n"
    else:
        report += "> ⚠️ 无法解析板块成分股详细数据，请检查网络。\n\n"

    # 策略提示
    try:
        strong = [s for s in sector_stats if s["avg_pct"] >= 2.0 and s["up_ratio"] >= 0.6]
        if strong:
            names = "、".join([s["sector"] for s in strong[:3]])
            report += f"> 💡 **热点策略提示**: 今日 **{names}** 板块资金聚合度高，\n"
            report += f"> 建议关注板块内龙一/龙二标的 (连续涨停/高换手+高量比) 作为情绪锚定；\n"
            report += f"> 若板块涨幅 > 5% 家数少于 3 家，则可能为概念脉冲行情，不宜追高。\n"
    except Exception:
        pass

    return report


# ---------------------------------------------------------------------------
# 3. 市场诊断增强版 (整合 extract_market_context + 额外估值信息)
# ---------------------------------------------------------------------------

def generate_market_diagnosis() -> Tuple[str, pd.DataFrame, bool]:
    """生成完整市场诊断区块，复用 main.extract_market_context"""
    log.info("📊 [1/5] 正在做 A 股市场深度诊断 ...")

    try:
        spot_df = _DATA_PROXY.get_spot()
    except Exception as e:
        log.warning(f"  ⚠️ 实时行情获取失败: {e}")
        spot_df = pd.DataFrame()

    c_conf = Config()
    if spot_df is None or spot_df.empty or len(spot_df) < 500:
        return (
            f"### 📊 A股深度诊断\n"
            f"> ⚠️ 横截面数据严重不足 (API 接口限流或断网)，无法做有效诊断。\n",
            pd.DataFrame(),
            False
        )

    df_clean, market_ok, market_msg, idx_ret, overheated, regime, vol_surge = extract_market_context(
        spot_df, c_conf
    )

    # 附加: 板块+涨停板集中度 (若有)
    extra = ""
    try:
        if C.S_PCT in spot_df.columns:
            pct_series = pd.to_numeric(spot_df[C.S_PCT], errors='coerce')
            zt_count = int((pct_series >= 9.0).sum())
            dt_count = int((pct_series <= -9.0).sum())
            up_count = int((pct_series > 0).sum())
            down_count = int((pct_series < 0).sum())
            total = len(pct_series.dropna())

            if total > 0:
                extra += (
                    f"- **涨停/跌停比**: `{zt_count} / {dt_count}` "
                    f"(红盘 `{up_count}` / 绿盘 `{down_count}`)\n"
                )
                if zt_count > 0 and dt_count > 0:
                    ratio = zt_count / max(dt_count, 1)
                    if ratio > 5:
                        extra += f"- **情绪温度计**: 🔥🔥🔥 极端多头 (涨停/跌停比 {ratio:.1f})\n"
                    elif ratio < 0.3:
                        extra += f"- **情绪温度计**: 🧊🧊🧊 恐慌冰点 (涨停/跌停比 {ratio:.1f})\n"
                    else:
                        extra += f"- **情绪温度计**: ⚖️ 多空平衡 (涨停/跌停比 {ratio:.1f})\n"
    except Exception:
        pass

    if extra:
        # 在 "市场广度" 后追加信息 —— 简单直接替换
        if "市场广度" in market_msg:
            market_msg = market_msg.replace("- **市场广度**", f"{extra}- **市场广度**", 1)
        else:
            market_msg += "\n" + extra

    return market_msg, df_clean, market_ok


# ---------------------------------------------------------------------------
# 4. 个股信号 (复用 main.get_signals)
# ---------------------------------------------------------------------------

def generate_stock_signal_block() -> Tuple[str, int, int]:
    """生成个股信号区块，复用 main.get_signals 的全周期共振精选"""
    log.info("🎯 [2/5] 正在扫描个股量化信号 ...")

    try:
        sigs, watch, pushed, pool_size, m_msg, total_mkt = get_signals()

        total_signals = 0
        if isinstance(sigs, dict):
            total_signals = sum(len(v) for v in sigs.values())
        else:
            total_signals = len(sigs)

        # 简易格式化为 markdown
        lines = []
        if isinstance(sigs, dict):
            for cat, arr in sigs.items():
                if not arr:
                    continue
                lines.append(f"**🏷 {cat} ({len(arr)} 只)**")
                for s in arr:
                    warn = " ⚠️创业板" if str(getattr(s, 'code', '')).startswith('300') else ""
                    try:
                        code = str(s.code).zfill(6)
                        prefix = '1' if code.startswith('6') else '0'
                        link = f"https://quote.eastmoney.com/unify/r/{prefix}.{code}"
                        lines.append(
                            f"- **{s.name}** (`{code}`) ¥{s.price} {s.pct_chg} "
                            f"| 评分 `{s.score}` {s.level}{warn} [📈]({link})"
                        )
                        # 附上核心逻辑 / 止损点 (仅前 3 只)
                        if len(lines) < 15 and getattr(s, 'reasons', ''):
                            # 简化: 只取前 80 字
                            r = str(s.reasons).strip()
                            if len(r) > 120:
                                r = r[:120] + "..."
                            if r:
                                lines.append(f"  > {r}")
                            if hasattr(s, 'stop_loss') and s.stop_loss:
                                lines.append(f"  > 🛟 止损参考: ¥{s.stop_loss}")
                    except Exception:
                        lines.append(f"- **{getattr(s, 'name', '?')}** (`{getattr(s, 'code', '?')}`)")
                lines.append("")
        else:
            for s in sigs:
                lines.append(f"- **{s.name}** (`{s.code}`) ¥{s.price} 评分 {s.score}")

        if not lines:
            lines.append("> ✅ 今日未发现满足安全边际的个股信号，建议空仓防守。")

        block = (
            f"### 🎯 个股量化信号 (全周期共振精选)\n"
            f"> 全市场白名单 `{total_mkt}` 只 | 异动提取 `{pool_size}` 只 | 过线 **{total_signals}** 只\n\n"
        )
        block += "\n".join(lines) + "\n"

        # 候补观察池
        if watch:
            watch_lines = []
            for item in watch[:5]:
                try:
                    name, code, score, price = item[:4]
                    watch_lines.append(f"- `{code}` **{name}** (¥{price}) 得分 `{score}`")
                except Exception:
                    pass
            if watch_lines:
                block += "\n**👁 候补观察池**\n" + "\n".join(watch_lines) + "\n"

        return block, total_signals, total_mkt
    except Exception as e:
        log.error(f"  ⚠️ 个股信号扫描失败: {e}")
        return (
            f"### 🎯 个股量化信号\n"
            f"> ⚠️ 信号扫描异常: {e}\n"
        ), 0, 0


# ---------------------------------------------------------------------------
# 5. 主入口：拼装完整简报并推送
# ---------------------------------------------------------------------------

def build_full_briefing() -> str:
    """拼装 5 大模块为一份完整简报"""
    now = datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M') if TZ_BJS else datetime.now().strftime('%Y-%m-%d %H:%M')

    header = (
        f"## 📮 AI量化 · 每日投研简报\n"
        f"> **生成时间**: {now} | 运行模式 `{config.RUN_MODE}`\n\n"
        f"> 🧭 本简报整合 **隔夜外围 → A股诊断 → 个股信号 → ETF轮动 → 行业热点** 五大情报源，\n"
        f"> 帮助你在开盘前 15 分钟对当日盘面状态形成整体判断。\n\n"
        f"---\n\n"
    )

    # [1] 隔夜外围宏观
    log.info("🌍 [0/5] 正在获取隔夜外围宏观数据 ...")
    macro_block = generate_macro_section()

    # [2] 市场诊断
    market_block, _, market_ok = generate_market_diagnosis()

    # [3] 个股信号 (若 market_ok=False 仍会生成，只是会提示风险)
    signal_block, sig_count, total_mkt = generate_stock_signal_block()

    # [4] ETF 轮动
    etf_block = generate_etf_rotation_report()

    # [5] 行业热点
    sector_block = generate_hot_sector_deep_report()

    # 底部免责声明
    footer = (
        f"\n---\n\n"
        f"### 📌 操作纪律与免责\n"
        f"- 🛡 **止损纪律**: 单票破位 20 日均线 / 买入理由消失 / 亏损 ≥ 8%，任一触发无条件离场。\n"
        f"- 📊 **仓位管理**: 单票 ≤ 15% 总仓；单板块 ≤ 30% 总仓；保留 ≥ 20% 现金以应对极端行情。\n"
        f"- ⚠️ **免责声明**: 本简报由量化算法自动生成，仅做盘面客观数据呈现，**不构成任何投资建议**。\n"
        f"  历史信号/动量排名不代表未来表现，据此操作风险自担。\n\n"
        f"> 📅 每日 08:45 (早报) / 15:15 (晚报) 自动推送。仅供内部研究使用。\n"
    )

    full = header + macro_block + "\n\n" + market_block + "\n\n---\n\n" + \
           signal_block + "\n\n---\n\n" + etf_block + "\n\n---\n\n" + sector_block + footer

    return full


def main() -> None:
    log.info("=" * 60)
    log.info("🚀 [每日投研简报引擎] 启动")
    log.info("=" * 60)

    try:
        content = build_full_briefing()

        # 保存本地副本以便人工审阅 (避免依赖推送通道)
        try:
            stamp = datetime.now(TZ_BJS).strftime('%Y%m%d') if TZ_BJS else datetime.now().strftime('%Y%m%d')
            local_path = f"daily_briefing_{stamp}.md"
            with open(local_path, "w", encoding="utf-8") as fp:
                fp.write(content)
            log.info(f"📄 简报已本地存档: {local_path}")
        except Exception as e:
            log.warning(f"  ⚠️ 本地存档失败: {e}")

        # 推送
        if config.DINGTALK_WEBHOOK or config.FEISHU_WEBHOOK:
            log.info("📤 [5/5] 正在推送至钉钉/飞书 ...")
            title = "📮 AI量化 · 每日投研简报"
            NotificationGateway.send(title, content)
            log.info("✅ 推送完成")
        else:
            log.warning("⚠️ 未配置 DINGTALK_WEBHOOK / FEISHU_WEBHOOK，跳过推送")
            # 控制台输出前 1500 字方便调参
            preview = content[:1500] + "\n...(截断，详见本地 .md 文件)" if len(content) > 1500 else content
            print("\n" + "=" * 50)
            print(preview)
            print("=" * 50 + "\n")

    except Exception as e:
        log.critical(f"💥 简报生成崩溃: {e}", exc_info=True)
        try:
            error_msg = (
                f"🚨 **AI量化简报引擎崩溃告警**\n\n"
                f"- **时间**: {datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M') if TZ_BJS else datetime.now()}\n"
                f"- **异常信息**: {str(e)[:300]}\n"
                f"- **提示**: 可能是实时行情接口限流或网络波动，建议稍后重试。"
            )
            NotificationGateway.send("🚨 每日简报崩溃告警", error_msg, template="red")
        except Exception:
            pass
    finally:
        try:
            _DATA_PROXY.cleanup()
        except Exception:
            pass
        log.info("🏁 任务结束")


if __name__ == "__main__":
    main()
