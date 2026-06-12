#!/usr/bin/env python3
"""
每日投研简报生成器 (Daily Investment Briefing Generator) - 离线版
使用本地缓存数据生成投研简报
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
import pytz

TZ_BJS = pytz.timezone('Asia/Shanghai')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

# 数据缓存目录
HIST_CACHE_DIR = '/workspace/hist_cache'
QUANTBOT_DATA_DIR = '/workspace/.quantbot_data'


def load_local_spot():
    """从本地缓存加载实时行情数据"""
    spot_path = os.path.join(HIST_CACHE_DIR, 'spot.parquet')
    if os.path.exists(spot_path):
        df = pd.read_parquet(spot_path)
        log.info(f"📦 已加载本地spot缓存: {len(df)} 条记录")
        return df
    return None


def load_local_index():
    """从本地缓存加载指数数据"""
    index_path = os.path.join(HIST_CACHE_DIR, 'index_sh000001.parquet')
    if os.path.exists(index_path):
        df = pd.read_parquet(index_path)
        log.info(f"📦 已加载本地指数缓存: {len(df)} 条记录")
        return df
    return None


def load_local_hot_sectors():
    """从本地缓存加载热点板块数据"""
    sectors_path = os.path.join(HIST_CACHE_DIR, 'hot_sectors.pkl')
    if os.path.exists(sectors_path):
        with open(sectors_path, 'rb') as f:
            data = pickle.load(f)
        log.info(f"📦 已加载本地热点板块缓存: {len(data)} 条记录")
        return data
    return {}


def load_local_northbound():
    """从本地缓存加载北向资金数据"""
    northbound_path = os.path.join(HIST_CACHE_DIR, 'northbound.pkl')
    if os.path.exists(northbound_path):
        with open(northbound_path, 'rb') as f:
            data = pickle.load(f)
        log.info(f"📦 已加载本地北向资金缓存")
        return data
    return (0.0, "")


def load_local_core_pool():
    """从本地缓存加载核心股票池"""
    pool_path = os.path.join(HIST_CACHE_DIR, 'core_pool.pkl')
    if os.path.exists(pool_path):
        with open(pool_path, 'rb') as f:
            data = pickle.load(f)
        log.info(f"📦 已加载本地核心池缓存: {len(data)} 只股票")
        return data
    return set()


def load_ashare_daily():
    """加载A股日线数据"""
    daily_path = os.path.join(QUANTBOT_DATA_DIR, 'ashare_daily.parquet')
    if os.path.exists(daily_path):
        df = pd.read_parquet(daily_path)
        log.info(f"📦 已加载A股日线数据: {len(df)} 条记录")
        return df
    return None


def load_macro_daily():
    """加载宏观日线数据"""
    macro_path = os.path.join(QUANTBOT_DATA_DIR, 'macro_daily.parquet')
    if os.path.exists(macro_path):
        df = pd.read_parquet(macro_path)
        log.info(f"📦 已加载宏观日线数据: {len(df)} 条记录")
        return df
    return None


def load_model_predictions():
    """加载模型预测结果"""
    preds_path = os.path.join(QUANTBOT_DATA_DIR, 'oos_preds.csv')
    if os.path.exists(preds_path):
        df = pd.read_csv(preds_path)
        log.info(f"📦 已加载模型预测结果: {len(df)} 条记录")
        return df
    return None


def load_regime_results():
    """加载市场状态结果"""
    regime_path = os.path.join(QUANTBOT_DATA_DIR, 'regime_results.json')
    if os.path.exists(regime_path):
        with open(regime_path, 'r') as f:
            data = json.load(f)
        log.info(f"📦 已加载市场状态结果")
        return data
    return {}


def load_advisory_tracker():
    """加载辅助跟踪器"""
    tracker_path = '/workspace/advisory_tracker.json'
    if os.path.exists(tracker_path):
        with open(tracker_path, 'r') as f:
            data = json.load(f)
        log.info(f"📦 已加载辅助跟踪器: {len(data)} 条记录")
        return data
    return {}


def generate_market_analysis(spot_df, index_df):
    """生成市场分析部分"""
    if spot_df is None or index_df is None:
        return "### 📊 A股深度诊断\n⚠️ 本地缓存数据不足\n"
    
    lines = ["### 📊 A股深度诊断\n"]
    
    try:
        # 指数分析
        cl = index_df['close']
        ma5 = cl.rolling(5).mean().iloc[-1] if len(cl) >= 5 else cl.iloc[-1]
        ma20 = cl.rolling(20).mean().iloc[-1] if len(cl) >= 20 else cl.iloc[-1]
        ma60 = cl.rolling(60).mean().iloc[-1] if len(cl) >= 60 else cl.iloc[-1]
        
        pct = (cl.iloc[-1] - cl.iloc[-2]) / cl.iloc[-2] * 100 if len(cl) >= 2 else 0.0
        
        # 市场广度
        pct_col = '涨跌幅' if '涨跌幅' in spot_df.columns else 'pct_chg'
        up_count = (spot_df[pct_col] > 0).sum() if pct_col in spot_df.columns else 0
        down_count = (spot_df[pct_col] < 0).sum() if pct_col in spot_df.columns else 0
        zt_count = (spot_df[pct_col] >= 9.0).sum() if pct_col in spot_df.columns else 0
        dt_count = (spot_df[pct_col] <= -9.0).sum() if pct_col in spot_df.columns else 0
        
        # 趋势判断
        if cl.iloc[-1] > ma20 and ma5 > ma20:
            trend = "🔥 **强势多头 (BULL)**"
            advice = "仓位 60%-80%。赚钱效应佳，跟随主线积极做多。"
        elif cl.iloc[-1] < ma20 and ma5 < ma20:
            trend = "🐻 **弱势空头 (BEAR)**"
            advice = "仓位 20%-30%。均线压制，控制回撤。"
        else:
            trend = "⚖️ **震荡均衡 (NEUTRAL)**"
            advice = "仓位 40%-60%。重个股轻大盘。"
        
        lines.append(f"- **上证指数**：`{cl.iloc[-1]:.2f}` (涨跌 **{pct:+.2f}%**)\n")
        lines.append(f"- **综合判定**：{trend}\n")
        lines.append(f"- **市场广度**：红盘 `{up_count}` 家 / 绿盘 `{down_count}` 家 (涨停 `{zt_count}` / 跌停 `{dt_count}`)\n")
        lines.append(f"\n**💡 仓位建议**：{advice}\n")
        
    except Exception as e:
        log.warning(f"市场分析生成失败: {e}")
        lines.append(f"⚠️ 分析生成失败: {e}\n")
    
    return ''.join(lines)


def generate_etf_rotation_section(index_df):
    """生成ETF轮动策略分析"""
    if index_df is None:
        return "### 🔄 ETF轮动策略\n⚠️ 指数数据不足\n"
    
    lines = ["### 🔄 ETF轮动策略分析\n"]
    
    try:
        cl = index_df['close']
        ma5 = cl.rolling(5).mean().iloc[-1] if len(cl) >= 5 else cl.iloc[-1]
        ma20 = cl.rolling(20).mean().iloc[-1] if len(cl) >= 20 else cl.iloc[-1]
        
        pct_5d = (cl.iloc[-1] / cl.iloc[-6] - 1) * 100 if len(cl) >= 6 else 0
        pct_20d = (cl.iloc[-1] / cl.iloc[-21] - 1) * 100 if len(cl) >= 21 else 0
        
        trend = "强势" if cl.iloc[-1] > ma5 and ma5 > ma20 else \
                "弱势" if cl.iloc[-1] < ma5 and ma5 < ma20 else "震荡"
        
        lines.append(f"| 指数 | 收盘价 | 5日涨幅 | 20日涨幅 | 趋势状态 |\n")
        lines.append(f"|------|--------|---------|----------|----------|\n")
        lines.append(f"| 上证指数 | {cl.iloc[-1]:.2f} | {pct_5d:+.2f}% | {pct_20d:+.2f}% | {trend} |\n")
        
        lines.append("\n**💡 轮动建议**\n")
        if trend == "强势":
            lines.append("- 🚀 当前处于强势状态，建议配置大盘ETF (如510300)\n")
        elif trend == "弱势":
            lines.append("- ⚠️ 当前处于弱势状态，建议减仓或观望\n")
        else:
            lines.append("- ⚖️ 当前处于震荡状态，建议均衡配置\n")
        
    except Exception as e:
        log.warning(f"ETF轮动分析生成失败: {e}")
        lines.append(f"⚠️ 分析生成失败: {e}\n")
    
    return ''.join(lines)


def generate_sector_hotspot_section(hot_sectors):
    """生成行业热点分析"""
    if not hot_sectors:
        return "### 🌋 行业热点追踪\n⚠️ 热点板块数据不足\n"
    
    lines = ["### 🌋 行业热点追踪\n"]
    
    try:
        from collections import Counter
        
        # 处理不同数据格式
        if isinstance(hot_sectors, dict):
            # 如果是字典，统计板块数量
            sector_counts = Counter(hot_sectors.values())
            top_sectors = sector_counts.most_common(5)
            
            lines.append("**今日领涨主线板块**\n\n")
            for sector, count in top_sectors:
                lines.append(f"- 🔥 **{sector}**：包含 {count} 只成分股\n")
            
            # 板块龙头股示例
            lines.append("\n**板块龙头股示例**\n")
            sample_stocks = {}
            for code, sector in hot_sectors.items():
                if sector not in sample_stocks:
                    sample_stocks[sector] = code
                if len(sample_stocks) >= 5:
                    break
            
            for sector, code in sample_stocks.items():
                lines.append(f"- `{code}` ({sector})\n")
        else:
            lines.append(f"- 数据格式: {type(hot_sectors).__name__}\n")
        
    except Exception as e:
        log.warning(f"行业热点分析生成失败: {e}")
        lines.append(f"⚠️ 分析生成失败: {e}\n")
    
    return ''.join(lines)


def generate_stock_signals_section(preds_df):
    """生成股票信号部分"""
    if preds_df is None or preds_df.empty:
        return "### 🎯 今日精选股票信号\n✅ 本地缓存无预测数据，建议空仓防守\n"
    
    lines = ["### 🎯 今日精选股票信号\n"]
    
    try:
        # 获取最新预测结果
        if 'date' in preds_df.columns:
            latest_date = preds_df['date'].max()
            latest_preds = preds_df[preds_df['date'] == latest_date]
        else:
            latest_preds = preds_df.tail(50)
        
        if latest_preds.empty:
            lines.append("✅ 无最新预测数据\n")
            return ''.join(lines)
        
        # 按预测分数排序 (使用 xgb_score_t10 或 xgb_score_t20)
        score_col = 'xgb_score_t10' if 'xgb_score_t10' in latest_preds.columns else 'xgb_score_t20'
        if score_col in latest_preds.columns:
            top_preds = latest_preds.nlargest(5, score_col)
            
            lines.append("**🔥 ML模型精选 (Top 5)**\n\n")
            
            for _, row in top_preds.iterrows():
                code = row.get('code', '未知')
                score_t10 = row.get('xgb_score_t10', 0)
                score_t20 = row.get('xgb_score_t20', 0)
                fwd_ret = row.get('fwd_ret_real', 0)
                lines.append(f"- `{code}` T+10得分: **{score_t10:.3f}** | T+20得分: **{score_t20:.3f}** | 实际收益: {fwd_ret*100:.2f}%\n")
        else:
            lines.append("✅ 预测数据格式不完整\n")
        
    except Exception as e:
        log.warning(f"股票信号生成失败: {e}")
        lines.append(f"⚠️ 信号生成失败: {e}\n")
    
    return ''.join(lines)


def generate_tracker_section(tracker):
    """生成往期信号跟踪部分"""
    if not tracker:
        return ""
    
    lines = ["### 📢 往期信号跟踪反馈\n"]
    
    try:
        today = datetime.now(TZ_BJS)
        for code, info in tracker.items():
            name = info.get('name', '未知')
            entry_date = info.get('entry_date', '')
            target = info.get('target', 0)
            stop = info.get('stop', 0)
            
            lines.append(f"- `{code}` ({name}): 入场日期 {entry_date}, 目标价 ¥{target}, 止损价 ¥{stop}\n")
        
    except Exception as e:
        log.warning(f"跟踪反馈生成失败: {e}")
    
    return ''.join(lines)


def generate_northbound_section(northbound_data):
    """生成北向资金分析"""
    flow, msg = northbound_data
    if msg:
        return f"### 🌊 北向资金流向\n{msg}\n"
    return f"### 🌊 北向资金流向\n- 今日北向资金净流入: **{flow:.0f}亿**\n"


def generate_daily_briefing():
    """生成完整的每日投研简报"""
    now = datetime.now(TZ_BJS)
    now_str = now.strftime('%Y-%m-%d %H:%M')
    
    log.info("🚀 每日投研简报生成器启动 (离线模式)...")
    
    # 加载本地缓存数据
    log.info("📦 加载本地缓存数据...")
    spot_df = load_local_spot()
    index_df = load_local_index()
    hot_sectors = load_local_hot_sectors()
    northbound_data = load_local_northbound()
    core_pool = load_local_core_pool()
    preds_df = load_model_predictions()
    regime_results = load_regime_results()
    tracker = load_advisory_tracker()
    
    # 1. 头部信息
    header = f"## 📊 AI量化每日投研简报\n> **生成时间**: {now_str}\n> **数据来源**: 本地缓存 (离线模式)\n\n"
    
    # 2. 市场分析
    log.info("📊 生成市场分析...")
    market_section = generate_market_analysis(spot_df, index_df) + "\n"
    
    # 3. 北向资金
    log.info("🌊 生成北向资金分析...")
    northbound_section = generate_northbound_section(northbound_data) + "\n"
    
    # 4. ETF轮动策略
    log.info("🔄 生成ETF轮动策略...")
    etf_section = generate_etf_rotation_section(index_df) + "\n"
    
    # 5. 行业热点
    log.info("🌋 生成行业热点...")
    sector_section = generate_sector_hotspot_section(hot_sectors) + "\n"
    
    # 6. 股票信号
    log.info("🎯 生成股票信号...")
    stock_section = generate_stock_signals_section(preds_df) + "\n"
    
    # 7. 往期跟踪
    log.info("📢 生成往期跟踪...")
    tracker_section = generate_tracker_section(tracker) + "\n"
    
    # 8. 市场状态
    log.info("📈 生成市场状态...")
    regime_section = ""
    if regime_results:
        regime_section = "### 📈 市场状态分析\n"
        for key, value in regime_results.items():
            regime_section += f"- **{key}**: {value}\n"
        regime_section += "\n"
    
    # 9. 核心池统计
    log.info("💎 生成核心池统计...")
    pool_section = ""
    if core_pool:
        pool_section = f"### 💎 核心股票池统计\n- 当前核心池规模: **{len(core_pool)}** 只股票\n\n"
    
    # 10. 尾部风险提示
    footer = (
        "---\n\n"
        "### ⚠️ 风险提示\n"
        "- 本简报基于本地缓存数据生成，仅供参考\n"
        "- 所有信号均基于历史数据回测，实盘需谨慎\n"
        "- 请严格执行止损纪律，破防守线立即离场\n"
        "- 市场有风险，投资需谨慎\n\n"
        f"> 🤖 *AI量化投研系统自动生成 (离线模式) | {now_str}*"
    )
    
    # 组合完整简报
    full_briefing = header + market_section + northbound_section + etf_section + sector_section + stock_section + tracker_section + regime_section + pool_section + footer
    
    log.info("✅ 每日投研简报生成完成")
    return full_briefing


def send_to_dingtalk(content: str):
    """发送简报到钉钉"""
    webhook = os.environ.get('DINGTALK_WEBHOOK', '')
    if not webhook:
        log.warning("⚠️ 未配置 DINGTALK_WEBHOOK，简报仅输出到本地")
        print("\n" + "="*60)
        print(content)
        print("="*60)
        return
    
    import requests
    
    # 钉钉消息格式
    payload = {
        'msgtype': 'markdown',
        'markdown': {
            'title': '📊 AI量化每日投研简报',
            'text': content
        }
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        res = requests.post(webhook, json=payload, headers=headers, timeout=10)
        res.raise_for_status()
        result = res.json()
        if result.get('errcode', 0) == 0:
            log.info("✅ 简报已成功发送到钉钉")
        else:
            log.error(f"❌ 钉钉推送失败: {result}")
    except Exception as e:
        log.error(f"❌ 发送简报到钉钉失败: {e}")
        print("\n" + "="*60)
        print(content)
        print("="*60)


def main():
    """主函数"""
    try:
        # 生成简报
        briefing = generate_daily_briefing()
        
        # 发送到钉钉
        send_to_dingtalk(briefing)
        
    except Exception as e:
        log.critical(f"❌ 每日投研简报生成失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()