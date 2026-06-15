#!/usr/bin/env python3
"""
每日投研简报生成器
生成包含市场分析、股票信号、ETF轮动、行业热点等内容的每日简报，并发送到钉钉通知
"""
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

import pandas as pd
import numpy as np

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

# 导入主模块的核心功能
from main import (
    TZ_BJS, Config, Cols, C, Signal,
    fetch_spot, fetch_index, fetch_core_pool, fetch_hot_sectors, fetch_northbound_flow,
    extract_market_context, NotificationGateway, AdvisoryTracker,
    load_pushed_state, save_pushed_state, is_recently_pushed,
    AShareTechnicals, process_stock, apply_scoring,
    format_money_risk_msg, generate_tranche_plan, generate_plan_b,
    calc_target_price,
    _DATA_PROXY, config
)

def generate_daily_briefing(use_cache: bool = True) -> Dict[str, Any]:
    """
    生成每日投研简报
    
    Returns:
        包含市场分析、股票信号、ETF轮动、行业热点等内容的字典
    """
    now = datetime.now(TZ_BJS)
    today_str = now.strftime('%Y-%m-%d')
    
    briefing = {
        'date': today_str,
        'time': now.strftime('%H:%M'),
        'market_analysis': {},
        'stock_signals': [],
        'etf_rotation': {},
        'hot_sectors': [],
        'northbound_flow': {},
        'advisory_tracking': [],
        'summary': ''
    }
    
    # 1. 获取市场数据
    log.info("📊 正在获取市场数据...")
    try:
        if use_cache:
            # 尝试从缓存获取
            df_raw = _load_spot_cache()
        else:
            df_raw = fetch_spot()
            
        if df_raw is None or df_raw.empty:
            log.warning("⚠️ 无法获取市场数据，使用缓存数据")
            df_raw = _load_spot_cache()
    except Exception as e:
        log.error(f"获取市场数据失败: {e}")
        df_raw = _load_spot_cache()
    
    if df_raw is None or df_raw.empty:
        briefing['summary'] = "⚠️ 今日无法获取市场数据，简报生成失败"
        return briefing
    
    # 2. 市场分析
    log.info("📈 正在进行市场分析...")
    c_conf = Config()
    try:
        df_clean, m_ok, m_msg, idx_ret, m_overheated, m_regime, vol_surge = extract_market_context(df_raw, c_conf)
        
        # 提取市场指标
        briefing['market_analysis'] = {
            'market_ok': m_ok,
            'market_regime': m_regime,
            'market_overheated': m_overheated,
            'vol_surge': vol_surge,
            'index_return': idx_ret,
            'market_message': m_msg,
            'total_stocks': len(df_raw),
            'filtered_stocks': len(df_clean) if not df_clean.empty else 0
        }
    except Exception as e:
        log.error(f"市场分析失败: {e}")
        briefing['market_analysis']['error'] = str(e)
    
    # 3. 北向资金
    log.info("💰 正在获取北向资金数据...")
    try:
        north_flow, north_msg = fetch_northbound_flow()
        briefing['northbound_flow'] = {
            'flow': north_flow,
            'message': north_msg
        }
    except Exception as e:
        log.warning(f"北向资金数据获取失败: {e}")
        briefing['northbound_flow'] = {'flow': 0.0, 'message': ''}
    
    # 4. 行业热点
    log.info("🔥 正在获取行业热点...")
    try:
        hot_map = fetch_hot_sectors()
        if hot_map:
            from collections import Counter
            sec_counts = Counter(hot_map.values())
            top_sectors = sec_counts.most_common(10)
            briefing['hot_sectors'] = [
                {'sector': s, 'count': c} for s, c in top_sectors
            ]
    except Exception as e:
        log.warning(f"行业热点获取失败: {e}")
    
    # 5. 往期信号跟踪
    log.info("📋 正在跟踪往期信号...")
    try:
        tracker_msgs = AdvisoryTracker.evaluate_and_clean(df_raw)
        briefing['advisory_tracking'] = tracker_msgs
    except Exception as e:
        log.warning(f"往期信号跟踪失败: {e}")
    
    # 6. ETF轮动分析
    log.info("🔄 正在进行ETF轮动分析...")
    try:
        briefing['etf_rotation'] = _analyze_etf_rotation(df_raw)
    except Exception as e:
        log.warning(f"ETF轮动分析失败: {e}")
    
    # 7. 股票信号生成
    log.info("🎯 正在生成股票信号...")
    try:
        if not df_clean.empty:
            signals = _generate_stock_signals(df_clean, m_ok, idx_ret, hot_map, m_regime, vol_surge, m_overheated)
            briefing['stock_signals'] = signals
    except Exception as e:
        log.error(f"股票信号生成失败: {e}")
    
    # 8. 生成摘要
    briefing['summary'] = _generate_summary(briefing)
    
    return briefing

def _load_spot_cache() -> Optional[pd.DataFrame]:
    """从缓存加载spot数据"""
    cache_path = os.path.join('hist_cache', 'spot.parquet')
    if os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            log.warning(f"读取spot缓存失败: {e}")
    return None

def _analyze_etf_rotation(df: pd.DataFrame) -> Dict[str, Any]:
    """分析ETF轮动情况"""
    etf_codes = df[df[Cols.S_CODE].astype(str).str.startswith(('51', '15', '588', '56'))]
    
    if etf_codes.empty:
        return {'message': '未找到ETF数据'}
    
    # 按涨跌幅排序
    etf_sorted = etf_codes.sort_values(by=Cols.S_PCT, ascending=False)
    
    top_etfs = []
    for _, row in etf_sorted.head(10).iterrows():
        top_etfs.append({
            'code': str(row[Cols.S_CODE]),
            'name': str(row[Cols.S_NAME]),
            'price': float(row[Cols.S_PRICE]),
            'pct_chg': float(row[Cols.S_PCT]),
            'volume': float(row.get(Cols.S_VOL, 0))
        })
    
    return {
        'top_etfs': top_etfs,
        'message': f"今日ETF市场表现：领涨 {top_etfs[0]['name']} ({top_etfs[0]['pct_chg']:+.2f}%)"
    }

def _generate_stock_signals(df: pd.DataFrame, market_ok: bool, index_ret: float, 
                            hot_sectors_map: dict, m_regime: str, vol_surge: bool,
                            m_overheated: bool) -> List[Dict[str, Any]]:
    """生成股票信号"""
    signals = []
    pushed = load_pushed_state()
    
    # 过滤已推送的股票
    recent_pushed_codes = {str(c) for c in df[Cols.S_CODE] if is_recently_pushed(str(c), pushed)}
    pool = df[~df[Cols.S_CODE].isin(recent_pushed_codes)].copy()
    
    # 基本面筛选
    pool = pool[
        (pool[Cols.S_PCT] >= -4.0) &
        (pool[Cols.S_PRICE] <= 500.0) &
        (pool[Cols.S_HIGH] > pool[Cols.S_LOW])
    ].copy()
    
    if pool.empty:
        return signals
    
    # 限制处理数量
    if len(pool) > 50:
        pool = pool.head(50)
    
    now = datetime.now(TZ_BJS)
    end_s, start_s = now.strftime('%Y%m%d'), (now - timedelta(days=450)).strftime('%Y%m%d')
    
    for _, row in pool.iterrows():
        try:
            # 尝试从缓存获取历史数据
            hist = _load_hist_cache(str(row[Cols.S_CODE]), end_s)
            if hist is None or len(hist) < 120:
                continue
            
            result = process_stock(row, hist, now, market_ok, index_ret, hot_sectors_map)
            if result:
                data, stop, risk = result
                
                # 计算得分
                win_stats = {}  # 简化版本不使用历史胜率统计
                score, level, reasons = apply_scoring(data, now, m_regime, vol_surge, win_stats)
                
                if score >= 70:  # 只保留B+级以上信号
                    target1 = calc_target_price(row[Cols.S_PRICE], stop, data)
                    
                    signal = {
                        'code': str(row[Cols.S_CODE]),
                        'name': str(row[Cols.S_NAME]),
                        'price': float(row[Cols.S_PRICE]),
                        'pct_chg': float(row[Cols.S_PCT]),
                        'score': score,
                        'level': level,
                        'stop_loss': stop,
                        'target1': target1,
                        'reasons': reasons,
                        'ma10': data.get('ma10_val', 0),
                        'atr': data.get('atr_val', 0),
                        'adx': data.get('adx', 0),
                        'in_hot_sector': data.get('in_hot_sector', False),
                        'hot_sector_name': data.get('hot_sector_name', '')
                    }
                    signals.append(signal)
        except Exception as e:
            log.debug(f"处理股票 {row[Cols.S_CODE]} 时出错: {e}")
            continue
    
    # 按得分排序
    signals.sort(key=lambda x: x['score'], reverse=True)
    return signals[:10]  # 只返回前10个

def _load_hist_cache(code: str, end_date: str) -> Optional[pd.DataFrame]:
    """从缓存加载历史数据"""
    cache_path = os.path.join('hist_cache', f"hist_{code}_{end_date}.parquet")
    if os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            log.debug(f"读取历史缓存失败 {code}: {e}")
    return None

def _generate_summary(briefing: Dict[str, Any]) -> str:
    """生成简报摘要"""
    market = briefing.get('market_analysis', {})
    
    summary_parts = []
    
    # 市场状态
    regime = market.get('market_regime', 'NEUTRAL')
    regime_map = {
        'BULL': '🔥 强势多头',
        'BEAR': '🐻 弱势空头',
        'PANIC': '🧊 恐慌冰点',
        'NEUTRAL': '⚖️ 震荡均衡'
    }
    summary_parts.append(f"市场状态: {regime_map.get(regime, regime)}")
    
    # 北向资金
    north = briefing.get('northbound_flow', {})
    flow = north.get('flow', 0.0)
    if flow > 30:
        summary_parts.append(f"北向资金: 大举流入 +{flow:.0f}亿")
    elif flow < -30:
        summary_parts.append(f"北向资金: 大幅流出 {flow:.0f}亿")
    else:
        summary_parts.append(f"北向资金: 温和 ({flow:+.0f}亿)")
    
    # 行业热点
    hot = briefing.get('hot_sectors', [])
    if hot:
        top_sector = hot[0]['sector']
        summary_parts.append(f"领涨板块: {top_sector}")
    
    # 信号数量
    signals = briefing.get('stock_signals', [])
    summary_parts.append(f"精选信号: {len(signals)}只")
    
    return " | ".join(summary_parts)

def format_briefing_message(briefing: Dict[str, Any]) -> str:
    """格式化简报为Markdown消息"""
    now = datetime.now(TZ_BJS)
    now_str = now.strftime('%Y-%m-%d %H:%M')
    
    # 标题
    msg = f"## 🤖 AI量化每日投研简报\n> **{now_str}**\n\n"
    
    # 市场分析
    market = briefing.get('market_analysis', {})
    market_msg = market.get('market_message', '')
    if market_msg:
        msg += market_msg + "\n\n---\n\n"
    
    # 北向资金
    north = briefing.get('northbound_flow', {})
    north_msg = north.get('message', '')
    if north_msg:
        msg += north_msg + "\n\n"
    
    # 行业热点
    hot = briefing.get('hot_sectors', [])
    if hot:
        msg += "### 🔥 今日行业热点\n"
        for item in hot[:5]:
            msg += f"- **{item['sector']}**: {item['count']}只成分股\n"
        msg += "\n"
    
    # ETF轮动
    etf = briefing.get('etf_rotation', {})
    top_etfs = etf.get('top_etfs', [])
    if top_etfs:
        msg += "### 🔄 ETF轮动观察\n"
        for etf_item in top_etfs[:5]:
            msg += f"- `{etf_item['code']}` **{etf_item['name']}**: ¥{etf_item['price']:.2f} ({etf_item['pct_chg']:+.2f}%)\n"
        msg += "\n"
    
    # 往期跟踪
    tracking = briefing.get('advisory_tracking', [])
    if tracking:
        msg += "### 📋 往期信号跟踪\n"
        for t_msg in tracking:
            msg += t_msg + "\n"
        msg += "\n"
    
    # 股票信号
    signals = briefing.get('stock_signals', [])
    if signals:
        msg += "### 🎯 今日精选信号\n\n"
        for sig in signals[:5]:
            warn_msg = "> ⚡ **【风险警示】** 该股为创业板(波动±20%)，请务必**缩减仓位**。\n\n" if str(sig['code']).startswith('300') else ""
            
            sina_market = 'sh' if str(sig['code']).startswith('6') else 'sz'
            kline_url = f"http://image.sinajs.cn/newchart/weekly/n/{sina_market}{sig['code']}.gif"
            
            msg += (
                f"#### 🎯 {sig['name']} (`{sig['code']}`)\n"
                f"{warn_msg}"
                f"- **综合评级**: `{sig['score']}` 分\n"
                f"- **今日收盘**: `¥{sig['price']}` ({sig['pct_chg']:+.2f}%) [📈 周K图]({kline_url})\n"
                f"- **防守线**: `¥{sig['stop_loss']}` | **目标价**: `¥{sig['target1']}`\n\n"
                f"**💡 核心逻辑**\n{sig['reasons']}\n\n"
                f"---\n\n"
            )
    else:
        msg += "### 🎯 今日精选信号\n✅ 今日未发现 B+ 级以上核心机会，建议**空仓防守**。\n\n"
    
    # 摘要
    msg += f"### 📊 简报摘要\n{briefing.get('summary', '')}\n"
    
    return msg

def send_briefing_to_dingtalk(briefing: Dict[str, Any], webhook: Optional[str] = None) -> bool:
    """发送简报到钉钉"""
    msg = format_briefing_message(briefing)
    
    # 使用配置的webhook或传入的webhook
    if webhook:
        # 直接发送到指定webhook
        import requests
        headers = {"Content-Type": "application/json"}
        sec_keyword = config.NOTIFY_SEC_KEYWORD
        title = f"{sec_keyword} | 每日投研简报"
        
        payload = {
            'msgtype': 'markdown',
            'markdown': {
                'title': title,
                'text': msg
            }
        }
        
        try:
            res = requests.post(webhook, json=payload, headers=headers, timeout=10)
            res.raise_for_status()
            res_dict = res.json()
            if res_dict.get('errcode', 0) == 0:
                log.info("✅ 钉钉推送成功")
                return True
            else:
                log.error(f"❌ 钉钉推送失败: {res_dict}")
                return False
        except Exception as e:
            log.error(f"❌ 钉钉推送异常: {e}")
            return False
    else:
        # 使用NotificationGateway
        NotificationGateway.send('🤖 AI量化每日投研简报', msg)
        return True

def main():
    """主函数"""
    log.info("=" * 50)
    log.info("🚀 每日投研简报生成器启动")
    log.info("=" * 50)
    
    # 检查是否配置了钉钉webhook
    dingtalk_webhook = os.environ.get('DINGTALK_WEBHOOK', '')
    
    # 生成简报
    use_cache = os.environ.get('DATA_CACHE_MODE', 'online') == 'offline'
    briefing = generate_daily_briefing(use_cache=use_cache)
    
    # 打印摘要
    log.info(f"📊 简报摘要: {briefing.get('summary', '')}")
    
    # 格式化消息并打印
    msg = format_briefing_message(briefing)
    print("\n" + "=" * 50)
    print("每日投研简报内容:")
    print("=" * 50)
    print(msg)
    print("=" * 50 + "\n")
    
    # 发送到钉钉
    if dingtalk_webhook:
        log.info("📤 正在发送到钉钉...")
        success = send_briefing_to_dingtalk(briefing, dingtalk_webhook)
        if success:
            log.info("✅ 钉钉通知发送成功")
        else:
            log.error("❌ 钉钉通知发送失败")
    else:
        log.warning("⚠️ 未配置 DINGTALK_WEBHOOK，通知已跳过")
    
    # 清理
    _DATA_PROXY.cleanup()
    
    return briefing

if __name__ == '__main__':
    try:
        briefing = main()
    except Exception as e:
        log.critical(f"系统崩溃: {e}", exc_info=True)
        sys.exit(1)