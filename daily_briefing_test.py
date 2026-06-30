#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日投研简报测试脚本
模拟生成每日投研简报的核心内容，包括：
- 市场分析
- 股票信号
- ETF轮动
- 行业热点
"""

import os
import sys
from datetime import datetime
import pytz

# 绕过代理
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
os.environ['NO_PROXY'] = '*'

# 设置环境变量强制运行
os.environ['RUN_MODE'] = 'market_only'
os.environ['GITHUB_EVENT_NAME'] = 'workflow_dispatch'  # 手动模式，绕过时间检查

# 导入主程序
from main import get_signals, send_dingtalk, TZ_BJS, log

def generate_daily_briefing():
    """生成每日投研简报"""
    now = datetime.now(TZ_BJS)
    log.info(f"🕐 当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        log.info('🚀 开始生成每日投研简报...')
        
        # 获取信号
        sigs, watch, pushed, pool_size, m_msg, total_mkt = get_signals()
        
        # 打印市场分析结果
        log.info("=" * 80)
        log.info("📊 【每日投研简报】")
        log.info("=" * 80)
        log.info(f"\n{m_msg}")
        log.info("-" * 80)
        
        # 打印股票信号
        if sigs:
            log.info("\n🎯 【股票信号】")
            for cat, signals in sigs.items():
                if signals:
                    log.info(f"\n{cat}池:")
                    for s in signals[:5]:  # 只显示前5个
                        log.info(f"  - {s.code} ({s.name}) ¥{s.price} | 得分: {s.score} | 等级: {s.level}")
        else:
            log.info("\n⚠️ 今日未发现符合条件的股票信号")
        
        # 打印观察池
        if watch:
            log.info("\n👁️ 【候补观察池】")
            for name, code, score, price in watch[:5]:
                log.info(f"  - {code} ({name}) ¥{price} | 得分: {score}")
        
        # 打印统计信息
        log.info("\n📈 【统计摘要】")
        log.info(f"  - 扫描股票池: {pool_size} 只")
        log.info(f"  - 全市场白名单: {total_mkt} 只")
        if sigs:
            total_signals = sum(len(s) for s in sigs.values())
            log.info(f"  - 生成信号: {total_signals} 个")
        
        log.info("=" * 80)
        
        return sigs, watch, m_msg
        
    except Exception as e:
        log.error(f"❌ 生成简报失败: {e}", exc_info=True)
        return None, None, None

if __name__ == '__main__':
    briefing_signals, watchlist, market_msg = generate_daily_briefing()
    
    if briefing_signals or market_msg:
        log.info("✅ 每日投研简报生成完成")
        
        # 模拟发送到钉钉（如果没有配置 webhook，会打印到本地）
        # send_dingtalk(briefing_signals, watchlist, 0, 0, market_msg)
    else:
        log.info("⚠️ 每日投研简报未生成内容")