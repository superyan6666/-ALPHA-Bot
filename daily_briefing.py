import os
import sys
import json
import time
import logging
import signal
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz

TZ_BJS = pytz.timezone('Asia/Shanghai')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

USE_SIMULATED_DATA = True

def generate_macro_section_simulated() -> str:
    """生成模拟宏观数据"""
    now = datetime.now(TZ_BJS)
    dow_change = np.random.uniform(-1.5, 1.5)
    nasdaq_change = np.random.uniform(-2.0, 2.0)
    vix_val = np.random.uniform(12, 35)
    gold_change = np.random.uniform(-1.0, 1.5)
    oil_change = np.random.uniform(-3.0, 3.0)
    ten_year_yield = np.random.uniform(3.5, 4.5)
    
    vix_status = "⚠️ **极度恐慌**" if vix_val > 25 else "✅ 情绪稳定"
    
    return f"""### 🌍 隔夜外围与宏观风控快报

- **道琼斯**: `38,924.56` ({dow_change:+.2f}%)
- **纳斯达克**: `15,876.23` ({nasdaq_change:+.2f}%)
- **恐慌指数 (VIX)**: `{vix_val:.1f}` {vix_status}
- **COMEX 黄金**: `2,045.80` ({gold_change:+.2f}%)
- **NYMEX 原油**: `78.25` ({oil_change:+.2f}%)
- **美债10年期收益率**: `{ten_year_yield:.2f}%`

> *注: 当前网络环境受限，数据为模拟演示*
"""

def get_etf_rotation_simulated() -> str:
    """生成模拟ETF轮动数据"""
    etfs = [
        {'name': '半导体ETF', 'code': '512480', 'price': 6.23, 'pct': 3.25},
        {'name': '证券ETF', 'code': '512880', 'price': 1.28, 'pct': 2.45},
        {'name': '新能源车ETF', 'code': '516160', 'price': 0.85, 'pct': 1.82},
        {'name': '沪深300ETF', 'code': '510300', 'price': 3.85, 'pct': 0.95},
        {'name': '医药ETF', 'code': '512010', 'price': 0.72, 'pct': -0.56},
        {'name': '纳指ETF', 'code': '513100', 'price': 2.15, 'pct': -1.23},
    ]
    
    lines = ["### 📊 ETF轮动监控"]
    for etf in etfs:
        trend = "🟢" if etf['pct'] > 0 else "🔴"
        lines.append(f"- {trend} **{etf['name']}** (`{etf['code']}`): ¥{etf['price']} ({etf['pct']:+.2f}%)")
    
    return '\n'.join(lines) + "\n\n> *注: 当前网络环境受限，数据为模拟演示*"

def get_sector_heatmap_simulated() -> str:
    """生成模拟行业热点数据"""
    sectors = [
        ('半导体', 15),
        ('证券', 12),
        ('AI概念', 10),
        ('新能源', 8),
        ('消费电子', 6),
        ('生物医药', 5),
    ]
    
    lines = ["### 🌋 行业热点"]
    for sector, count in sectors:
        lines.append(f"- 🔥 **{sector}**: {count}只强势股")
    
    return '\n'.join(lines) + "\n\n> *注: 当前网络环境受限，数据为模拟演示*"

def get_market_breadth_simulated() -> str:
    """生成模拟市场广度数据"""
    up_count = np.random.randint(1800, 3500)
    down_count = np.random.randint(1200, 2800)
    flat_count = np.random.randint(100, 500)
    zt_count = np.random.randint(10, 150)
    dt_count = np.random.randint(0, 30)
    total_amt = np.random.uniform(7000, 15000)
    
    return f"""### 📈 市场广度

- **涨跌家数**: 🟢 {up_count} / ⚪ {flat_count} / 🔴 {down_count}
- **涨停/跌停**: 🚀 {zt_count} 只 / 💀 {dt_count} 只
- **两市成交额**: ¥{total_amt:.0f} 亿元

> *注: 当前网络环境受限，数据为模拟演示*
"""

def get_index_summary_simulated() -> str:
    """生成模拟指数数据"""
    indices = [
        ('上证指数', 3185.45, 0.52),
        ('沪深300', 3756.80, 0.78),
        ('中证500', 5324.15, 0.95),
        ('创业板指', 2156.30, 1.25),
    ]
    
    lines = ["### 📊 主要指数表现"]
    for name, value, pct in indices:
        trend = "🟢" if pct > 0 else "🔴"
        lines.append(f"- {trend} **{name}**: `{value:.2f}` ({pct:+.2f}%)")
    
    return '\n'.join(lines) + "\n\n> *注: 当前网络环境受限，数据为模拟演示*"

def get_signals_simulated() -> str:
    """生成模拟精选信号"""
    core_signals = [
        {'name': '兆易创新', 'code': '603986', 'price': 78.50, 'score': 8.5, 'pct': '+2.35%', 'reason': '放量突破MA20'},
        {'name': '中芯国际', 'code': '688981', 'price': 52.80, 'score': 7.8, 'pct': '+3.12%', 'reason': '半导体板块领涨'},
        {'name': '东方财富', 'code': '300059', 'price': 15.20, 'score': 7.2, 'pct': '+1.85%', 'reason': '成交量放大'},
    ]
    
    satellite_signals = [
        {'name': '宁德时代', 'code': '300750', 'price': 168.50, 'score': 6.8, 'pct': '+1.25%'},
        {'name': '比亚迪', 'code': '002594', 'price': 256.80, 'score': 6.5, 'pct': '+0.95%'},
        {'name': '科大讯飞', 'code': '002230', 'price': 48.20, 'score': 6.2, 'pct': '-0.55%'},
        {'name': '汇川技术', 'code': '300124', 'price': 65.80, 'score': 6.0, 'pct': '+0.35%'},
        {'name': '立讯精密', 'code': '002475', 'price': 32.50, 'score': 5.8, 'pct': '-0.85%'},
    ]
    
    lines = ["### 🎯 精选信号"]
    
    lines.append("#### 🔥 核心主力池")
    for s in core_signals:
        lines.append(f"- **{s['name']}** (`{s['code']}`): ¥{s['price']} | 得分: `{s['score']}` | {s['pct']} | {s['reason']}")
    
    lines.append("\n#### 🛰️ 卫星观察池")
    for s in satellite_signals:
        lines.append(f"- **{s['name']}** (`{s['code']}`): ¥{s['price']} | 得分: `{s['score']}` | {s['pct']}")
    
    lines.append("\n> *注: 当前网络环境受限，数据为模拟演示*")
    
    return '\n'.join(lines)

def generate_briefing() -> str:
    """生成完整的每日投研简报"""
    now = datetime.now(TZ_BJS)
    now_str = now.strftime('%Y-%m-%d %H:%M')
    
    content = f"## 📋 A股每日投研简报\n> **{now_str}**\n\n"
    
    content += "---\n\n"
    
    content += generate_macro_section_simulated()
    
    content += "---\n\n"
    
    content += get_index_summary_simulated()
    
    content += "---\n\n"
    
    content += get_market_breadth_simulated()
    
    content += "---\n\n"
    
    content += get_sector_heatmap_simulated()
    
    content += "---\n\n"
    
    content += get_etf_rotation_simulated()
    
    content += "---\n\n"
    
    content += get_signals_simulated()
    
    content += "\n---\n\n"
    content += "*Generated by Hermes 量化投研系统*"
    
    return content

def main():
    try:
        log.info("🚀 每日投研简报生成器启动...")
        
        briefing = generate_briefing()
        
        log.info("📤 发送简报到钉钉...")
        
        try:
            from main import NotificationGateway
            NotificationGateway.send("📋 A股每日投研简报", briefing)
            log.info("✅ 每日投研简报发送完成")
        except Exception as e:
            log.warning(f"钉钉推送失败: {e}")
            print("\n" + "=" * 80)
            print("📋 每日投研简报")
            print("=" * 80)
            print(briefing)
            print("=" * 80 + "\n")
            
    except Exception as e:
        log.critical(f"系统异常: {e}", exc_info=True)
        print(f"🚨 系统异常: {e}")

if __name__ == '__main__':
    main()