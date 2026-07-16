import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main import (
        AppConfig, NotificationGateway, TZ_BJS, _today_str,
        Config, C
    )
    HAS_MAIN = True
except Exception as e:
    HAS_MAIN = False
    print(f"无法导入主模块: {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

if HAS_MAIN:
    config = AppConfig()
else:
    class MockConfig:
        DINGTALK_WEBHOOK = os.environ.get('DINGTALK_WEBHOOK', '')
        FEISHU_WEBHOOK = os.environ.get('FEISHU_WEBHOOK', '')
        NOTIFY_SEC_KEYWORD = 'Hermes'
    config = MockConfig()

os.environ['IS_MANUAL'] = 'true'
os.environ['RUN_MODE'] = 'normal'
os.environ['PUSH_EMPTY_RESULT'] = 'true'

def generate_macro_section_demo():
    now = datetime.now(TZ_BJS)
    msg = (
        f"### 🌍 隔夜外围与宏观风控快报\n"
        f"- **标普500 (^GSPC)**: `5420.58` (+0.85%) 📈\n"
        f"- **恐慌指数 (^VIX)**: `13.25` (-2.3%) ✅ 情绪稳定\n"
        f"- **黑天鹅指数 (^SKEW)**: `132.50`\n"
        f"- **美债10年期 (^TNX)**: `4.28%` (-0.05%)\n"
        f"- **COMEX 黄金 (GC=F)**: `2425.80` (+0.55%) 🛡️\n"
        f"- **WTI 原油 (CL=F)**: `78.35` (-1.2%)\n\n"
        f"> *数据源: Yahoo Finance (演示数据)*\n"
    )
    return msg

def generate_market_analysis_demo():
    now = datetime.now(TZ_BJS)
    msg = (
        f"### 📊 A股深度诊断\n"
        f"- **大盘趋势 (MA系统)**: `多头排列(短期回踩)` - 大趋势向上但短期回踩，关注下方均线支撑\n"
        f"- **上证指数**: `3528.65` (今日 **+0.62%**) 📈\n"
        f"- **综合判定**: 🔥 **强势多头 (BULL)**\n"
        f"- **市场广度**: 红盘 `3218` 家 / 绿盘 `1562` 家 (涨停 `68` / 跌停 `12`)\n"
        f"- **两市量能**: 约 `9258` 亿元\n"
        f"- 🌊 **聪明钱流向**：北水大举流入 **+45亿**\n\n"
        f"**💡 仓位建议**：仓位 60%-80%。赚钱效应极佳，资金活跃，跟随主线积极做多。\n"
    )
    return msg

def analyze_hot_sectors_demo():
    sectors = [
        ('半导体', 45),
        ('人工智能', 38),
        ('消费电子', 32),
        ('新能源汽车', 28),
        ('生物医药', 25),
        ('光伏', 22),
        ('军工', 18),
        ('金融科技', 15),
        ('云计算', 14),
        ('有色', 12),
    ]
    
    msg = "### 🔥 行业热点扫描\n\n"
    msg += "#### 📊 领涨板块TOP10\n"
    
    for sector, count in sectors:
        msg += f"- **{sector}** 🔥 (成分股: {count}只)\n"
    
    msg += "\n"
    return msg

def analyze_etf_rotation_demo():
    etf_data = [
        {'code': '512760', 'name': '半导体ETF', 'price': 1.285, 'pct_5d': 8.5, 'pct_20d': 15.3, 'pct_60d': 22.8, 'trend': '📈 多头排列', 'adx': 28.5, 'rsi': 68},
        {'code': '159995', 'name': '芯片ETF', 'price': 0.956, 'pct_5d': 7.2, 'pct_20d': 12.8, 'pct_60d': 18.5, 'trend': '📈 多头排列', 'adx': 26.2, 'rsi': 65},
        {'code': '510300', 'name': '沪深300ETF', 'price': 4.820, 'pct_5d': 2.8, 'pct_20d': 6.5, 'pct_60d': 10.2, 'trend': '⚖️ 震荡', 'adx': 15.8, 'rsi': 58},
        {'code': '510500', 'name': '中证500ETF', 'price': 7.150, 'pct_5d': 3.2, 'pct_20d': 7.8, 'pct_60d': 12.5, 'trend': '⚖️ 震荡', 'adx': 18.3, 'rsi': 61},
        {'code': '513100', 'name': '纳指ETF', 'price': 2.150, 'pct_5d': 4.5, 'pct_20d': 9.2, 'pct_60d': 18.5, 'trend': '📈 多头排列', 'adx': 22.1, 'rsi': 72},
        {'code': '513500', 'name': '中概互联', 'price': 1.380, 'pct_5d': -2.1, 'pct_20d': -5.8, 'pct_60d': -8.5, 'trend': '📉 空头排列', 'adx': 18.5, 'rsi': 42},
        {'code': '512170', 'name': '医疗ETF', 'price': 0.890, 'pct_5d': -1.5, 'pct_20d': -4.2, 'pct_60d': -6.8, 'trend': '📉 空头排列', 'adx': 14.2, 'rsi': 45},
        {'code': '512690', 'name': '酒ETF', 'price': 0.720, 'pct_5d': -0.8, 'pct_20d': -3.5, 'pct_60d': -5.2, 'trend': '⚖️ 震荡', 'adx': 12.5, 'rsi': 48},
        {'code': '512880', 'name': '证券ETF', 'price': 1.085, 'pct_5d': 2.5, 'pct_20d': 4.8, 'pct_60d': 6.2, 'trend': '⚖️ 震荡', 'adx': 16.8, 'rsi': 55},
        {'code': '159915', 'name': '创业板ETF', 'price': 3.520, 'pct_5d': 5.8, 'pct_20d': 10.5, 'pct_60d': 15.8, 'trend': '📈 多头排列', 'adx': 24.5, 'rsi': 66},
    ]
    
    df = pd.DataFrame(etf_data)
    df = df.sort_values('pct_20d', ascending=False)
    
    leaders = df.head(5).to_dict('records')
    laggards = df.tail(5).to_dict('records')
    
    bullish = df[df['trend'] == '📈 多头排列'].sort_values('pct_20d', ascending=False).head(3).to_dict('records')
    bearish = df[df['trend'] == '📉 空头排列'].sort_values('pct_20d', ascending=True).head(3).to_dict('records')
    
    msg = "### 📊 ETF轮动分析\n\n"
    
    msg += "#### 🚀 近期强势ETF (20日涨幅TOP5)\n"
    msg += "| 代码 | 名称 | 现价 | 5日 | 20日 | 60日 | 趋势 |\n"
    msg += "|------|------|------|-----|------|------|------|\n"
    for r in leaders:
        msg += f"| `{r['code']}` | {r['name']} | ¥{r['price']} | {r['pct_5d']:+.1f}% | {r['pct_20d']:+.1f}% | {r['pct_60d']:+.1f}% | {r['trend']} |\n"
    msg += "\n"
    
    msg += "#### 📉 近期弱势ETF (20日跌幅TOP5)\n"
    msg += "| 代码 | 名称 | 现价 | 5日 | 20日 | 60日 | 趋势 |\n"
    msg += "|------|------|------|-----|------|------|------|\n"
    for r in laggards:
        msg += f"| `{r['code']}` | {r['name']} | ¥{r['price']} | {r['pct_5d']:+.1f}% | {r['pct_20d']:+.1f}% | {r['pct_60d']:+.1f}% | {r['trend']} |\n"
    msg += "\n"
    
    if bullish:
        msg += "#### 🛡️ 多头排列ETF\n"
        for r in bullish:
            msg += f"- `{r['code']}` **{r['name']}** (¥{r['price']}) - ADX:{r['adx']} RSI:{r['rsi']:.0f}\n"
        msg += "\n"
    
    if bearish:
        msg += "#### ⚠️ 空头排列ETF\n"
        for r in bearish:
            msg += f"- `{r['code']}` **{r['name']}** (¥{r['price']}) - ADX:{r['adx']} RSI:{r['rsi']:.0f}\n"
        msg += "\n"
    
    return msg

def generate_stock_signals_demo():
    signals = [
        {'name': '北方华创', 'code': '002371', 'score': 92.5, 'price': 328.50, 'pct_chg': '+5.2%', 'level': '⭐⭐⭐⭐⭐ 🐯 S级·老虎机'},
        {'name': '中芯国际', 'code': '688981', 'score': 88.3, 'price': 78.20, 'pct_chg': '+4.8%', 'level': '⭐⭐⭐⭐🐕 A级·看门狗'},
        {'name': '兆易创新', 'code': '603986', 'score': 85.7, 'price': 168.80, 'pct_chg': '+3.5%', 'level': '⭐⭐⭐⭐🐕 A级·看门狗'},
        {'name': '韦尔股份', 'code': '603501', 'score': 82.1, 'price': 145.60, 'pct_chg': '+2.8%', 'level': '⭐⭐⭐🦊 B+级·小狐狸'},
        {'name': '卓胜微', 'code': '300782', 'score': 79.5, 'price': 285.30, 'pct_chg': '+2.1%', 'level': '⭐⭐⭐🦊 B+级·小狐狸'},
    ]
    
    msg = "### 🎯 精选股票信号\n\n"
    
    msg += "#### 🔥 核心主力池 (Top 3)\n"
    for s in signals[:3]:
        msg += f"- **{s['name']}** (`{s['code']}`) - 评分: `{s['score']}` {s['level']}\n"
        msg += f"  现价: ¥{s['price']} ({s['pct_chg']})\n\n"
    
    msg += "#### 🛰️ 卫星观察池\n"
    for s in signals[3:]:
        msg += f"- **{s['name']}** (`{s['code']}`) - 评分: `{s['score']}` - 现价: ¥{s['price']} ({s['pct_chg']})\n"
    msg += "\n"
    
    msg += "#### 👁️ 候补观察池 (5只)\n"
    watchlist = [
        ('汇顶科技', '603160', 68, 89.50),
        ('闻泰科技', '600745', 65, 52.30),
        ('长电科技', '600584', 62, 28.80),
        ('三安光电', '600703', 60, 25.60),
        ('紫光国微', '002049', 58, 128.50),
    ]
    for name, code, score, price in watchlist:
        msg += f"- `{code}` **{name}** (¥{price}) 得分: {score}\n"
    msg += "\n"
    
    return msg

def generate_briefing():
    now = datetime.now(TZ_BJS)
    now_str = now.strftime('%Y-%m-%d %H:%M')
    
    log.info(f"🚀 开始生成每日投研简报 ({now_str})...")
    
    briefing = f"## 📋 每日投研简报\n> **{now_str}**\n\n"
    
    briefing += "---\n\n"
    
    log.info("📊 获取宏观数据...")
    briefing += generate_macro_section_demo() + "\n\n"
    
    briefing += "---\n\n"
    
    log.info("📈 获取市场分析...")
    briefing += generate_market_analysis_demo() + "\n\n"
    
    briefing += "---\n\n"
    
    log.info("🔥 获取行业热点...")
    briefing += analyze_hot_sectors_demo()
    
    briefing += "---\n\n"
    
    log.info("🔄 获取ETF轮动分析...")
    briefing += analyze_etf_rotation_demo()
    
    briefing += "---\n\n"
    
    log.info("🎯 获取股票信号...")
    briefing += generate_stock_signals_demo()
    
    briefing += "---\n\n"
    
    briefing += "> 🤖 **AI量化选股系统** - 基于多因子模型与机器学习的智能投研助手\n"
    briefing += "> *注：当前为演示模式，数据为模拟数据。实际运行时将从真实数据源获取。*\n"
    
    return briefing

def main():
    try:
        log.info("🔧 核心配置已加载")
        
        if not config.DINGTALK_WEBHOOK:
            log.warning("⚠️ 未配置 DINGTALK_WEBHOOK 环境变量，将输出到控制台...")
        
        briefing = generate_briefing()
        
        print("\n" + "="*60)
        print("📋 每日投研简报")
        print("="*60)
        print(briefing)
        print("="*60 + "\n")
        
        if config.DINGTALK_WEBHOOK:
            NotificationGateway.send('📋 每日投研简报', briefing)
            log.info("✅ 钉钉通知发送成功")
        else:
            log.info("📝 简报已输出到控制台，未发送钉钉通知 (未配置WEBHOOK)")
            
    except Exception as e:
        log.critical(f"生成简报失败: {e}", exc_info=True)
        error_msg = f"🚨 **每日投研简报生成失败**\n\n**时间**: {datetime.now(TZ_BJS).strftime('%Y-%m-%d')}\n**异常信息**: {str(e)[:300]}..."
        try:
            NotificationGateway.send("🚨 投研简报告警", error_msg, template="red")
        except:
            pass

if __name__ == '__main__':
    main()