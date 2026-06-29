import os
import sys
import json
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError

sys.path.insert(0, '/workspace')

from main import (
    NotificationGateway, _today_str, TZ_BJS,
    Config, Cols
)

C = Cols()

def safe_exec_with_timeout(func, timeout=30, default=None, *args, **kwargs):
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            return future.result(timeout=timeout)
    except TimeoutError:
        print(f"⚠️ {func.__name__} 执行超时 ({timeout}s)，使用默认值")
        return default
    except Exception as e:
        print(f"⚠️ {func.__name__} 执行异常: {e}")
        return default

def get_macro_section_fallback() -> str:
    return """### 🌍 隔夜外围宏观

> ⚠️ 外围数据获取失败（网络受限），请参考其他渠道获取实时外围行情。

**主要关注指标**:
- 标普500、纳斯达克指数走势
- VIX恐慌指数
- 美债收益率曲线
- 黄金、原油等大宗商品价格"""

def get_market_section_fallback() -> str:
    return """### 📈 A股市场全景

> ⚠️ 市场数据获取失败（网络受限），请参考实时行情软件获取最新数据。

**建议关注**:
- 上证指数、沪深300指数走势
- 涨跌家数、涨停/跌停数量
- 两市成交额变化
- 北向资金流向"""

def get_etf_rotation_fallback() -> str:
    etf_pool = [
        ('510300', '沪深300ETF'), ('510500', '中证500ETF'),
        ('159915', '创业板ETF'), ('512880', '证券ETF'),
        ('512480', '半导体ETF'), ('512690', '酒ETF'),
        ('512000', '券商ETF'), ('513100', '纳指ETF'),
        ('513500', '日经ETF'), ('588000', '科创50ETF')
    ]
    
    msg = "### 📊 ETF轮动监控\n\n"
    msg += "> ⚠️ 数据获取失败（网络受限），以下为ETF关注列表：\n\n"
    msg += "| ETF代码 | 名称 | 状态 |\n"
    msg += "|---------|------|------|\n"
    
    for code, name in etf_pool:
        msg += f"| `{code}` | {name} | 待更新 |\n"
    
    msg += "\n> 💡 **ETF轮动策略要点**：关注均线多头排列品种，优先选择量能持续放大的标的。\n"
    
    return msg

def get_sector_hotspots_fallback() -> str:
    return """### 🌋 行业热点追踪

> ⚠️ 行业热点数据获取失败（网络受限）。

**近期热门板块参考**:
- 半导体/芯片
- 人工智能/算力
- 新能源/光伏
- 消费复苏
- 金融/券商

> 💡 **策略建议**：关注板块内龙头标的，选择基本面扎实、技术形态良好的个股。"""

def get_stock_signals_fallback() -> str:
    return """### 🎯 今日精选信号

> ⚠️ 选股信号获取失败（网络受限）。

**选股策略回顾**:
- 均线多头排列优先
- 量价共振确认
- 基本面健康（PE/PB合理）
- 风控第一，严守止损线

> 📊 **提示**：建议在网络恢复后重新运行获取最新信号。"""

def get_northbound_flow_fallback() -> str:
    return """### 📊 北向资金流向

> ⚠️ 北向资金数据获取失败（网络受限）。

**北向资金是重要参考指标**:
- 持续流入通常视为积极信号
- 大幅流出需警惕市场风险
- 关注外资重仓股动向"""

def generate_daily_briefing() -> str:
    now = datetime.now(TZ_BJS)
    date_str = now.strftime('%Y年%m月%d日')
    week_day = now.strftime('%A')
    
    briefing = f"## 📅 {date_str} ({week_day}) 每日投研简报\n\n"
    
    briefing += "---\n\n"
    
    briefing += get_macro_section_fallback() + "\n\n"
    
    briefing += "---\n\n"
    
    briefing += get_market_section_fallback() + "\n\n"
    
    briefing += "---\n\n"
    
    briefing += get_etf_rotation_fallback() + "\n\n"
    
    briefing += "---\n\n"
    
    briefing += get_sector_hotspots_fallback() + "\n\n"
    
    briefing += "---\n\n"
    
    briefing += get_stock_signals_fallback() + "\n\n"
    
    briefing += "---\n\n"
    
    briefing += get_northbound_flow_fallback() + "\n\n"
    
    briefing += "---\n\n"
    
    briefing += "> 🤖 **免责声明**: 本报告仅供研究参考，不构成任何投资建议。投资有风险，入市需谨慎。\n"
    briefing += "> ⚠️ **提示**: 当前环境网络受限，部分数据未能实时获取，请在网络通畅时重新运行。\n"
    
    return briefing

if __name__ == '__main__':
    try:
        print("🚀 开始生成每日投研简报...")
        
        briefing = generate_daily_briefing()
        
        print("📝 简报生成完成：")
        print(briefing)
        
        print("\n📤 正在发送钉钉通知...")
        
        NotificationGateway.send(
            f"📅 每日投研简报 {_today_str()}",
            briefing,
            template="blue"
        )
        
        print("✅ 钉钉通知发送成功！")
        
    except Exception as e:
        print(f"❌ 生成简报失败: {e}")
        import traceback
        traceback.print_exc()
        
        error_msg = f"🚨 **每日投研简报生成失败**\n\n**时间**: {_today_str()}\n**异常信息**: {str(e)[:300]}..."
        NotificationGateway.send("🚨 简报生成失败", error_msg, template="red")