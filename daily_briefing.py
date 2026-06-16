#!/usr/bin/env python3
"""
每日投研简报生成器
包含市场分析、股票信号、ETF轮动、行业热点等内容
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple
import logging
import requests

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
        self.FEISHU_WEBHOOK = self.get('FEISHU_WEBHOOK', '')
        self.NOTIFY_SEC_KEYWORD = self.get('NOTIFY_SEC_KEYWORD', 'AI量化').strip()
    
    def get(self, key: str, default=None):
        return self._env.get(key, default)

config = AppConfig()

class NotificationGateway:
    @staticmethod
    def _send_to_webhook(url: str, is_feishu: bool, msg_title: str, msg_text: str, sec_keyword: str, template="blue") -> requests.Response:
        headers = {"Content-Type": "application/json"}
        if is_feishu:
            payload = {
                "msg_type": "interactive",
                "card": {
                    "header": {
                        "title": {"tag": "plain_text", "content": msg_title},
                        "template": template
                    },
                    "elements": [{"tag": "markdown", "content": msg_text}]
                }
            }
        else:
            final_title = msg_title if sec_keyword in msg_title else f"{sec_keyword} | {msg_title}"
            final_text = msg_text
            if sec_keyword not in final_text:
                final_text = f"### {sec_keyword}\n\n{final_text}"
            payload = {
                'msgtype': 'markdown',
                'markdown': {
                    'title': final_title,
                    'text': final_text
                }
            }
        
        for attempt in range(2):
            try:
                res = requests.post(url, json=payload, headers=headers, timeout=10)
                res.raise_for_status()
                return res
            except Exception as e:
                if attempt == 1:
                    raise e
                time.sleep(1)
        raise RuntimeError("Push failed after retries")

    @classmethod
    def send(cls, title: str, content: str, template="blue") -> None:
        webhooks = []
        if config.DINGTALK_WEBHOOK:
            webhooks.append((config.DINGTALK_WEBHOOK, False, "钉钉"))
        if config.FEISHU_WEBHOOK:
            webhooks.append((config.FEISHU_WEBHOOK, True, "飞书"))
            
        if not webhooks:
            log.warning("⚠️ 未配置任何 Webhook 环境变量 (DINGTALK_WEBHOOK / FEISHU_WEBHOOK)")
            return
            
        sec_keyword = config.NOTIFY_SEC_KEYWORD
        CHUNK_SIZE = 18000
        chunks = [content[i:i+CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
        
        if len(chunks) > 3:
            log.warning(f"⚠️ 推送消息过长，强行截断至前 3 篇")
            chunks = chunks[:3]
            chunks[-1] += f"\n\n> ⚠️ *(本文因超出承载极限，尾部数据已被系统强制截断)*"
            
        for idx, chunk in enumerate(chunks):
            text = chunk if idx == 0 else f"_(续上条)_\n\n{chunk}"
            msg_title = title if len(chunks) == 1 else f"{title} (Part {idx+1}/{len(chunks)})"
            
            for url, is_feishu, name in webhooks:
                try:
                    res = cls._send_to_webhook(url, is_feishu, msg_title, text, sec_keyword, template)
                    res_dict = res.json()
                    is_err = False
                    if is_feishu:
                        if res_dict.get('code', 0) != 0: is_err = True
                    else:
                        if res_dict.get('errcode', 0) != 0: is_err = True
                        
                    if is_err:
                        log.error(f"❌ {name} 推送接口拒绝: {res_dict}")
                    else:
                        log.info(f"✅ {name} 推送成功")
                except Exception as e:
                    log.error(f"❌ {name} 推送失败: {e}")
            
            if idx < len(chunks) - 1:
                time.sleep(1)

import akshare as ak

def fetch_spot():
    """获取实时行情数据"""
    try:
        df = ak.stock_zh_a_spot_em()
        if df is not None and not df.empty:
            return df
    except Exception as e:
        log.warning(f"东方财富接口失败: {e}")
    
    try:
        df = ak.stock_zh_a_spot_tx()
        if df is not None and not df.empty:
            return df
    except Exception as e:
        log.warning(f"腾讯接口失败: {e}")
    
    return pd.DataFrame()

def fetch_index(symbol):
    """获取指数数据"""
    try:
        df = ak.stock_zh_index_daily_tx(symbol=symbol)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            return df
    except Exception as e:
        log.warning(f"腾讯指数接口失败: {e}")
    
    try:
        df = ak.stock_zh_index_daily_em(symbol=symbol)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            return df
    except Exception as e:
        log.warning(f"东方财富指数接口失败: {e}")
    
    return pd.DataFrame()

def fetch_hot_sectors():
    """获取热点板块"""
    hot_stocks = {}
    try:
        df = ak.stock_board_industry_name_em()
        if df is not None and not df.empty:
            top_sectors = df.nlargest(5, '涨跌幅')['板块名称'].tolist()
            for sector in top_sectors:
                try:
                    cons = ak.stock_board_industry_cons_em(symbol=sector)
                    if cons is not None and not cons.empty:
                        col = next((c for c in cons.columns if '代码' in c), None)
                        if col:
                            for code in cons[col].astype(str).str.zfill(6).tolist():
                                hot_stocks[code] = sector
                except Exception:
                    pass
            if hot_stocks:
                return hot_stocks
    except Exception as e:
        log.warning(f"东方财富板块接口失败: {e}")
    
    return hot_stocks

def fetch_northbound_flow():
    """获取北向资金流向"""
    try:
        df = ak.stock_em_hsgt_north_net_flow_in(indicator="沪深港通")
        if df is not None and not df.empty:
            col = 'value' if 'value' in df.columns else df.columns[-1]
            today_flow = float(df.iloc[-1][col]) / 1e8
            if today_flow > 30:
                return today_flow, f"\n- 🌊 **聪明钱流向**：北水大举流入 **+{today_flow:.0f}亿**"
            elif today_flow < -30:
                return today_flow, f"\n- ❄️ **聪明钱流向**：北水大幅流出 **{today_flow:.0f}亿**"
            else:
                return today_flow, f"\n- ⚖️ **聪明钱流向**：北向资金温和 (**{today_flow:+.0f}亿**)"
    except Exception as e:
        log.warning(f"北向资金获取失败: {e}")
    return 0.0, ""

def _today_str():
    return datetime.now(TZ_BJS).strftime('%Y-%m-%d')

def generate_etf_rotation_report() -> str:
    """生成ETF轮动分析报告"""
    try:
        df_raw = fetch_spot()
        if df_raw is None or df_raw.empty:
            return "### 📦 ETF轮动分析\n⚠️ 获取ETF数据失败\n"
        
        etf_mask = df_raw['代码'].astype(str).str.startswith(('51', '15', '588', '56'))
        etf_df = df_raw[etf_mask].copy()
        
        if etf_df.empty:
            return "### 📦 ETF轮动分析\n暂无ETF数据\n"
        
        etf_df['涨跌幅'] = pd.to_numeric(etf_df['涨跌幅'], errors='coerce')
        etf_df['成交额'] = pd.to_numeric(etf_df['成交额'], errors='coerce')
        
        top_gainers = etf_df.nlargest(5, '涨跌幅')
        top_volume = etf_df.nlargest(5, '成交额')
        
        gainers_str = "\n".join([
            f"- `{row['代码']}` **{row['名称']}**: {row['涨跌幅']:.2f}% (¥{row['最新价']})"
            for _, row in top_gainers.iterrows()
        ])
        
        volume_str = "\n".join([
            f"- `{row['代码']}` **{row['名称']}**: {row['成交额']/1e8:.1f}亿"
            for _, row in top_volume.iterrows()
        ])
        
        return (
            "### 📦 ETF轮动分析\n"
            "#### 📈 涨幅榜TOP5\n"
            f"{gainers_str}\n\n"
            "#### 💰 资金流向TOP5\n"
            f"{volume_str}\n\n"
            "> *数据来源: 实时行情接口*\n"
        )
    except Exception as e:
        log.warning(f"ETF轮动分析失败: {e}")
        return f"### 📦 ETF轮动分析\n⚠️ 获取数据失败: {e}\n"

def generate_sector_report() -> str:
    """生成行业热点报告"""
    try:
        hot_map = fetch_hot_sectors()
        if not hot_map:
            return "### 🏛️ 行业热点追踪\n⚠️ 获取热点板块数据失败\n"
        
        from collections import Counter
        sector_counts = Counter(hot_map.values())
        top_sectors = sector_counts.most_common(10)
        
        sector_str = "\n".join([
            f"- **{sector}**: {count}只成分股强势"
            for sector, count in top_sectors
        ])
        
        return (
            "### 🏛️ 行业热点追踪\n"
            "#### 🔥 今日领涨主线\n"
            f"{sector_str}\n\n"
            "> *注：基于板块涨幅和成分股活跃度综合评估*\n"
        )
    except Exception as e:
        log.warning(f"行业热点分析失败: {e}")
        return f"### 🏛️ 行业热点追踪\n⚠️ 获取数据失败: {e}\n"

def generate_market_summary() -> str:
    """生成市场综合分析报告"""
    try:
        idx_df = fetch_index('sh000001')
        if idx_df is None or idx_df.empty:
            return "### 📊 市场全景分析\n⚠️ 获取大盘数据失败\n"
        
        cl = idx_df['close']
        if len(cl) < 2:
            return "### 📊 市场全景分析\n数据不足\n"
        
        close = cl.iloc[-1]
        prev_close = cl.iloc[-2] if len(cl) >= 2 else close
        pct_change = (close - prev_close) / prev_close * 100
        
        ma5 = cl.rolling(5).mean().iloc[-1] if len(cl) >= 5 else close
        ma20 = cl.rolling(20).mean().iloc[-1] if len(cl) >= 20 else close
        ma60 = cl.rolling(60).mean().iloc[-1] if len(cl) >= 60 else close
        
        north_flow, north_msg = fetch_northbound_flow()
        
        trend_status = ""
        if close > ma20 and ma5 > ma20:
            trend_status = "✅ 多头趋势健康"
        elif close < ma20 and ma5 < ma20:
            trend_status = "⚠️ 空头趋势压制"
        else:
            trend_status = "⚖️ 震荡整理阶段"
        
        return (
            "### 📊 市场全景分析\n"
            f"- **上证指数**: `{close:.2f}` ({pct_change:+.2f}%)\n"
            f"- **MA5**: `{ma5:.2f}` | **MA20**: `{ma20:.2f}` | **MA60**: `{ma60:.2f}`\n"
            f"- **趋势状态**: {trend_status}\n"
            f"{north_msg}\n\n"
            "> *数据来源: A股实时行情*\n"
        )
    except Exception as e:
        log.warning(f"市场分析失败: {e}")
        return f"### 📊 市场全景分析\n⚠️ 获取数据失败: {e}\n"

def generate_daily_briefing() -> str:
    """生成完整的每日投研简报"""
    now = datetime.now(TZ_BJS)
    now_str = now.strftime('%Y-%m-%d %H:%M')
    
    briefing = f"## 📋 AI量化每日投研简报\n> **{now_str}**\n\n"
    
    briefing += generate_market_summary()
    briefing += "\n---\n\n"
    
    briefing += generate_sector_report()
    briefing += "\n---\n\n"
    
    briefing += generate_etf_rotation_report()
    briefing += "\n---\n\n"
    
    return briefing

def main():
    """主执行函数"""
    try:
        log.info("🚀 每日投研简报生成器启动...")
        
        briefing = generate_daily_briefing()
        
        if config.DINGTALK_WEBHOOK or config.FEISHU_WEBHOOK:
            NotificationGateway.send('📋 AI量化每日投研简报', briefing)
            log.info("✅ 每日投研简报已发送")
        else:
            log.warning("未配置Webhook，仅本地打印简报")
            print(briefing)
        
    except Exception as e:
        log.error(f"每日简报生成失败: {e}", exc_info=True)
        error_msg = f"🚨 **每日投研简报生成失败**\n\n**时间**: {_today_str()}\n**错误**: {str(e)[:300]}..."
        NotificationGateway.send("🚨 投研简报生成失败", error_msg, template="red")

if __name__ == '__main__':
    import time
    main()
