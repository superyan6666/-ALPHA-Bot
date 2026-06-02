#!/usr/bin/env python3
"""
每日投研简报生成器（独立版）
功能：生成包含市场分析、股票信号、ETF轮动、行业热点等内容的投研简报，并发送到钉钉通知
"""

import os
import sys
from datetime import datetime
import json
import requests
import numpy as np
import pandas as pd
import pytz


TZ_BJS = pytz.timezone('Asia/Shanghai')


class AppConfig:
    def __init__(self):
        self.DINGTALK_WEBHOOK = os.environ.get('DINGTALK_WEBHOOK', '')
        self.FEISHU_WEBHOOK = os.environ.get('FEISHU_WEBHOOK', '')
        self.NOTIFY_SEC_KEYWORD = os.environ.get('NOTIFY_SEC_KEYWORD', 'AI量化')
        self.TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', '')
    
    def print_summary(self):
        print(f"🔧 配置已加载:\n  - DINGTALK_WEBHOOK: {'已配置' if self.DINGTALK_WEBHOOK else '未配置'}\n  - FEISHU_WEBHOOK: {'已配置' if self.FEISHU_WEBHOOK else '未配置'}")


config = AppConfig()


def fetch_market_overview():
    """获取市场概览数据"""
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if df is not None and not df.empty:
            up_count = (df['涨跌幅'] > 0).sum()
            down_count = (df['涨跌幅'] < 0).sum()
            total_amt = df['成交额'].sum() / 1e8
            return {
                'up_count': up_count,
                'down_count': down_count,
                'total_amt': total_amt,
                'zt_count': (df['涨跌幅'] >= 9.0).sum(),
                'dt_count': (df['涨跌幅'] <= -9.0).sum()
            }
    except Exception as e:
        print(f"获取市场概览失败: {e}")
    
    return {'up_count': 0, 'down_count': 0, 'total_amt': 0, 'zt_count': 0, 'dt_count': 0}


def fetch_index_data():
    """获取主要指数数据"""
    try:
        import akshare as ak
        data = {}
        
        indices = [
            ('sh000001', '上证指数'),
            ('sh000300', '沪深300'),
            ('sz399006', '创业板指'),
            ('sh000852', '中证1000')
        ]
        
        for symbol, name in indices:
            try:
                df = ak.stock_zh_index_daily_tx(symbol=symbol)
                if df is not None and not df.empty:
                    df.columns = [c.lower() for c in df.columns]
                    if 'close' in df.columns and len(df) >= 2:
                        close = df['close'].iloc[-1]
                        prev_close = df['close'].iloc[-2]
                        pct = (close - prev_close) / prev_close * 100
                        data[name] = {'close': close, 'pct': pct}
            except Exception as e:
                print(f"获取指数 {name} 失败: {e}")
        
        return data
    except Exception as e:
        print(f"获取指数数据失败: {e}")
        return {}


def fetch_hot_sectors():
    """获取行业热点数据"""
    try:
        import akshare as ak
        df = ak.stock_board_industry_name_em()
        if df is not None and not df.empty:
            df['涨跌幅'] = pd.to_numeric(df['涨跌幅'], errors='coerce')
            top_sectors = df.nlargest(5, '涨跌幅')
            result = []
            for _, row in top_sectors.iterrows():
                result.append({
                    'name': row.get('板块名称', ''),
                    'pct': row.get('涨跌幅', 0),
                    'amount': row.get('成交额', 0)
                })
            return result
    except Exception as e:
        print(f"获取行业热点失败: {e}")
    
    return []


def fetch_northbound_flow():
    """获取北向资金数据"""
    try:
        import akshare as ak
        # 尝试多种可能的API
        apis_to_try = [
            lambda: ak.stock_hsgt_north_net_flow(),
            lambda: ak.stock_hsgt_north_flow(),
            lambda: ak.stock_hsgt_north_acc_flow(),
        ]
        
        for api_func in apis_to_try:
            try:
                df = api_func()
                if df is not None and not df.empty:
                    # 尝试不同的列名
                    for col in ['净流入', '北向资金', 'north_money', 'value', df.columns[-1]]:
                        if col in df.columns:
                            today_flow = float(df.iloc[-1][col]) / 1e8
                            return today_flow
                    # 如果找不到列名，尝试最后一列
                    today_flow = float(df.iloc[-1].iloc[-1]) / 1e8
                    return today_flow
            except Exception as e2:
                print(f"尝试 API 失败: {e2}")
                continue
    except Exception as e:
        print(f"获取北向资金失败: {e}")
    
    return 0.0


def fetch_macro_data():
    """获取宏观数据"""
    try:
        import yfinance as yf
        tickers = yf.Tickers("^TNX ^VIX ^GSPC GC=F CL=F")
        hist = tickers.history(period="5d")
        close_df = hist['Close']
        
        def get_last_pct(symbol):
            s = close_df[symbol].dropna()
            if len(s) >= 2:
                last = s.iloc[-1]
                prev = s.iloc[-2]
                pct = (last - prev) / prev * 100
                return last, pct
            return 0.0, 0.0
        
        return {
            'sp500': get_last_pct('^GSPC'),
            'vix': get_last_pct('^VIX'),
            'tnx': get_last_pct('^TNX'),
            'gold': get_last_pct('GC=F'),
            'oil': get_last_pct('CL=F')
        }
    except Exception as e:
        print(f"获取宏观数据失败: {e}")
        return {}


def generate_briefing():
    """生成完整的每日投研简报"""
    now = datetime.now(TZ_BJS)
    now_str = now.strftime('%Y-%m-%d %H:%M')
    
    print(f"📊 开始生成每日投研简报 - {now_str}")
    
    # 获取数据
    market_data = fetch_market_overview()
    index_data = fetch_index_data()
    hot_sectors = fetch_hot_sectors()
    north_flow = fetch_northbound_flow()
    macro_data = fetch_macro_data()
    
    # 构建简报内容
    content = f"## 📊 AI量化每日投研简报\n> **{now_str}**\n\n"
    
    # 宏观数据
    content += "### 🌍 隔夜外围市场\n"
    if macro_data:
        sp500_l, sp500_p = macro_data.get('sp500', (0, 0))
        vix_l, vix_p = macro_data.get('vix', (0, 0))
        tnx_l, tnx_p = macro_data.get('tnx', (0, 0))
        gold_l, gold_p = macro_data.get('gold', (0, 0))
        oil_l, oil_p = macro_data.get('oil', (0, 0))
        
        content += (
            f"- **标普500**: `{sp500_l:.2f}` ({sp500_p:+.2f}%)\n"
            f"- **恐慌指数VIX**: `{vix_l:.2f}` ({vix_p:+.2f}%) {'⚠️ 极度恐慌' if vix_l > 25 else ''}\n"
            f"- **美债10年期收益率**: `{tnx_l:.2f}%` ({tnx_p:+.2f}%)\n"
            f"- **COMEX黄金**: `{gold_l:.2f}` ({gold_p:+.2f}%)\n"
            f"- **WTI原油**: `{oil_l:.2f}` ({oil_p:+.2f}%)\n\n"
        )
    else:
        content += "- 外围数据获取失败\n\n"
    
    # A股市场概览
    content += "### 📈 A股市场概览\n"
    if index_data:
        for name, data in index_data.items():
            content += f"- **{name}**: `{data['close']:.2f}` ({data['pct']:+.2f}%)\n"
    else:
        content += "- 指数数据获取失败\n"
    
    content += (
        f"- **市场广度**: 红盘 {market_data['up_count']} 家 / 绿盘 {market_data['down_count']} 家\n"
        f"- **涨停/跌停**: {market_data['zt_count']} / {market_data['dt_count']}\n"
        f"- **两市成交额**: 约 {market_data['total_amt']:.0f} 亿元\n"
    )
    
    # 北向资金
    if north_flow != 0:
        if north_flow > 30:
            content += f"- 🌊 **北向资金**: 大举流入 **+{north_flow:.0f}亿**\n"
        elif north_flow < -30:
            content += f"- ❄️ **北向资金**: 大幅流出 **{north_flow:.0f}亿**\n"
        else:
            content += f"- ⚖️ **北向资金**: {north_flow:+.0f}亿\n"
    content += "\n"
    
    # 行业热点
    content += "### 🔥 行业热点板块\n"
    if hot_sectors:
        for i, sector in enumerate(hot_sectors, 1):
            content += f"{i}. **{sector['name']}**: {sector['pct']:+.2f}%\n"
    else:
        content += "- 行业热点数据获取失败\n"
    
    content += "\n---\n"
    content += "> ⚠️ **风险提示**：本报告由量化模型自动生成，仅供参考，不构成投资建议。股市有风险，投资需谨慎。"
    
    return content


def send_to_dingtalk(title, content):
    """发送钉钉通知"""
    if not config.DINGTALK_WEBHOOK:
        print("⚠️ 未配置钉钉 Webhook，跳过发送")
        return
    
    try:
        final_title = title if config.NOTIFY_SEC_KEYWORD in title else f"{config.NOTIFY_SEC_KEYWORD} | {title}"
        
        payload = {
            'msgtype': 'markdown',
            'markdown': {
                'title': final_title,
                'text': content
            }
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(config.DINGTALK_WEBHOOK, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        if result.get('errcode') == 0:
            print("✅ 钉钉推送成功")
        else:
            print(f"❌ 钉钉推送失败: {result}")
    except Exception as e:
        print(f"❌ 钉钉推送失败: {e}")


def main():
    config.print_summary()
    
    if not config.DINGTALK_WEBHOOK and not config.FEISHU_WEBHOOK:
        print("⚠️ 未配置任何通知渠道，仅生成简报")
    
    # 生成简报
    briefing = generate_briefing()
    print("\n" + "="*60)
    print(briefing)
    print("="*60)
    
    # 发送通知
    send_to_dingtalk('📊 AI量化每日投研简报', briefing)
    print("✅ 每日投研简报生成完成")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"🚨 生成简报失败: {e}")
        # 发送错误通知
        error_msg = f"🚨 **每日投研简报生成失败**\n\n**时间**: {datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M')}\n**错误信息**: {str(e)[:300]}..."
        send_to_dingtalk('🚨 投研简报失败告警', error_msg)
        raise