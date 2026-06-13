#!/usr/bin/env python3
"""
每日投研简报生成脚本 - 离线模式
使用本地缓存数据生成简报
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Optional
import glob

import requests
import numpy as np
import pandas as pd
import pytz
import pickle

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

TZ_BJS = pytz.timezone('Asia/Shanghai')
HIST_CACHE_DIR = 'hist_cache'


class DingTalkNotifier:
    """钉钉通知发送器"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.secret_keyword = "AI量化"
    
    def send(self, title: str, content: str) -> bool:
        """发送钉钉消息"""
        if not self.webhook_url:
            log.warning("⚠️ 未配置钉钉 Webhook，消息将打印到控制台")
            print(f"\n{'='*60}")
            print(f"标题: {title}")
            print(f"{'='*60}")
            print(content)
            print(f"{'='*60}\n")
            return False
        
        # 确保标题包含关键词
        if self.secret_keyword not in title:
            title = f"{self.secret_keyword} | {title}"
        
        # 确保内容包含关键词
        if self.secret_keyword not in content:
            content = f"### {self.secret_keyword}\n\n{content}"
        
        payload = {
            'msgtype': 'markdown',
            'markdown': {
                'title': title,
                'text': content
            }
        }
        
        headers = {'Content-Type': 'application/json'}
        
        try:
            resp = requests.post(self.webhook_url, json=payload, headers=headers, timeout=15)
            resp.raise_for_status()
            result = resp.json()
            if result.get('errcode', 0) == 0:
                log.info(f"✅ 钉钉推送成功")
                return True
            else:
                log.error(f"❌ 钉钉推送失败: {result}")
                return False
        except Exception as e:
            log.error(f"❌ 钉钉推送异常: {e}")
            return False


def load_spot_cache() -> Optional[pd.DataFrame]:
    """从本地缓存加载行情数据"""
    spot_file = os.path.join(HIST_CACHE_DIR, 'spot.parquet')
    if os.path.exists(spot_file):
        try:
            df = pd.read_parquet(spot_file)
            log.info(f"✅ 从缓存加载行情数据: {len(df)} 只股票")
            return df
        except Exception as e:
            log.warning(f"加载 spot.parquet 失败: {e}")
    return None


def load_index_cache() -> Optional[pd.DataFrame]:
    """从本地缓存加载指数数据"""
    index_file = os.path.join(HIST_CACHE_DIR, 'index_sh000001.parquet')
    if os.path.exists(index_file):
        try:
            df = pd.read_parquet(index_file)
            log.info(f"✅ 从缓存加载指数数据")
            return df
        except Exception as e:
            log.warning(f"加载指数缓存失败: {e}")
    return None


def load_hot_sectors_cache() -> dict:
    """从本地缓存加载热门板块"""
    sectors_file = os.path.join(HIST_CACHE_DIR, 'hot_sectors.pkl')
    if os.path.exists(sectors_file):
        try:
            with open(sectors_file, 'rb') as f:
                payload = pickle.load(f)
                # 处理嵌套结构 {'created_at': ..., 'data': {...}}
                if isinstance(payload, dict):
                    if 'data' in payload and isinstance(payload['data'], dict):
                        data = payload['data']
                        log.info(f"✅ 从缓存加载热门板块: {len(data)} 只股票")
                        return data
                    elif 'created_at' not in payload:
                        # 直接是股票->板块映射
                        log.info(f"✅ 从缓存加载热门板块: {len(payload)} 只股票")
                        return payload
        except Exception as e:
            log.warning(f"加载热门板块缓存失败: {e}")
    return {}


def load_northbound_cache() -> tuple:
    """从本地缓存加载北向资金"""
    north_file = os.path.join(HIST_CACHE_DIR, 'northbound.pkl')
    if os.path.exists(north_file):
        try:
            with open(north_file, 'rb') as f:
                payload = pickle.load(f)
                # 处理嵌套结构 {'created_at': ..., 'data': (flow, msg)}
                if isinstance(payload, dict) and 'data' in payload:
                    data = payload['data']
                    if isinstance(data, tuple) and len(data) >= 2:
                        log.info(f"✅ 从缓存加载北向资金: {data[0]:.0f}亿")
                        return data
                elif isinstance(payload, tuple) and len(payload) >= 2:
                    log.info(f"✅ 从缓存加载北向资金: {payload[0]:.0f}亿")
                    return payload
        except Exception as e:
            log.warning(f"加载北向资金缓存失败: {e}")
    return 0.0, ""


def load_hist_data(code: str) -> Optional[pd.DataFrame]:
    """加载单只股票的历史数据"""
    # 尝试找到最新的历史文件
    pattern = os.path.join(HIST_CACHE_DIR, f"hist_{code}_*.parquet")
    files = glob.glob(pattern)
    if files:
        # 按文件名中的日期排序，取最新的
        files.sort(reverse=True)
        try:
            df = pd.read_parquet(files[0])
            return df
        except Exception as e:
            log.warning(f"加载 {code} 历史数据失败: {e}")
    return None


def analyze_market(df_spot: pd.DataFrame, df_index: pd.DataFrame, hot_sectors: dict, northbound: tuple) -> str:
    """分析市场状态"""
    now_str = datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M')
    
    # 基础统计
    if df_spot is None or df_spot.empty:
        return f"## 🤖 AI量化每日投研简报\n> **{now_str}**\n\n⚠️ 行情数据不足，无法进行市场分析"
    
    # 尝试获取涨跌幅列
    pct_col = None
    for col in df_spot.columns:
        if '涨跌幅' in str(col) or 'pct' in str(col).lower():
            pct_col = col
            break
    
    if pct_col:
        df_spot[pct_col] = pd.to_numeric(df_spot[pct_col], errors='coerce')
        up_count = (df_spot[pct_col] > 0).sum()
        down_count = (df_spot[pct_col] < 0).sum()
        zt_count = (df_spot[pct_col] >= 9.0).sum()
        dt_count = (df_spot[pct_col] <= -9.0).sum()
        total_count = up_count + down_count
        breadth = up_count / total_count if total_count > 0 else 0.5
    else:
        up_count = down_count = zt_count = dt_count = 0
        breadth = 0.5
    
    # 指数分析
    idx_close = 0.0
    idx_pct = 0.0
    market_state = "⚖️ **数据不足**"
    advice = "请参考其他指标"
    
    if df_index is not None and len(df_index) >= 60:
        close_col = None
        for col in df_index.columns:
            if 'close' in str(col).lower() or '收盘' in str(col):
                close_col = col
                break
        
        if close_col:
            cl = df_index[close_col]
            ma5 = cl.rolling(5).mean().iloc[-1] if len(cl) >= 5 else cl.iloc[-1]
            ma20 = cl.rolling(20).mean().iloc[-1] if len(cl) >= 20 else cl.iloc[-1]
            ma60 = cl.rolling(60).mean().iloc[-1] if len(cl) >= 60 else cl.iloc[-1]
            idx_close = cl.iloc[-1]
            idx_pct = (cl.iloc[-1] - cl.iloc[-2]) / cl.iloc[-2] * 100 if len(cl) >= 2 else 0.0
            
            if idx_close > ma20 and breadth > 0.6:
                market_state = "🔥 **强势多头 (BULL)**"
                advice = "仓位 60%-80%，积极做多"
            elif idx_close < ma60 and breadth < 0.4:
                market_state = "🐻 **弱势空头 (BEAR)**"
                advice = "仓位 20%-30%，防守为主"
            elif breadth < 0.25:
                market_state = "🧊 **恐慌冰点 (PANIC)**"
                advice = "仓位 10%-20%，多看少动"
            else:
                market_state = "⚖️ **震荡均衡 (NEUTRAL)**"
                advice = "仓位 40%-60%，重个股轻大盘"
    
    # 北向资金
    north_flow, north_msg = northbound
    if north_msg:
        north_msg = f"北向资金 **{north_flow:+.0f}亿**"
    else:
        north_msg = "北向资金数据暂无"
    
    # 热门板块
    hot_str = ""
    if hot_sectors:
        from collections import Counter
        sec_counts = Counter(hot_sectors.values())
        top_sectors = [f"{s}({c})" for s, c in sec_counts.most_common(5)]
        hot_str = f"\n- **核心主线**: {', '.join(top_sectors)}"
    
    # 成交额
    amt_col = None
    for col in df_spot.columns:
        if '成交额' in str(col) or 'amount' in str(col).lower():
            amt_col = col
            break
    total_amt = df_spot[amt_col].sum() / 1e8 if amt_col else 0.0
    
    report = (
        f"## 🤖 AI量化每日投研简报\n"
        f"> **{now_str}** (基于本地缓存数据)\n\n"
        f"### 📊 A股深度诊断\n"
        f"- **上证指数**: `{idx_close:.2f}` (近期 **{idx_pct:+.2f}%**)\n"
        f"- **综合判定**: {market_state}\n"
        f"- **市场广度**: 红盘 `{up_count}` 家 / 绿盘 `{down_count}` 家 (涨停 `{zt_count}` / 跌停 `{dt_count}`)\n"
        f"- **两市量能**: 约 `{total_amt:.0f}` 亿元\n"
        f"- **聪明钱流向**: {north_msg}{hot_str}\n\n"
        f"**💡 仓位建议**: {advice}\n\n"
    )
    
    return report


def select_top_stocks(df_spot: pd.DataFrame, hot_sectors: dict) -> list:
    """筛选优质股票"""
    if df_spot is None or df_spot.empty:
        return []
    
    log.info("🎯 开始筛选优质股票...")
    
    df = df_spot.copy()
    
    # 去除 ST 和退市股
    name_col = None
    for col in df.columns:
        if '名称' in str(col) or 'name' in str(col).lower():
            name_col = col
            break
    if name_col:
        df = df[~df[name_col].str.contains('ST|退', na=False)]
    
    # 过滤科创板、北交所
    code_col = None
    for col in df.columns:
        if '代码' in str(col) or 'code' in str(col).lower():
            code_col = col
            break
    if code_col:
        df[code_col] = df[code_col].astype(str).str.zfill(6)
        df = df[~df[code_col].astype(str).str.startswith(('688', '8', '4', '9'))]
    
    # 基础筛选条件
    conditions = []
    
    # 涨跌幅在合理范围
    pct_col = None
    for col in df.columns:
        if '涨跌幅' in str(col) or 'pct' in str(col).lower():
            pct_col = col
            break
    if pct_col:
        df[pct_col] = pd.to_numeric(df[pct_col], errors='coerce')
        conditions.append(df[pct_col].between(-5.0, 9.5))
    
    # 流通市值适中
    mcap_col = None
    for col in df.columns:
        if '流通市值' in str(col) or 'mcap' in str(col).lower():
            mcap_col = col
            break
    if mcap_col:
        df[mcap_col] = pd.to_numeric(df[mcap_col], errors='coerce')
        conditions.append(df[mcap_col].between(30e8, 1000e8))
    
    # 换手率适中
    turn_col = None
    for col in df.columns:
        if '换手率' in str(col) or 'turn' in str(col).lower():
            turn_col = col
            break
    if turn_col:
        df[turn_col] = pd.to_numeric(df[turn_col], errors='coerce')
        conditions.append(df[turn_col].between(1.0, 15.0))
    
    # 市盈率合理
    pe_col = None
    for col in df.columns:
        if '市盈率' in str(col) or 'pe' in str(col).lower():
            pe_col = col
            break
    if pe_col:
        df[pe_col] = pd.to_numeric(df[pe_col], errors='coerce')
        conditions.append((df[pe_col] > 0) & (df[pe_col] < 80))
    
    # 合并条件
    if conditions:
        mask = conditions[0]
        for cond in conditions[1:]:
            mask = mask & cond
        df_filtered = df[mask]
    else:
        df_filtered = df
    
    log.info(f"📊 基础筛选后剩余 {len(df_filtered)} 只股票")
    
    # 计算综合得分
    df_filtered['score'] = 50.0
    
    # 量比加分
    vr_col = None
    for col in df_filtered.columns:
        if '量比' in str(col) or 'vr' in str(col).lower():
            vr_col = col
            break
    if vr_col and pct_col:
        df_filtered[vr_col] = pd.to_numeric(df_filtered[vr_col], errors='coerce')
        df_filtered['score'] += np.where((df_filtered[vr_col] > 1.5) & (df_filtered[pct_col] > 0), 15.0, 0.0)
    
    # 热门板块加分
    if code_col:
        df_filtered['score'] += np.where(df_filtered[code_col].isin(hot_sectors.keys()), 10.0, 0.0)
    
    # 涨幅适中加分
    if pct_col:
        df_filtered['score'] += np.where((df_filtered[pct_col] > 1.0) & (df_filtered[pct_col] < 5.0), 10.0, 0.0)
    
    # 排序取前10
    df_filtered = df_filtered.sort_values('score', ascending=False).head(10)
    
    results = []
    price_col = None
    for col in df_filtered.columns:
        if '最新价' in str(col) or 'price' in str(col).lower() or '收盘' in str(col):
            price_col = col
            break
    
    for _, row in df_filtered.iterrows():
        code = str(row.get(code_col, '')) if code_col else ''
        name = str(row.get(name_col, '')) if name_col else ''
        price = float(row.get(price_col, 0)) if price_col else 0.0
        pct = float(row.get(pct_col, 0)) if pct_col else 0.0
        score = float(row.get('score', 0))
        sector = hot_sectors.get(code, "")
        
        results.append({
            'code': code,
            'name': name,
            'price': price,
            'pct': pct,
            'score': score,
            'sector': sector
        })
    
    log.info(f"✅ 筛选出 {len(results)} 只优质股票")
    return results


def generate_stock_report(stocks: list) -> str:
    """生成股票推荐报告"""
    if not stocks:
        return "\n### 🎯 今日精选股票\n\n✅ 今日未发现符合安全边际的优质标的，建议空仓防守。\n"
    
    report = "\n### 🎯 今日精选股票 (Top 10)\n\n"
    
    for i, s in enumerate(stocks, 1):
        sector_tag = f" [{s['sector']}] " if s['sector'] else ""
        level = "⭐⭐⭐⭐⭐" if s['score'] >= 80 else "⭐⭐⭐⭐" if s['score'] >= 70 else "⭐⭐⭐"
        
        # 构建链接
        prefix = '1' if s['code'].startswith('6') else '0'
        sina_market = 'sh' if s['code'].startswith('6') else 'sz'
        kline_url = f"http://image.sinajs.cn/newchart/weekly/n/{sina_market}{s['code']}.gif"
        eastmoney_url = f"https://quote.eastmoney.com/unify/r/{prefix}.{s['code']}"
        
        report += (
            f"#### {i}. {s['name']} (`{s['code']}`){sector_tag}\n"
            f"- **综合评级**: `{s['score']:.0f}` 分 {level}\n"
            f"- **近期收盘**: `¥{s['price']:.2f}` ({s['pct']:+.2f}%)\n"
            f"- [📈 周K图]({kline_url}) | [🔗 东财详情]({eastmoney_url})\n\n"
        )
    
    report += "> ⚠️ **风险提示**: 以上股票仅供参考，不构成投资建议。请结合自身风险承受能力谨慎决策。\n"
    
    return report


def generate_etf_rotation() -> str:
    """生成ETF轮动建议"""
    log.info("🔄 生成ETF轮动建议...")
    
    # 主要ETF列表
    etf_list = [
        ('510300', '沪深300ETF'),
        ('510500', '中证500ETF'),
        ('159915', '创业板ETF'),
        ('512880', '证券ETF'),
        ('512690', '酒ETF'),
        ('159766', '旅游ETF'),
        ('515790', '光伏ETF'),
        ('512480', '半导体ETF'),
    ]
    
    report = "\n### 🔄 ETF轮动建议\n\n"
    report += "| ETF代码 | 名称 | 建议 |\n"
    report += "|---------|------|------|\n"
    
    for code, name in etf_list:
        # 尝试加载历史数据
        hist = load_hist_data(code)
        if hist is not None and len(hist) >= 20:
            close_col = None
            for col in hist.columns:
                if '收盘' in str(col) or 'close' in str(col).lower():
                    close_col = col
                    break
            if close_col:
                close = hist[close_col].iloc[-1]
                pct_20d = (close / hist[close_col].iloc[-21] - 1) * 100 if len(hist) >= 21 else 0
                
                if pct_20d > 5:
                    advice = "🟢 强势持有"
                elif pct_20d > 0:
                    advice = "🟡 观察持有"
                else:
                    advice = "🔴 暂时回避"
                
                report += f"| {code} | {name} | {advice} (近20日 {pct_20d:+.2f}%) |\n"
        else:
            report += f"| {code} | {name} | 🟡 数据暂缺 |\n"
    
    report += "\n> 💡 **轮动策略**: 建议关注近20日涨幅排名前3的ETF，回避跌幅较大的板块ETF。\n"
    
    return report


def main():
    """主函数"""
    log.info("="*60)
    log.info("🚀 AI量化每日投研简报生成系统启动 (离线模式)")
    log.info("="*60)
    
    # 获取钉钉 Webhook
    dingtalk_webhook = os.environ.get('DINGTALK_WEBHOOK', '')
    
    # 初始化通知器
    notifier = DingTalkNotifier(dingtalk_webhook)
    
    # 1. 加载本地缓存数据
    df_spot = load_spot_cache()
    df_index = load_index_cache()
    hot_sectors = load_hot_sectors_cache()
    northbound = load_northbound_cache()
    
    if df_spot is None:
        error_msg = (
            "## 🤖 AI量化每日投研简报\n"
            f"> **{datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M')}**\n\n"
            "⚠️ **本地缓存数据不足**，今日简报无法生成。\n\n"
            "可能原因:\n"
            "- 本地无 spot.parquet 缓存文件\n"
            "- 缓存文件损坏或格式不兼容\n"
            "- 请先运行在线模式获取数据\n"
        )
        notifier.send("🤖 AI量化简报异常", error_msg)
        return
    
    # 2. 市场分析
    market_report = analyze_market(df_spot, df_index, hot_sectors, northbound)
    
    # 3. 股票筛选
    top_stocks = select_top_stocks(df_spot, hot_sectors)
    stock_report = generate_stock_report(top_stocks)
    
    # 4. ETF轮动
    etf_report = generate_etf_rotation()
    
    # 5. 组合完整报告
    full_report = market_report + stock_report + etf_report
    
    # 添加尾部信息
    full_report += (
        "\n---\n\n"
        f"**📌 数据来源**: 本地缓存 (hist_cache目录)\n"
        f"**⏰ 生成时间**: {datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M:%S')}\n"
        "\n> 🤖 *本报告由AI量化系统自动生成，仅供参考，不构成投资建议。*\n"
        "\n> ⚠️ *当前使用离线缓存数据，时效性可能存在偏差。*\n"
    )
    
    # 6. 发送通知
    log.info("📤 发送钉钉通知...")
    success = notifier.send("🤖 AI量化每日投研简报", full_report)
    
    if success:
        log.info("✅ 每日投研简报已成功发送到钉钉")
    else:
        log.info("ℹ️ 报告已生成（钉钉未配置或推送失败，已打印到控制台）")
    
    log.info("="*60)
    log.info("🏁 每日投研简报生成完成")
    log.info("="*60)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.critical(f"系统崩溃: {e}", exc_info=True)
        
        # 尝试发送错误通知
        webhook = os.environ.get('DINGTALK_WEBHOOK', '')
        if webhook:
            try:
                notifier = DingTalkNotifier(webhook)
                error_msg = f"🚨 **AI量化简报系统崩溃**\n\n**时间**: {datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M')}\n**异常**: {str(e)[:200]}"
                notifier.send("🚨 AI量化简报异常", error_msg)
            except Exception:
                pass