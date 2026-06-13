#!/usr/bin/env python3
"""
每日投研简报生成脚本
生成包含市场分析、股票信号、ETF轮动、行业热点等内容，并发送到钉钉通知
"""

import os
# 强制清除代理设置，防止网络请求失败
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
os.environ['NO_PROXY'] = '*'

import sys
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests
import numpy as np
import pandas as pd
import pytz

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

TZ_BJS = pytz.timezone('Asia/Shanghai')

# 尝试导入 akshare
try:
    import akshare as ak
    log.info("✅ akshare 已加载")
except ImportError:
    log.error("❌ akshare 未安装，请运行: pip install akshare")
    sys.exit(1)

# 尝试导入 baostock
try:
    import baostock as bs
except ImportError:
    bs = None
    log.warning("⚠️ baostock 未安装，将使用备用数据源")

# 尝试导入 yfinance (用于宏观指标)
try:
    import yfinance as yf
except ImportError:
    yf = None
    log.warning("⚠️ yfinance 未安装，宏观指标将跳过")


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


def fetch_spot_data() -> Optional[pd.DataFrame]:
    """获取实时行情数据"""
    log.info("🚀 开始获取实时行情数据...")
    
    try:
        # 使用 akshare 获取东方财富实时行情
        df = ak.stock_zh_a_spot_em()
        if df is not None and not df.empty:
            log.info(f"✅ 获取到 {len(df)} 只股票的实时行情")
            return df
    except Exception as e:
        log.warning(f"东方财富接口异常: {e}")
    
    # 备用：使用新浪接口
    try:
        log.info("尝试新浪备用接口...")
        df = ak.stock_zh_a_spot()
        if df is not None and not df.empty:
            log.info(f"✅ 新浪接口获取到 {len(df)} 只股票")
            return df
    except Exception as e:
        log.warning(f"新浪接口也异常: {e}")
    
    return None


def fetch_index_data(symbol: str = "sh000001") -> Optional[pd.DataFrame]:
    """获取指数数据"""
    try:
        df = ak.stock_zh_index_daily_tx(symbol=symbol)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            return df
    except Exception as e:
        log.warning(f"获取指数 {symbol} 失败: {e}")
    
    try:
        df = ak.stock_zh_index_daily_em(symbol=symbol)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            return df
    except Exception as e:
        log.warning(f"东方财富指数接口也失败: {e}")
    
    return None


def fetch_hot_sectors() -> dict:
    """获取热门板块"""
    hot_stocks = {}
    try:
        df = ak.stock_board_industry_name_em()
        if df is not None and not df.empty:
            top_sectors = df.nlargest(5, '涨跌幅')['板块名称'].tolist()
            log.info(f"🌋 今日领涨板块: {', '.join(top_sectors)}")
            
            for sector in top_sectors:
                try:
                    cons = ak.stock_board_industry_cons_em(symbol=sector)
                    if cons is not None and not cons.empty:
                        col = next((c for c in cons.columns if '代码' in c), None)
                        if col:
                            for code in cons[col].astype(str).str.zfill(6).tolist()[:10]:
                                hot_stocks[code] = sector
                except Exception:
                    pass
    except Exception as e:
        log.warning(f"获取热门板块失败: {e}")
    
    return hot_stocks


def fetch_northbound_flow() -> tuple:
    """获取北向资金流向"""
    try:
        df = ak.stock_em_hsgt_north_net_flow_in(indicator="沪深港通")
        if df is not None and not df.empty:
            col = 'value' if 'value' in df.columns else df.columns[-1]
            today_flow = float(df.iloc[-1][col]) / 1e8
            return today_flow, f"北向资金 **{today_flow:+.0f}亿**"
    except Exception as e:
        log.warning(f"获取北向资金失败: {e}")
    
    return 0.0, ""


def generate_macro_report() -> str:
    """生成宏观快报"""
    if yf is None:
        return "### 🌍 隔夜外围与宏观指标\n⚠️ yfinance 未安装，跳过宏观指标"
    
    try:
        tickers = yf.Tickers("^TNX ^VIX ^GSPC GC=F CL=F")
        hist = tickers.history(period="5d")
        close_df = hist['Close']
        
        def get_last_pct(ticker):
            s = close_df[ticker].dropna()
            if len(s) >= 2:
                last = s.iloc[-1]
                prev = s.iloc[-2]
                pct = (last - prev) / prev * 100
                return last, pct
            return 0.0, 0.0
        
        tnx_l, tnx_p = get_last_pct('^TNX')
        vix_l, vix_p = get_last_pct('^VIX')
        sp500_l, sp500_p = get_last_pct('^GSPC')
        gc_l, gc_p = get_last_pct('GC=F')
        cl_l, cl_p = get_last_pct('CL=F')
        
        vix_warning = "⚠️ **极度恐慌**" if vix_l > 25 else "✅ 情绪稳定"
        
        msg = (
            f"### 🌍 隔夜外围与宏观风控快报\n"
            f"- **标普500 (^GSPC)**: `{sp500_l:.2f}` ({sp500_p:+.2f}%)\n"
            f"- **恐慌指数 (^VIX)**: `{vix_l:.2f}` ({vix_p:+.2f}%) {vix_warning}\n"
            f"- **美债10年期 (^TNX)**: `{tnx_l:.2f}%` ({tnx_p:+.2f}%)\n"
            f"- **COMEX 黄金 (GC=F)**: `{gc_l:.2f}` ({gc_p:+.2f}%)\n"
            f"- **WTI 原油 (CL=F)**: `{cl_l:.2f}` ({cl_p:+.2f}%)\n\n"
            f"> *数据源: Yahoo Finance*"
        )
        return msg
    except Exception as e:
        log.warning(f"获取宏观数据失败: {e}")
        return f"### 🌍 隔夜外围与宏观指标\n⚠️ 数据获取失败 ({e})"


def analyze_market(df_spot: pd.DataFrame) -> str:
    """分析市场状态"""
    if df_spot is None or df_spot.empty:
        return "⚠️ 行情数据获取失败，无法进行市场分析"
    
    log.info("📊 开始市场分析...")
    
    # 基础统计
    up_count = (df_spot['涨跌幅'] > 0).sum()
    down_count = (df_spot['涨跌幅'] < 0).sum()
    zt_count = (df_spot['涨跌幅'] >= 9.0).sum()
    dt_count = (df_spot['涨跌幅'] <= -9.0).sum()
    total_count = up_count + down_count
    breadth = up_count / total_count if total_count > 0 else 0.5
    
    # 获取指数数据
    idx_df = fetch_index_data()
    if idx_df is not None and len(idx_df) >= 60:
        cl = idx_df['close']
        ma5 = cl.rolling(5).mean().iloc[-1]
        ma20 = cl.rolling(20).mean().iloc[-1]
        ma60 = cl.rolling(60).mean().iloc[-1]
        idx_close = cl.iloc[-1]
        idx_pct = (cl.iloc[-1] - cl.iloc[-2]) / cl.iloc[-2] * 100 if len(cl) >= 2 else 0.0
        
        # 判断趋势
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
    else:
        idx_close = 0.0
        idx_pct = 0.0
        market_state = "⚖️ **数据不足**"
        advice = "请参考其他指标"
    
    # 北向资金
    north_flow, north_msg = fetch_northbound_flow()
    
    # 热门板块
    hot_map = fetch_hot_sectors()
    hot_str = ""
    if hot_map:
        from collections import Counter
        sec_counts = Counter(hot_map.values())
        top_sectors = [f"{s}({c})" for s, c in sec_counts.most_common(5)]
        hot_str = f"\n- **核心主线**: {', '.join(top_sectors)}"
    
    # 成交额
    total_amt = df_spot['成交额'].sum() / 1e8 if '成交额' in df_spot.columns else 0.0
    
    # 构建报告
    now_str = datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M')
    
    report = (
        f"## 🤖 AI量化每日投研简报\n"
        f"> **{now_str}**\n\n"
        f"{generate_macro_report()}\n\n"
        f"### 📊 A股深度诊断\n"
        f"- **上证指数**: `{idx_close:.2f}` (今日 **{idx_pct:+.2f}%**)\n"
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
    
    # 基础过滤
    df = df_spot.copy()
    
    # 去除 ST 和退市股
    if '名称' in df.columns:
        df = df[~df['名称'].str.contains('ST|退', na=False)]
    
    # 过滤科创板、北交所
    if '代码' in df.columns:
        df = df[~df['代码'].astype(str).str.startswith(('688', '8', '4', '9'))]
    
    # 基础筛选条件
    conditions = []
    
    # 涨跌幅在合理范围
    if '涨跌幅' in df.columns:
        conditions.append(df['涨跌幅'].between(-5.0, 9.5))
    
    # 流通市值适中
    if '流通市值' in df.columns:
        df['流通市值'] = pd.to_numeric(df['流通市值'], errors='coerce')
        conditions.append(df['流通市值'].between(30e8, 1000e8))
    
    # 换手率适中
    if '换手率' in df.columns:
        df['换手率'] = pd.to_numeric(df['换手率'], errors='coerce')
        conditions.append(df['换手率'].between(1.0, 15.0))
    
    # 市盈率合理
    if '市盈率-动态' in df.columns:
        df['市盈率-动态'] = pd.to_numeric(df['市盈率-动态'], errors='coerce')
        conditions.append((df['市盈率-动态'] > 0) & (df['市盈率-动态'] < 80))
    
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
    if '量比' in df_filtered.columns:
        df_filtered['量比'] = pd.to_numeric(df_filtered['量比'], errors='coerce')
        df_filtered['score'] += np.where((df_filtered['量比'] > 1.5) & (df_filtered['涨跌幅'] > 0), 15.0, 0.0)
    
    # 热门板块加分
    if '代码' in df_filtered.columns:
        df_filtered['score'] += np.where(df_filtered['代码'].astype(str).str.zfill(6).isin(hot_sectors.keys()), 10.0, 0.0)
    
    # 涨幅适中加分
    if '涨跌幅' in df_filtered.columns:
        df_filtered['score'] += np.where((df_filtered['涨跌幅'] > 1.0) & (df_filtered['涨跌幅'] < 5.0), 10.0, 0.0)
    
    # 排序取前10
    df_filtered = df_filtered.sort_values('score', ascending=False).head(10)
    
    results = []
    for _, row in df_filtered.iterrows():
        code = str(row.get('代码', ''))
        name = str(row.get('名称', ''))
        price = float(row.get('最新价', 0))
        pct = float(row.get('涨跌幅', 0))
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
            f"- **今日收盘**: `¥{s['price']:.2f}` ({s['pct']:+.2f}%)\n"
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
    
    try:
        # 获取ETF行情
        etf_data = []
        for code, name in etf_list:
            try:
                df = ak.fund_etf_hist_em(symbol=code, period='daily', adjust='qfq')
                if df is not None and len(df) >= 20:
                    close = df['收盘'].iloc[-1]
                    pct_5d = (close / df['收盘'].iloc[-6] - 1) * 100 if len(df) >= 6 else 0
                    pct_20d = (close / df['收盘'].iloc[-21] - 1) * 100 if len(df) >= 21 else 0
                    etf_data.append({
                        'code': code,
                        'name': name,
                        'price': close,
                        'pct_5d': pct_5d,
                        'pct_20d': pct_20d
                    })
            except Exception:
                pass
        
        if not etf_data:
            return "\n### 🔄 ETF轮动建议\n\n⚠️ ETF数据获取失败\n"
        
        # 按近20日涨幅排序
        etf_data.sort(key=lambda x: x['pct_20d'], reverse=True)
        
        report = "\n### 🔄 ETF轮动建议\n\n"
        report += "| ETF代码 | 名称 | 最新价 | 近5日 | 近20日 | 建议 |\n"
        report += "|---------|------|--------|-------|--------|------|\n"
        
        for etf in etf_data[:5]:
            if etf['pct_20d'] > 5:
                advice = "🟢 强势持有"
            elif etf['pct_20d'] > 0:
                advice = "🟡 观察持有"
            else:
                advice = "🔴 暂时回避"
            
            report += f"| {etf['code']} | {etf['name']} | ¥{etf['price']:.3f} | {etf['pct_5d']:+.2f}% | {etf['pct_20d']:+.2f}% | {advice} |\n"
        
        report += "\n> 💡 **轮动策略**: 建议关注近20日涨幅排名前3的ETF，回避跌幅较大的板块ETF。\n"
        
        return report
    except Exception as e:
        log.warning(f"ETF轮动生成失败: {e}")
        return f"\n### 🔄 ETF轮动建议\n\n⚠️ 数据获取失败: {e}\n"


def main():
    """主函数"""
    log.info("="*60)
    log.info("🚀 AI量化每日投研简报生成系统启动")
    log.info("="*60)
    
    # 获取钉钉 Webhook
    dingtalk_webhook = os.environ.get('DINGTALK_WEBHOOK', '')
    
    # 初始化通知器
    notifier = DingTalkNotifier(dingtalk_webhook)
    
    # 1. 获取实时行情
    df_spot = fetch_spot_data()
    
    if df_spot is None or df_spot.empty:
        error_msg = (
            "## 🤖 AI量化每日投研简报\n"
            f"> **{datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M')}**\n\n"
            "⚠️ **行情数据获取失败**，今日简报无法生成。\n\n"
            "可能原因:\n"
            "- 数据接口临时限制\n"
            "- 网络连接问题\n"
            "- 非交易时段数据未更新\n"
        )
        notifier.send("🤖 AI量化简报异常", error_msg)
        return
    
    # 2. 获取热门板块
    hot_sectors = fetch_hot_sectors()
    
    # 3. 市场分析
    market_report = analyze_market(df_spot)
    
    # 4. 股票筛选
    top_stocks = select_top_stocks(df_spot, hot_sectors)
    stock_report = generate_stock_report(top_stocks)
    
    # 5. ETF轮动
    etf_report = generate_etf_rotation()
    
    # 6. 组合完整报告
    full_report = market_report + stock_report + etf_report
    
    # 添加尾部信息
    full_report += (
        "\n---\n\n"
        f"**📌 数据来源**: 东方财富/新浪财经/Yahoo Finance\n"
        f"**⏰ 生成时间**: {datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M:%S')}\n"
        "\n> 🤖 *本报告由AI量化系统自动生成，仅供参考，不构成投资建议。*\n"
    )
    
    # 7. 发送通知
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