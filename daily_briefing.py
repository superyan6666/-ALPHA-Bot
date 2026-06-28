#!/usr/bin/env python3
"""
每日投研简报生成器
基于本地数据和可用数据源生成市场分析、股票信号、ETF轮动、行业热点等内容
支持钉钉/飞书通知推送
"""

import os
import sys
import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# 常量
TZ_BJS = None  # 将在导入 pytz 后设置
try:
    import pytz
    TZ_BJS = pytz.timezone('Asia/Shanghai')
except ImportError:
    TZ_BJS = None

# 数据目录
DATA_DIR = '.quantbot_data'
ASHARE_PARQUET = os.path.join(DATA_DIR, 'ashare_daily.parquet')
MACRO_PARQUET = os.path.join(DATA_DIR, 'macro_daily.parquet')


class Config:
    """配置管理"""
    def __init__(self):
        self.DINGTALK_WEBHOOK = os.environ.get('DINGTALK_WEBHOOK', '')
        self.FEISHU_WEBHOOK = os.environ.get('FEISHU_WEBHOOK', '')
        self.NOTIFY_SEC_KEYWORD = os.environ.get('NOTIFY_SEC_KEYWORD', 'Hermes')
        
    def has_webhook(self) -> bool:
        return bool(self.DINGTALK_WEBHOOK or self.FEISHU_WEBHOOK)


config = Config()


def load_local_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载本地数据"""
    ashare_df = pd.DataFrame()
    macro_df = pd.DataFrame()
    
    if os.path.exists(ASHARE_PARQUET):
        ashare_df = pd.read_parquet(ASHARE_PARQUET)
        log.info(f"✅ 加载A股日线数据: {len(ashare_df)} 条")
    
    if os.path.exists(MACRO_PARQUET):
        macro_df = pd.read_parquet(MACRO_PARQUET)
        log.info(f"✅ 加载宏观数据: {len(macro_df)} 条")
    
    return ashare_df, macro_df


def generate_market_analysis(ashare_df: pd.DataFrame, macro_df: pd.DataFrame) -> str:
    """生成市场分析"""
    content = "### 📊 市场分析\n\n"
    
    if ashare_df.empty:
        content += "⚠️ 本地A股数据为空，无法生成分析\n\n"
        return content
    
    # 获取最新交易日数据
    latest_date = ashare_df['date'].max()
    latest_data = ashare_df[ashare_df['date'] == latest_date]
    
    content += f"**最新交易日**: {latest_date}\n\n"
    
    # 市场整体统计
    total_stocks = len(latest_data)
    up_stocks = len(latest_data[latest_data['pctChg'] > 0])
    down_stocks = len(latest_data[latest_data['pctChg'] < 0])
    flat_stocks = total_stocks - up_stocks - down_stocks
    
    avg_pct = latest_data['pctChg'].mean()
    
    content += f"- **涨跌统计**: 上涨 `{up_stocks}` 只 / 下跌 `{down_stocks}` 只 / 平盘 `{flat_stocks}` 只\n"
    content += f"- **平均涨跌幅**: `{avg_pct:.2f}%`\n"
    
    # 涨幅榜前10
    top_gainers = latest_data.nlargest(10, 'pctChg')
    if not top_gainers.empty:
        content += "\n**涨幅榜 Top10**:\n"
        for _, row in top_gainers.iterrows():
            name = row.get('name', row['code'])
            content += f"- `{row['code']}` **{name}** +{row['pctChg']:.2f}% (¥{row['close']:.2f})\n"
    
    # 跌幅榜前10
    top_losers = latest_data.nsmallest(10, 'pctChg')
    if not top_losers.empty:
        content += "\n**跌幅榜 Top10**:\n"
        for _, row in top_losers.iterrows():
            name = row.get('name', row['code'])
            content += f"- `{row['code']}` **{name}** {row['pctChg']:.2f}% (¥{row['close']:.2f})\n"
    
    # 宏观指标（如有）
    if not macro_df.empty:
        latest_macro = macro_df.iloc[-1]
        content += "\n**宏观指标**:\n"
        if 'cn_10y' in macro_df.columns:
            content += f"- 中国10年期国债收益率: `{latest_macro['cn_10y']:.4f}%`\n"
        if 'us_10y' in macro_df.columns:
            content += f"- 美国10年期国债收益率: `{latest_macro['us_10y']:.2f}%`\n"
        if 'vix_close' in macro_df.columns:
            content += f"- VIX恐慌指数: `{latest_macro['vix_close']:.2f}`\n"
    
    content += "\n"
    return content


def generate_stock_signals(ashare_df: pd.DataFrame) -> str:
    """生成股票信号"""
    content = "### 🎯 股票信号\n\n"
    
    if ashare_df.empty:
        content += "⚠️ 本地数据为空，无法生成信号\n\n"
        return content
    
    # 获取最近20天数据用于计算技术指标
    latest_date = ashare_df['date'].max()
    recent_dates = ashare_df['date'].unique()[-20:] if len(ashare_df['date'].unique()) >= 20 else ashare_df['date'].unique()
    recent_data = ashare_df[ashare_df['date'].isin(recent_dates)]
    
    # 按股票分组计算
    signals = []
    
    for code in recent_data['code'].unique()[:50]:  # 只分析前50只股票以节省时间
        stock_data = recent_data[recent_data['code'] == code].sort_values('date')
        
        if len(stock_data) < 5:
            continue
        
        close = stock_data['close']
        volume = stock_data['volume'] if 'volume' in stock_data.columns else stock_data['amount']
        
        # 计算简单技术指标
        ma5 = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1] if len(close) >= 10 else ma5
        ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else ma10
        
        latest_close = close.iloc[-1]
        latest_pct = stock_data['pctChg'].iloc[-1]
        
        # 判断信号
        signal_type = "观望"
        reason = ""
        
        if latest_close > ma5 and ma5 > ma10:
            signal_type = "强势"
            reason = "收盘价高于MA5，MA5>MA10，短期趋势向上"
        elif latest_close < ma5 and ma5 < ma10:
            signal_type = "弱势"
            reason = "收盘价低于MA5，MA5<MA10，短期趋势向下"
        elif abs(latest_pct) > 5:
            signal_type = "异动"
            reason = f"当日涨跌幅 {latest_pct:.2f}%，存在较大波动"
        
        if signal_type != "观望":
            name = stock_data['name'].iloc[-1] if 'name' in stock_data.columns else code
            signals.append({
                'code': code,
                'name': name,
                'close': latest_close,
                'pct': latest_pct,
                'signal': signal_type,
                'reason': reason
            })
    
    # 输出信号
    strong_signals = [s for s in signals if s['signal'] == '强势']
    weak_signals = [s for s in signals if s['signal'] == '弱势']
    alert_signals = [s for s in signals if s['signal'] == '异动']
    
    if strong_signals:
        content += "**强势信号** (收盘>MA5>MA10):\n"
        for s in strong_signals[:5]:
            content += f"- `{s['code']}` **{s['name']}** ¥{s['close']:.2f} ({s['pct']:+.2f}%)\n"
        content += "\n"
    
    if alert_signals:
        content += "**异动信号** (涨跌幅>5%):\n"
        for s in alert_signals[:5]:
            content += f"- `{s['code']}` **{s['name']}** ¥{s['close']:.2f} ({s['pct']:+.2f}%)\n"
        content += "\n"
    
    if not signals:
        content += "今日未发现明显技术信号，建议观望。\n\n"
    
    return content


def generate_etf_rotation(ashare_df: pd.DataFrame) -> str:
    """生成ETF轮动建议"""
    content = "### 🔄 ETF轮动建议\n\n"
    
    # 主要ETF代码
    etf_codes = [
        '510300',  # 沪深300ETF
        '510500',  # 中证500ETF
        '510050',  # 50ETF
        '159915',  # 创业板ETF
        '588000',  # 科创50ETF
        '512880',  # 证券ETF
        '512690',  # 酒ETF
        '159766',  # 旅游ETF
    ]
    
    if ashare_df.empty:
        content += "⚠️ 本地数据为空\n\n"
        return content
    
    latest_date = ashare_df['date'].max()
    latest_data = ashare_df[ashare_df['date'] == latest_date]
    
    etf_performance = []
    for code in etf_codes:
        etf_data = latest_data[latest_data['code'].str.contains(code, na=False)]
        if not etf_data.empty:
            row = etf_data.iloc[0]
            etf_performance.append({
                'code': code,
                'pct': row['pctChg'],
                'close': row['close']
            })
    
    if etf_performance:
        # 按涨幅排序
        etf_performance.sort(key=lambda x: x['pct'], reverse=True)
        
        content += "**ETF涨幅排名**:\n"
        for etf in etf_performance:
            content += f"- `{etf['code']}` {etf['pct']:+.2f}% (¥{etf['close']:.3f})\n"
        
        # 轮动建议
        best_etf = etf_performance[0]
        content += f"\n**轮动建议**: 关注 `{best_etf['code']}` 近期表现较强\n"
    else:
        content += "今日ETF数据未获取，建议关注沪深300、中证500等主流ETF\n"
    
    content += "\n"
    return content


def generate_hot_sectors(ashare_df: pd.DataFrame) -> str:
    """生成行业热点"""
    content = "### 🔥 行业热点\n\n"
    
    if ashare_df.empty:
        content += "⚠️ 本地数据为空\n\n"
        return content
    
    latest_date = ashare_df['date'].max()
    latest_data = ashare_df[ashare_df['date'] == latest_date]
    
    # 简单的行业分类（根据代码前缀）
    sector_map = {
        '银行': ['601'],
        '证券': ['600', '601'],
        '医药': ['300'],
        '科技': ['002', '300'],
        '消费': ['000', '002'],
        '新能源': ['300', '688'],
        '科创板': ['688'],
    }
    
    sector_performance = {}
    for sector, prefixes in sector_map.items():
        sector_data = latest_data[latest_data['code'].str.startswith(tuple(prefixes), na=False)]
        if not sector_data.empty:
            avg_pct = sector_data['pctChg'].mean()
            sector_performance[sector] = avg_pct
    
    if sector_performance:
        # 排序
        sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
        
        content += "**行业表现**:\n"
        for sector, pct in sorted_sectors[:5]:
            status = "🔥 强势" if pct > 1 else "❄️ 弱势" if pct < -1 else "📊 震荡"
            content += f"- **{sector}**: {pct:+.2f}% {status}\n"
        
        # 热点建议
        hot_sector = sorted_sectors[0][0] if sorted_sectors[0][1] > 0 else None
        if hot_sector:
            content += f"\n**热点关注**: {hot_sector}行业近期表现活跃\n"
        else:
            content += "\n**市场状态**: 各行业普遍调整，建议观望\n"
    else:
        content += "今日行业数据未分类，建议关注市场整体走势\n"
    
    content += "\n"
    return content


def generate_macro_brief() -> str:
    """生成宏观简报"""
    content = "### 🌍 宏观快报\n\n"
    
    try:
        import yfinance as yf
        tickers = yf.Tickers("^TNX ^VIX ^GSPC GC=F")
        hist = tickers.history(period="2d")
        close_df = hist['Close']
        
        if not close_df.empty:
            def get_pct(ticker):
                s = close_df[ticker].dropna()
                if len(s) >= 2:
                    last = s.iloc[-1]
                    prev = s.iloc[-2]
                    return last, (last - prev) / prev * 100
                return s.iloc[-1] if len(s) > 0 else 0, 0
            
            vix_l, vix_p = get_pct('^VIX')
            sp500_l, sp500_p = get_pct('^GSPC')
            tnx_l, tnx_p = get_pct('^TNX')
            gc_l, gc_p = get_pct('GC=F')
            
            content += f"- **VIX恐慌指数**: `{vix_l:.2f}` ({vix_p:+.2f}%) " + ("⚠️ **极度恐慌**" if vix_l > 25 else "✅ 情绪稳定") + "\n"
            content += f"- **标普500**: `{sp500_l:.2f}` ({sp500_p:+.2f}%)\n"
            content += f"- **美债10年期**: `{tnx_l:.2f}%` ({tnx_p:+.2f}%)\n"
            content += f"- **COMEX黄金**: `{gc_l:.2f}` ({gc_p:+.2f}%)\n"
            
            # 风险提示
            if vix_l > 20:
                content += "\n> ⚠️ **风险警示**: VIX高于20，市场波动加剧，建议降低仓位\n"
    except Exception as e:
        content += f"⚠️ 外围数据获取失败: {str(e)[:50]}\n"
    
    content += "\n"
    return content


def send_notification(title: str, content: str) -> bool:
    """发送钉钉/飞书通知"""
    if not config.has_webhook():
        log.warning("⚠️ 未配置 WEBHOOK，跳过推送")
        return False
    
    webhooks = []
    if config.DINGTALK_WEBHOOK:
        webhooks.append((config.DINGTALK_WEBHOOK, False, "钉钉"))
    if config.FEISHU_WEBHOOK:
        webhooks.append((config.FEISHU_WEBHOOK, True, "飞书"))
    
    sec_keyword = config.NOTIFY_SEC_KEYWORD
    
    # 分段发送
    CHUNK_SIZE = 15000
    chunks = [content[i:i+CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
    
    success = True
    for idx, chunk in enumerate(chunks[:3]):  # 最多3段
        msg_title = title if len(chunks) == 1 else f"{title} (Part {idx+1}/{len(chunks)})"
        
        for url, is_feishu, name in webhooks:
            try:
                headers = {"Content-Type": "application/json"}
                
                if sec_keyword and sec_keyword not in msg_title:
                    msg_title = f"{sec_keyword} | {msg_title}"
                
                if is_feishu:
                    payload = {
                        "msg_type": "interactive",
                        "card": {
                            "header": {
                                "title": {"tag": "plain_text", "content": msg_title},
                                "template": "blue"
                            },
                            "elements": [{"tag": "markdown", "content": chunk}]
                        }
                    }
                else:
                    final_text = chunk
                    if sec_keyword and sec_keyword not in final_text:
                        final_text = f"### {sec_keyword}\n\n{final_text}"
                    payload = {
                        'msgtype': 'markdown',
                        'markdown': {
                            'title': msg_title,
                            'text': final_text
                        }
                    }
                
                res = requests.post(url, json=payload, headers=headers, timeout=10)
                res_dict = res.json()
                
                if is_feishu:
                    if res_dict.get('code', 0) == 0:
                        log.info(f"✅ {name} 推送成功")
                    else:
                        log.error(f"❌ {name} 推送失败: {res_dict}")
                        success = False
                else:
                    if res_dict.get('errcode', 0) == 0:
                        log.info(f"✅ {name} 推送成功")
                    else:
                        log.error(f"❌ {name} 推送失败: {res_dict}")
                        success = False
                        
            except Exception as e:
                log.error(f"❌ {name} 推送异常: {e}")
                success = False
    
    return success


def generate_daily_briefing() -> str:
    """生成每日投研简报"""
    now = datetime.now(TZ_BJS) if TZ_BJS else datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M')
    
    header = f"## 🤖 每日投研简报\n> **{now_str}**\n\n"
    
    # 加载本地数据
    ashare_df, macro_df = load_local_data()
    
    # 生成各部分内容
    content = header
    content += generate_macro_brief()
    content += generate_market_analysis(ashare_df, macro_df)
    content += generate_stock_signals(ashare_df)
    content += generate_etf_rotation(ashare_df)
    content += generate_hot_sectors(ashare_df)
    
    # 尾部信息
    content += "---\n\n"
    content += "> 📌 *本简报由量化系统自动生成，仅供参考，不构成投资建议*\n"
    content += "> 📌 *数据来源: 本地Parquet缓存 / Yahoo Finance*\n"
    
    return content


def main():
    """主函数"""
    log.info("=" * 50)
    log.info("🚀 每日投研简报生成器启动")
    log.info("=" * 50)
    
    # 生成简报
    briefing = generate_daily_briefing()
    
    # 打印到控制台
    print("\n" + "=" * 60)
    print(briefing)
    print("=" * 60 + "\n")
    
    # 发送通知
    if config.has_webhook():
        log.info("📤 发送钉钉/飞书通知...")
        success = send_notification("🤖 每日投研简报", briefing)
        if success:
            log.info("✅ 简报推送完成")
        else:
            log.warning("⚠️ 简报推送部分失败")
    else:
        log.info("⚠️ 未配置WEBHOOK，仅本地打印")
    
    log.info("=" * 50)
    log.info("✅ 每日投研简报生成完成")
    log.info("=" * 50)


if __name__ == '__main__':
    main()