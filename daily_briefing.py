import os
import sys
import logging
import socket
from datetime import datetime, timedelta
from typing import Tuple, Dict, List

for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
os.environ['NO_PROXY'] = '*'

import numpy as np
import pandas as pd
import requests
import pytz

socket.setdefaulttimeout(15.0)

TZ_BJS = pytz.timezone('Asia/Shanghai')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

class BriefingConfig:
    def __init__(self):
        self.DINGTALK_WEBHOOK = os.environ.get('DINGTALK_WEBHOOK', '')
        self.NOTIFY_SEC_KEYWORD = os.environ.get('NOTIFY_SEC_KEYWORD', 'Hermes').strip()
        self.MAX_SIGNALS = 5
        self.MAX_ETFs = 5
        self.MAX_SECTORS = 5

config = BriefingConfig()

def _today_str() -> str:
    return datetime.now(TZ_BJS).strftime('%Y-%m-%d')

def _format_pct(val: float) -> str:
    return f"{val:+.2f}%" if val != 0 else "0.00%"

def _format_money(val: float) -> str:
    if abs(val) >= 1e8:
        return f"{val/1e8:.1f}亿"
    elif abs(val) >= 1e4:
        return f"{val/1e4:.0f}万"
    return f"{val:.0f}"

class DingTalkSender:
    @staticmethod
    def send(title: str, content: str) -> bool:
        if not config.DINGTALK_WEBHOOK:
            log.warning("⚠️ 未配置 DINGTALK_WEBHOOK，跳过推送")
            print(f"\n{'='*60}")
            print(f"标题: {title}")
            print(f"{'='*60}")
            print(content)
            print(f"{'='*60}")
            return False

        if config.NOTIFY_SEC_KEYWORD and config.NOTIFY_SEC_KEYWORD not in title:
            title = f"{config.NOTIFY_SEC_KEYWORD} | {title}"

        payload = {
            'msgtype': 'markdown',
            'markdown': {
                'title': title,
                'text': content
            }
        }

        headers = {"Content-Type": "application/json"}
        try:
            res = requests.post(config.DINGTALK_WEBHOOK, json=payload, headers=headers, timeout=10)
            res.raise_for_status()
            result = res.json()
            if result.get('errcode') == 0:
                log.info("✅ 钉钉推送成功")
                return True
            else:
                log.error(f"❌ 钉钉推送失败: {result}")
                return False
        except Exception as e:
            log.error(f"❌ 钉钉推送异常: {e}")
            return False

def load_local_data() -> Optional[pd.DataFrame]:
    """加载本地parquet数据"""
    try:
        df = pd.read_parquet('.quantbot_data/ashare_daily.parquet')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['code', 'date'])
        log.info(f"📦 已加载本地数据: {len(df)} 条记录, {df['code'].nunique()} 只股票")
        log.info(f"📅 数据日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
        return df
    except Exception as e:
        log.error(f"加载本地数据失败: {e}")
        return None

def get_market_analysis(local_df: pd.DataFrame) -> str:
    """获取大盘分析数据"""
    try:
        if local_df is None:
            return "⚠️ 本地数据不可用\n"
        
        sh_stocks = local_df[local_df['code'].str.startswith('sh.6')]
        
        latest_date = local_df['date'].max()
        prev_date = (local_df['date'].unique()[:-2])[-1] if len(local_df['date'].unique()) >= 2 else latest_date
        
        today_df = local_df[local_df['date'] == latest_date]
        prev_df = local_df[local_df['date'] == prev_date]
        
        today_up = (today_df['pctChg'] > 0).sum()
        today_down = (today_df['pctChg'] < 0).sum()
        today_count = len(today_df)
        
        avg_pct_today = today_df['pctChg'].mean()
        avg_pct_prev = prev_df['pctChg'].mean() if len(prev_df) > 0 else 0
        
        analysis = f"### 🌍 A股市场概况\n"
        analysis += f"- **最新数据日期**: {latest_date.date()}\n"
        analysis += f"- **市场涨跌比**: 红盘 {today_up} / 绿盘 {today_down} ({today_count}只)\n"
        analysis += f"- **平均涨跌幅**: {avg_pct_today:+.2f}%\n\n"
        
        sse_stocks = local_df[local_df['code'].str.startswith('sh.60')]
        if len(sse_stocks) > 0:
            sse_close = sse_stocks.groupby('date')['close'].mean()
            sse_ma5 = sse_close.rolling(5).mean().iloc[-1]
            sse_ma20 = sse_close.rolling(20).mean().iloc[-1]
            sse_ma60 = sse_close.rolling(60).mean().iloc[-1]
            sse_close_today = sse_close.iloc[-1]
            sse_pct = (sse_close.iloc[-1] - sse_close.iloc[-2]) / sse_close.iloc[-2] * 100 if len(sse_close) >= 2 else 0
            
            trend_str = ""
            if sse_ma5 > sse_ma20 > sse_ma60:
                trend_str = "🔥 **多头排列**"
            elif sse_ma5 < sse_ma20 < sse_ma60:
                trend_str = "🧊 **空头排列**"
            elif sse_ma20 > sse_ma60 and sse_ma5 < sse_ma20:
                trend_str = "⚖️ **震荡筑底**"
            else:
                trend_str = "📊 **震荡整理**"
            
            analysis += f"### 📈 A股大盘深度分析\n"
            analysis += f"- **沪深平均指数**: `{sse_close_today:.2f}` ({_format_pct(sse_pct)})\n"
            analysis += f"- **趋势状态**: {trend_str}\n"
            analysis += f"- **MA5**: `{sse_ma5:.2f}` | **MA20**: `{sse_ma20:.2f}` | **MA60**: `{sse_ma60:.2f}`\n"
            
            recent_60 = sse_close.iloc[-60:] if len(sse_close) >= 60 else sse_close
            vol_60d = recent_60.pct_change().std() * np.sqrt(252) * 100
            analysis += f"- **60日波动率**: `{vol_60d:.2f}%`\n"
        
        return analysis
    except Exception as e:
        log.error(f"获取市场分析失败: {e}")
        return f"⚠️ 市场分析获取失败: {e}"

def get_stock_signals() -> Tuple[str, int]:
    """获取股票信号"""
    try:
        sys.path.insert(0, '/workspace')
        from main import get_signals
        
        original_manual = os.environ.get('GITHUB_EVENT_NAME')
        os.environ['GITHUB_EVENT_NAME'] = 'workflow_dispatch'
        
        sigs, watch, pushed, pool_size, m_msg, total_mkt = get_signals()
        
        if original_manual is None:
            os.environ.pop('GITHUB_EVENT_NAME', None)
        else:
            os.environ['GITHUB_EVENT_NAME'] = original_manual
        
        if not sigs or not any(sigs.values()):
            return "✅ 今日未发现符合条件的股票信号", 0
        
        content = ""
        total_count = 0
        
        if sigs.get('Core'):
            content += "### 🎯 核心股票信号\n"
            for s in sigs['Core'][:config.MAX_SIGNALS]:
                content += (
                    f"- **{s.name}** (`{s.code}`): ¥{s.price} ({s.pct_chg}) | "
                    f"评分: `{s.score:.1f}` | "
                    f"止损: ¥{s.stop_loss} | 目标: ¥{s.target1}\n"
                )
                total_count += 1
        
        if sigs.get('Satellite'):
            content += "\n### 🛰️ 卫星观察池\n"
            for s in sigs['Satellite'][:config.MAX_SIGNALS]:
                content += (
                    f"- **{s.name}** (`{s.code}`): ¥{s.price} ({s.pct_chg}) | "
                    f"评分: `{s.score:.1f}`\n"
                )
                total_count += 1
        
        return content, total_count
    except Exception as e:
        log.error(f"获取股票信号失败: {e}")
        return f"⚠️ 股票信号获取失败: {e}", 0

def get_etf_rotation(local_df: pd.DataFrame) -> str:
    """获取ETF轮动信号"""
    try:
        if local_df is None:
            return "⚠️ 本地数据不可用\n"
        
        etf_info = {
            '510300': '沪深300ETF', '510500': '中证500ETF',
            '510050': '上证50ETF', '588000': '科创50ETF',
            '512480': '半导体ETF', '512880': '证券ETF',
            '512690': '军工ETF', '512100': '医药ETF',
            '159915': '创业板ETF', '159920': '恒生ETF'
        }
        
        etf_results = []
        
        for code, name in etf_info.items():
            try:
                df_code = local_df[local_df['code'].str.endswith(code)]
                if df_code.empty:
                    continue
                
                df_code = df_code.sort_values('date')
                close = df_code['close']
                
                if len(close) < 60:
                    continue
                
                ma5 = close.rolling(5).mean().iloc[-1]
                ma20 = close.rolling(20).mean().iloc[-1]
                ma60 = close.rolling(60).mean().iloc[-1]
                rsi14 = _calc_rsi(close, 14)
                adx = _calc_adx_df(df_code, 14)
                
                price = close.iloc[-1]
                pct_chg = df_code['pctChg'].iloc[-1]
                
                score = 0
                if price > ma20: score += 30
                if ma20 > ma60: score += 25
                if adx > 20: score += 20
                if 30 < rsi14 < 70: score += 15
                if rsi14 > 50: score += 10
                
                etf_results.append({
                    'code': code, 'name': name, 'price': price,
                    'pct_chg': pct_chg, 'score': score,
                    'ma20': ma20, 'adx': adx, 'rsi': rsi14
                })
            except Exception as e:
                log.debug(f"ETF {code} 计算失败: {e}")
                continue
        
        if not etf_results:
            return "⚠️ 未获取到有效的ETF数据\n"
        
        etf_results.sort(key=lambda x: x['score'], reverse=True)
        
        content = "### 📊 ETF轮动分析\n"
        content += "| ETF名称 | 代码 | 现价 | 涨跌幅 | 评分 | ADX | RSI |\n"
        content += "|---------|------|------|--------|------|-----|-----|\n"
        
        for etf in etf_results[:config.MAX_ETFs]:
            content += (
                f"| {etf['name']} | `{etf['code']}` | ¥{etf['price']:.2f} | "
                f"{_format_pct(etf['pct_chg'])} | `{etf['score']}` | "
                f"`{etf['adx']:.1f}` | `{etf['rsi']:.1f}` |\n"
            )
        
        top_etf = etf_results[0]
        content += f"\n💡 **推荐关注**: {top_etf['name']} (`{top_etf['code']}`)，综合评分最高 `{top_etf['score']}` 分\n"
        
        return content
    except Exception as e:
        log.error(f"获取ETF轮动失败: {e}")
        return f"⚠️ ETF轮动分析失败: {e}"

def _calc_rsi(prices: pd.Series, period: int = 14) -> float:
    """计算RSI"""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if len(rsi) > 0 else 50.0

def _calc_adx_df(df: pd.DataFrame, period: int = 14) -> float:
    """计算ADX"""
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    up, dn = high.diff(), -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / denom * 100)
    adx = dx.rolling(period).mean()
    return float(adx.iloc[-1]) if len(adx) > 0 else 20.0

def get_hot_sectors(local_df: pd.DataFrame) -> str:
    """获取行业热点"""
    try:
        if local_df is None:
            return "⚠️ 本地数据不可用\n"
        
        sector_map = {
            '600': '金融', '601': '金融', '603': '制造业', '605': '制造业',
            '000': '综合', '001': '金融', '002': '中小板', '003': '创业板',
            '300': '创业板', '301': '创业板', '688': '科创板',
            '510': '宽基ETF', '512': '行业ETF', '513': '海外ETF',
            '588': '科创ETF', '1599': '深市ETF'
        }
        
        latest_date = local_df['date'].max()
        today_df = local_df[local_df['date'] == latest_date].copy()
        
        today_df['sector'] = today_df['code'].str.slice(3, 6).str[:3].map(sector_map).fillna('其他')
        
        sector_stats = today_df.groupby('sector').agg(
            count=('code', 'count'),
            avg_pct=('pctChg', 'mean')
        ).sort_values('avg_pct', ascending=False)
        
        content = "### 🌋 行业热点追踪\n"
        
        for i, (sector, stats) in enumerate(sector_stats.head(config.MAX_SECTORS).iterrows(), 1):
            content += f"{i}. 🔥 **{sector}**: {stats['count']}只 | 平均涨幅 {stats['avg_pct']:+.2f}%\n"
        
        top_sectors = sector_stats.head(3).index.tolist()
        content += f"\n💡 **当前主线**: {' → '.join(top_sectors)}\n"
        
        return content
    except Exception as e:
        log.error(f"获取行业热点失败: {e}")
        return f"⚠️ 行业热点获取失败: {e}"

def generate_briefing() -> str:
    """生成完整的每日投研简报"""
    now_str = datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M')
    
    briefing = f"## 📋 每日投研简报\n> **{now_str}**\n\n"
    briefing += "---\n\n"
    
    local_df = load_local_data()
    
    log.info("📊 获取市场分析...")
    briefing += get_market_analysis(local_df) + "\n\n---\n\n"
    
    log.info("🎯 获取股票信号...")
    sig_content, sig_count = get_stock_signals()
    briefing += sig_content
    if sig_count > 0:
        briefing += f"\n> 今日共筛选出 **{sig_count}** 只股票信号\n"
    briefing += "\n---\n\n"
    
    log.info("📊 获取ETF轮动...")
    briefing += get_etf_rotation(local_df) + "\n\n---\n\n"
    
    log.info("🌋 获取行业热点...")
    briefing += get_hot_sectors(local_df) + "\n\n"
    
    briefing += "> 🤖 本简报由AI量化系统自动生成，仅供参考，不构成投资建议\n"
    
    return briefing

def main():
    log.info("🚀 开始生成每日投研简报...")
    
    try:
        briefing = generate_briefing()
        
        log.info("📤 发送钉钉通知...")
        success = DingTalkSender.send("📋 每日投研简报", briefing)
        
        if success:
            log.info("✅ 每日投研简报发送成功")
        else:
            log.warning("⚠️ 每日投研简报发送失败或未配置Webhook")
        
        return success
    except Exception as e:
        log.error(f"生成简报失败: {e}", exc_info=True)
        error_msg = f"🚨 **每日投研简报生成失败**\n\n**时间**: {_today_str()}\n**异常**: {str(e)[:300]}..."
        DingTalkSender.send("🚨 简报生成失败", error_msg)
        return False

if __name__ == '__main__':
    main()