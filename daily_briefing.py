import os
import sys
import json
import logging
from datetime import datetime, timedelta

os.environ['DATA_CACHE_MODE'] = 'offline'
os.environ['IS_MANUAL'] = 'true'
os.environ['PUSH_EMPTY'] = 'true'

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    Config, Cols, fetch_spot, fetch_hist, fetch_index, 
    fetch_hot_sectors, fetch_northbound_flow, 
    extract_market_context, generate_macro_section,
    NotificationGateway, get_signals, send_dingtalk,
    TZ_BJS, _today_str
)

C = Cols()

class ETFAnalyzer:
    def __init__(self):
        self.etf_codes = {
            '510300': '沪深300ETF',
            '510500': '中证500ETF',
            '512760': '半导体ETF',
            '512010': '医药ETF',
        }

    def analyze_rotation(self) -> dict:
        results = {}
        now = datetime.now(TZ_BJS)
        end_s = now.strftime('%Y%m%d')
        start_s = (now - timedelta(days=60)).strftime('%Y%m%d')

        for code, name in self.etf_codes.items():
            try:
                hist = fetch_hist(code, start_s, end_s)
                if hist is None or len(hist) < 20:
                    continue

                close = hist[C.H_CLOSE]
                ma5 = close.rolling(5).mean().iloc[-1]
                ma20 = close.rolling(20).mean().iloc[-1]
                ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20
                
                pct_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
                pct_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0
                pct_60d = (close.iloc[-1] / close.iloc[-61] - 1) * 100 if len(close) >= 61 else 0
                
                volume = hist[C.H_VOL]
                vol_ratio = volume.iloc[-1] / volume.iloc[-6:-1].mean() if len(volume) >= 6 else 1.0
                
                trend = self._get_trend(close.iloc[-1], ma5, ma20, ma60)
                
                results[code] = {
                    'name': name,
                    'price': round(close.iloc[-1], 2),
                    'pct_5d': round(pct_5d, 2),
                    'pct_20d': round(pct_20d, 2),
                    'pct_60d': round(pct_60d, 2),
                    'vol_ratio': round(vol_ratio, 2),
                    'trend': trend,
                    'ma5': round(ma5, 2),
                    'ma20': round(ma20, 2),
                }
            except Exception as e:
                log.warning(f"分析 ETF {code} {name} 失败: {e}")

        return results

    def _get_trend(self, close, ma5, ma20, ma60):
        if close > ma5 > ma20 > ma60:
            return '强势多头'
        elif close > ma5 and ma5 > ma20:
            return '多头趋势'
        elif ma5 > ma20 and ma20 > ma60:
            return '均线多头'
        elif close < ma5 < ma20 < ma60:
            return '强势空头'
        elif close < ma5 and ma5 < ma20:
            return '空头趋势'
        elif close > ma20:
            return '震荡偏多'
        elif close < ma20:
            return '震荡偏空'
        else:
            return '震荡'

    def generate_report(self) -> str:
        rotation_data = self.analyze_rotation()
        if not rotation_data:
            return "### 📊 ETF 轮动分析\n暂无有效数据\n"

        sorted_by_5d = sorted(rotation_data.items(), key=lambda x: x[1]['pct_5d'], reverse=True)
        sorted_by_20d = sorted(rotation_data.items(), key=lambda x: x[1]['pct_20d'], reverse=True)
        sorted_by_vol = sorted(rotation_data.items(), key=lambda x: x[1]['vol_ratio'], reverse=True)

        top_5_5d = sorted_by_5d[:5]
        top_5_20d = sorted_by_20d[:5]
        top_5_vol = sorted_by_vol[:5]

        report = "### 📊 ETF 轮动分析\n\n"

        report += "**🔥 5日领涨榜**\n"
        for code, data in top_5_5d:
            report += f"- {data['name']} (`{code}`): ¥{data['price']} | 5日 {data['pct_5d']:+.2f}% | 量比 {data['vol_ratio']:.1f} | {data['trend']}\n"
        report += "\n"

        report += "**🚀 20日趋势榜**\n"
        for code, data in top_5_20d:
            report += f"- {data['name']} (`{code}`): ¥{data['price']} | 20日 {data['pct_20d']:+.2f}% | {data['trend']}\n"
        report += "\n"

        report += "**🌊 量能异动榜**\n"
        for code, data in top_5_vol:
            report += f"- {data['name']} (`{code}`): ¥{data['price']} | 量比 {data['vol_ratio']:.1f} | 5日 {data['pct_5d']:+.2f}%\n"

        return report


class SectorHotAnalyzer:
    def __init__(self):
        pass

    def get_hot_sectors_detail(self) -> list:
        hot_map = fetch_hot_sectors()
        if not hot_map:
            return []

        from collections import Counter
        sector_counts = Counter(hot_map.values())
        sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)

        results = []
        for sector, count in sorted_sectors[:10]:
            results.append({
                'name': sector,
                'stock_count': count,
            })

        return results

    def generate_report(self) -> str:
        sectors = self.get_hot_sectors_detail()
        if not sectors:
            return "### 🌋 行业热点\n暂无有效数据\n"

        report = "### 🌋 行业热点\n\n"
        report += "**🔥 今日主线板块**\n"
        
        for i, sector in enumerate(sectors[:5], 1):
            emoji = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][i-1] if i <= 5 else f'{i}️⃣'
            report += f"{emoji} **{sector['name']}** (成分股 {sector['stock_count']} 只)\n"

        report += "\n> 💡 注：以上为主力资金流向活跃的板块，建议关注板块内龙头标的。\n"
        return report


class MarketAnalyzer:
    def __init__(self):
        pass

    def get_index_stats(self) -> dict:
        results = {}
        
        indices = {
            'sh000001': '上证指数',
            'sh000300': '沪深300',
            'sh000905': '中证500',
            'sz399006': '创业板指',
        }

        for symbol, name in indices.items():
            try:
                df = fetch_index(symbol)
                if df is None or df.empty:
                    continue

                close = df['close']
                pct = (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close) >= 2 else 0
                ma5 = close.rolling(5).mean().iloc[-1]
                ma20 = close.rolling(20).mean().iloc[-1]
                ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20
                
                trend = self._get_trend(close.iloc[-1], ma5, ma20, ma60)
                
                results[symbol] = {
                    'name': name,
                    'close': round(close.iloc[-1], 2),
                    'pct': round(pct, 2),
                    'trend': trend,
                    'ma5': round(ma5, 2),
                    'ma20': round(ma20, 2),
                }
            except Exception as e:
                log.warning(f"获取指数 {name} 失败: {e}")

        return results

    def _get_trend(self, close, ma5, ma20, ma60):
        if close > ma5 > ma20 > ma60:
            return '多头排列'
        elif ma5 > ma20 > ma60:
            return '均线多头'
        elif close < ma5 < ma20 < ma60:
            return '空头排列'
        elif close > ma20:
            return '偏多'
        elif close < ma20:
            return '偏空'
        else:
            return '震荡'

    def generate_report(self) -> str:
        stats = self.get_index_stats()
        if not stats:
            return "### 📈 市场概览\n暂无有效数据\n"

        report = "### 📈 市场概览\n\n"
        report += "**主要指数表现**\n"
        
        for symbol, data in stats.items():
            color = '🟢' if data['pct'] >= 0 else '🔴'
            report += f"- {color} **{data['name']}**: ¥{data['close']} ({data['pct']:+.2f}%) | {data['trend']}\n"

        return report


def _safe_macro_section():
    try:
        import subprocess
        result = subprocess.run(
            ['python', '-c', '''
import time
start = time.time()
from main import generate_macro_section
content = generate_macro_section()
elapsed = time.time() - start
print(f"TIMING:{elapsed}")
print(content)
'''],
            capture_output=True,
            text=True,
            timeout=30,
            cwd='/workspace'
        )
        if result.returncode == 0 and 'TIMING:' in result.stdout:
            lines = result.stdout.split('\n')
            timing_line = [l for l in lines if 'TIMING:' in l][0]
            elapsed = float(timing_line.replace('TIMING:', ''))
            content = '\n'.join([l for l in lines if 'TIMING:' not in l])
            if elapsed < 30:
                return content.strip()
    except Exception as e:
        log.warning(f"子进程获取宏观数据失败: {e}")
    
    return "### 🌍 隔夜外围与宏观风控快报\n" \
           "⚠️ 外围数据获取超时，以下为本地计算的市场分析...\n\n"


def generate_daily_briefing() -> str:
    now = datetime.now(TZ_BJS)
    date_str = now.strftime('%Y年%m月%d日')
    time_str = now.strftime('%H:%M')

    briefing = f"# 📋 {date_str} A股每日投研简报\n"
    briefing += f"> 更新时间: {time_str}\n\n"

    briefing += "---\n\n"
    briefing += "## 一、宏观与外围\n\n"
    briefing += _safe_macro_section()
    briefing += "\n"

    briefing += "---\n\n"
    briefing += "## 二、市场分析\n\n"
    market_analyzer = MarketAnalyzer()
    briefing += market_analyzer.generate_report()
    briefing += "\n"

    briefing += "---\n\n"
    briefing += "## 三、行业热点\n\n"
    sector_analyzer = SectorHotAnalyzer()
    briefing += sector_analyzer.generate_report()
    briefing += "\n"

    briefing += "---\n\n"
    briefing += "## 四、ETF轮动\n\n"
    etf_analyzer = ETFAnalyzer()
    briefing += etf_analyzer.generate_report()
    briefing += "\n"

    briefing += "---\n\n"
    briefing += "## 五、股票信号\n\n"
    
    try:
        sigs, watch, pushed, pool_size, m_msg, total_mkt = get_signals()
        
        if m_msg:
            briefing += m_msg + "\n\n"
        
        total_signals = sum(len(sigs_list) for sigs_list in sigs.values())
        
        if not sigs or not any(sigs.values()):
            briefing += "✅ 今日未发现符合条件的标的，建议空仓防守。\n"
        else:
            if sigs.get('Core'):
                briefing += "### 🔥 核心主力池\n"
                for s in sigs['Core']:
                    briefing += f"- **{s.name}** (`{s.code}`): ¥{s.price} | 得分 {s.score:.1f} | {s.level}\n"
            
            if sigs.get('Satellite'):
                briefing += "\n### 🛰️ 卫星观察池\n"
                for s in sigs['Satellite']:
                    briefing += f"- **{s.name}** (`{s.code}`): ¥{s.price} | 得分 {s.score:.1f}\n"
        
        if watch:
            briefing += "\n### 👁️ 候补观察池\n"
            for name, code, score, price in watch[:5]:
                briefing += f"- **{name}** (`{code}`): ¥{price} | 得分 {score}\n"
    
    except Exception as e:
        log.error(f"获取股票信号失败: {e}")
        briefing += f"⚠️ 获取股票信号失败: {str(e)}\n"

    briefing += "\n"
    briefing += "---\n\n"
    briefing += "> 🤖 本简报由量化系统自动生成，仅供参考，不构成投资建议。\n"

    return briefing


def send_briefing_to_dingtalk(content: str):
    try:
        NotificationGateway.send('📋 每日投研简报', content)
        log.info("✅ 每日投研简报已发送到钉钉")
    except Exception as e:
        log.error(f"❌ 发送钉钉通知失败: {e}")


if __name__ == '__main__':
    try:
        log.info("🚀 开始生成每日投研简报...")
        
        briefing = generate_daily_briefing()
        
        print("=" * 60)
        print("每日投研简报内容预览:")
        print("=" * 60)
        print(briefing[:2000])
        if len(briefing) > 2000:
            print("...(内容已截断，完整内容见钉钉通知)")
        
        send_briefing_to_dingtalk(briefing)
        
        log.info("✅ 每日投研简报生成并发送完成")
        
    except Exception as e:
        log.error(f"❌ 生成每日投研简报失败: {e}", exc_info=True)
        error_msg = f"🚨 **每日投研简报生成失败**\n\n**时间**: {_today_str()}\n**异常信息**: {str(e)[:300]}..."
        try:
            NotificationGateway.send("🚨 每日投研简报生成失败", error_msg, template="red")
        except:
            pass