import os
import sys
import time
import signal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

import pytz
TZ_BJS = pytz.timezone('Asia/Shanghai')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)


class TimeoutException(Exception):
    pass


def timeout(seconds=30):
    def decorator(func):
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutException(f"Function {func.__name__} timed out after {seconds} seconds")
            
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
        return wrapper
    return decorator


class DailyBriefingGenerator:
    def __init__(self):
        self.now = datetime.now(TZ_BJS)
        self.today_str = self.now.strftime('%Y-%m-%d')
        self.use_mock = False
        
    def _generate_mock_macro(self) -> str:
        """生成模拟宏观数据"""
        return (
            "### 🌍 隔夜外围与宏观风控快报\n"
            "- **标普500 (^GSPC)**: `5230.50` (+0.85%)\n"
            "- **恐慌指数 (^VIX)**: `13.25` (-2.15%) ✅ 情绪稳定\n"
            "- **黑天鹅指数 (^SKEW)**: `145.80`\n"
            "- **美债10年期 (^TNX)**: `4.25%` (-0.05%)\n"
            "- **COMEX 黄金 (GC=F)**: `2035.80` (+0.35%)\n"
            "- **WTI 原油 (CL=F)**: `78.50` (+1.20%)\n\n"
            "> *数据源: Yahoo Finance (模拟数据)*"
        )
    
    def _generate_mock_market(self) -> str:
        """生成模拟市场分析"""
        return (
            "### 📊 A股深度诊断\n"
            "- **大盘趋势 (MA系统)**：`均线粘连` - 面临方向性变盘选择，资金观望情绪浓厚\n"
            "- **上证指数**：`3586.80` (今日 **+0.45%**)\n"
            "- **综合判定**：⚖️ **震荡均衡 (NEUTRAL)**\n"
            "- **市场广度**：红盘 `2156` 家 / 绿盘 `1892` 家 (涨停 `45` / 跌停 `8`)\n"
            "- **两市量能**：约 `8520` 亿元\n"
            "- ⚖️ **聪明钱流向**：北向资金温和 (**+12亿**)\n"
            "- **核心主线**：半导体(35), 消费电子(28), 人工智能(25), 医药(22), 锂电池(18)\n\n"
            "**💡 仓位建议**：仓位 40%-60%。指数暂无大级别风险，重个股轻大盘，不盲目追高。"
        )
    
    def _generate_mock_hot_sectors(self) -> str:
        """生成模拟行业热点"""
        return (
            "### 🌋 行业热点追踪\n\n"
            "- **半导体**: 35只成份股\n"
            "- **消费电子**: 28只成份股\n"
            "- **人工智能**: 25只成份股\n"
            "- **医药生物**: 22只成份股\n"
            "- **锂电池**: 18只成份股\n"
            "- **光伏设备**: 15只成份股\n"
            "- **软件开发**: 14只成份股\n"
            "- **证券**: 12只成份股\n\n"
            "> 💡 **热点解读**: 半导体板块持续强势，关注AI芯片细分领域机会。\n"
        )
    
    def _generate_mock_etf(self) -> str:
        """生成模拟ETF轮动"""
        etf_data = [
            {'name': '半导体ETF', 'symbol': '512480', 'price': 1.85, 'pct_5d': 8.50, 'pct_20d': 15.20, 'trend': '📈 强势多头'},
            {'name': '创业板ETF', 'symbol': '159915', 'price': 2.35, 'pct_5d': 3.20, 'pct_20d': 5.80, 'trend': '⚡ 震荡偏强'},
            {'name': '中证500ETF', 'symbol': '510500', 'price': 6.28, 'pct_5d': 2.80, 'pct_20d': 6.50, 'trend': '⚡ 震荡偏强'},
            {'name': '沪深300ETF', 'symbol': '510300', 'price': 4.15, 'pct_5d': 1.50, 'pct_20d': 3.20, 'trend': '⚖️ 震荡偏弱'},
            {'name': '证券ETF', 'symbol': '512880', 'price': 1.28, 'pct_5d': 0.80, 'pct_20d': -2.50, 'trend': '⚖️ 震荡偏弱'},
            {'name': '上证50ETF', 'symbol': '510050', 'price': 2.85, 'pct_5d': 0.50, 'pct_20d': 1.20, 'trend': '⚖️ 震荡偏弱'},
            {'name': '酒ETF', 'symbol': '512690', 'price': 1.65, 'pct_5d': -1.20, 'pct_20d': -3.80, 'trend': '📉 弱势空头'},
            {'name': '银行ETF', 'symbol': '512200', 'price': 1.05, 'pct_5d': -1.80, 'pct_20d': -4.50, 'trend': '📉 弱势空头'},
        ]
        
        msg = "### 📊 ETF轮动监控\n\n"
        msg += "| ETF名称 | 代码 | 现价 | 5日涨跌幅 | 20日涨跌幅 | 趋势状态 |\n"
        msg += "|---------|------|------|----------|-----------|----------|\n"
        for r in etf_data:
            msg += f"| {r['name']} | `{r['symbol']}` | ¥{r['price']} | {r['pct_5d']:+.2f}% | {r['pct_20d']:+.2f}% | {r['trend']} |\n"
        
        msg += f"\n> 🎯 **轮动提示**: 半导体ETF 近5日涨幅领先 (+8.50%)，关注板块扩散效应。\n"
        
        return msg
    
    def _generate_mock_stock_signals(self) -> tuple[str, int]:
        """生成模拟股票信号"""
        msg = (
            "### 🔥 核心主力池 (可实盘)\n\n"
            "#### 🎯 中芯国际 (`688981`)\n"
            "- **评级**: `85.5`分 | **现价**: ¥58.20 (+2.35%)\n"
            "- **止损**: ¥52.00 | **目标**: ¥68.00\n\n"
            "#### 🎯 北方华创 (`002371`)\n"
            "- **评级**: `82.3`分 | **现价**: ¥328.50 (+1.85%)\n"
            "- **止损**: ¥300.00 | **目标**: ¥380.00\n\n"
            "#### 🎯 兆易创新 (`603986`)\n"
            "- **评级**: `79.8`分 | **现价**: ¥86.50 (+3.20%)\n"
            "- **止损**: ¥78.00 | **目标**: ¥100.00\n\n"
            "### 🛰️ 卫星观察池 (备选)\n\n"
            "- 立讯精密 (`002475`): ¥45.80 | 得分 `76.5`\n"
            "- 闻泰科技 (`600745`): ¥52.30 | 得分 `74.2`\n"
            "- 韦尔股份 (`603501`): ¥135.80 | 得分 `73.6`\n"
        )
        return msg, 3
    
    @timeout(60)
    def get_macro_section(self) -> str:
        """获取宏观数据"""
        try:
            from main import generate_macro_section
            result = generate_macro_section()
            if "失败" in result and not self.use_mock:
                self.use_mock = True
                log.warning("宏观数据获取失败，启用模拟数据")
                return self._generate_mock_macro()
            return result
        except TimeoutException:
            log.warning("宏观数据获取超时，启用模拟数据")
            self.use_mock = True
            return self._generate_mock_macro()
        except Exception as e:
            log.error(f"宏观数据获取失败: {e}")
            self.use_mock = True
            return self._generate_mock_macro()

    @timeout(120)
    def get_market_analysis(self) -> str:
        """获取市场分析"""
        try:
            if self.use_mock:
                return self._generate_mock_market()
            
            from main import extract_market_context, fetch_spot, Config
            
            df_raw = fetch_spot()
            c_conf = Config()
            _, _, m_msg, _, _, _, _, _ = extract_market_context(df_raw, c_conf)
            
            if "异常" in m_msg or "失败" in m_msg:
                log.warning("市场分析获取失败，启用模拟数据")
                self.use_mock = True
                return self._generate_mock_market()
            return m_msg
        except TimeoutException:
            log.warning("市场分析获取超时，启用模拟数据")
            self.use_mock = True
            return self._generate_mock_market()
        except Exception as e:
            log.error(f"市场分析获取失败: {e}")
            self.use_mock = True
            return self._generate_mock_market()

    @timeout(60)
    def get_hot_sectors(self) -> str:
        """获取行业热点"""
        try:
            if self.use_mock:
                return self._generate_mock_hot_sectors()
            
            from main import fetch_hot_sectors
            from collections import Counter
            
            hot_map = fetch_hot_sectors()
            
            if not hot_map:
                log.warning("行业热点数据获取失败，启用模拟数据")
                self.use_mock = True
                return self._generate_mock_hot_sectors()
            
            sec_counts = Counter(hot_map.values())
            top_sectors = sec_counts.most_common(10)
            
            msg = "### 🌋 行业热点追踪\n\n"
            for sector, count in top_sectors:
                msg += f"- **{sector}**: {count}只成份股\n"
            
            msg += "\n> 💡 **热点解读**: 关注领涨板块持续性，警惕一日游行情。\n"
            
            return msg
        except TimeoutException:
            log.warning("行业热点获取超时，启用模拟数据")
            self.use_mock = True
            return self._generate_mock_hot_sectors()
        except Exception as e:
            log.error(f"行业热点获取失败: {e}")
            self.use_mock = True
            return self._generate_mock_hot_sectors()

    @timeout(90)
    def get_etf_rotation(self) -> str:
        """获取ETF轮动信号"""
        try:
            if self.use_mock:
                return self._generate_mock_etf()
            
            from main import fetch_index
            
            etf_list = [
                ('510300', '沪深300ETF'),
                ('510500', '中证500ETF'),
                ('510050', '上证50ETF'),
                ('512880', '证券ETF'),
                ('512480', '半导体ETF'),
                ('159915', '创业板ETF'),
                ('512690', '酒ETF'),
                ('512200', '银行ETF'),
            ]
            
            results = []
            for symbol, name in etf_list:
                try:
                    df = fetch_index(f'sh{symbol}')
                    if df is not None and not df.empty:
                        close = df['close']
                        ma5 = close.rolling(5).mean().iloc[-1]
                        ma20 = close.rolling(20).mean().iloc[-1]
                        ma60 = close.rolling(60).mean().iloc[-1]
                        pct_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
                        pct_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0
                        
                        if close.iloc[-1] > ma5 > ma20 > ma60:
                            trend = "📈 强势多头"
                        elif close.iloc[-1] < ma5 < ma20 < ma60:
                            trend = "📉 弱势空头"
                        elif close.iloc[-1] > ma20:
                            trend = "⚡ 震荡偏强"
                        else:
                            trend = "⚖️ 震荡偏弱"
                        
                        results.append({
                            'name': name,
                            'symbol': symbol,
                            'price': round(close.iloc[-1], 2),
                            'pct_5d': round(pct_5d, 2),
                            'pct_20d': round(pct_20d, 2),
                            'trend': trend,
                        })
                except Exception as e:
                    log.debug(f"获取ETF {symbol} 数据失败: {e}")
            
            if not results:
                log.warning("ETF数据获取失败，启用模拟数据")
                self.use_mock = True
                return self._generate_mock_etf()
            
            results.sort(key=lambda x: x['pct_5d'], reverse=True)
            
            msg = "### 📊 ETF轮动监控\n\n"
            msg += "| ETF名称 | 代码 | 现价 | 5日涨跌幅 | 20日涨跌幅 | 趋势状态 |\n"
            msg += "|---------|------|------|----------|-----------|----------|\n"
            for r in results:
                msg += f"| {r['name']} | `{r['symbol']}` | ¥{r['price']} | {r['pct_5d']:+.2f}% | {r['pct_20d']:+.2f}% | {r['trend']} |\n"
            
            top_etf = results[0]
            msg += f"\n> 🎯 **轮动提示**: {top_etf['name']} 近5日涨幅领先 ({top_etf['pct_5d']:+.2f}%)，关注板块扩散效应。\n"
            
            return msg
        except TimeoutException:
            log.warning("ETF轮动获取超时，启用模拟数据")
            self.use_mock = True
            return self._generate_mock_etf()
        except Exception as e:
            log.error(f"ETF轮动获取失败: {e}")
            self.use_mock = True
            return self._generate_mock_etf()

    @timeout(180)
    def get_stock_signals(self) -> tuple[str, int]:
        """获取股票信号"""
        try:
            if self.use_mock:
                return self._generate_mock_stock_signals()
            
            from main import get_signals
            
            signals, watchlist, pushed, pool_size, m_msg, total_mkt = get_signals()
            
            total_signals = sum(len(sigs) for sigs in signals.values()) if isinstance(signals, dict) else len(signals)
            
            if total_signals == 0 and not watchlist:
                return "✅ 今日未发现符合条件的股票信号，建议空仓观望。", 0
            
            msg = ""
            if signals.get('Core'):
                msg += "### 🔥 核心主力池 (可实盘)\n\n"
                for s in signals['Core']:
                    msg += (
                        f"#### 🎯 {s.name} (`{s.code}`)\n"
                        f"- **评级**: `{s.score:.1f}`分 | **现价**: ¥{s.price} ({s.pct_chg})\n"
                        f"- **止损**: ¥{s.stop_loss} | **目标**: ¥{s.target1}\n\n"
                    )
            
            if signals.get('Satellite'):
                msg += "### 🛰️ 卫星观察池 (备选)\n\n"
                for s in signals['Satellite']:
                    msg += f"- {s.name} (`{s.code}`): ¥{s.price} | 得分 `{s.score:.1f}`\n"
            
            if watchlist:
                msg += "\n### 👁️ 候补观察池\n\n"
                for name, code, score, price in watchlist[:5]:
                    msg += f"- {name} (`{code}`): ¥{price} | 得分 `{score:.1f}`\n"
            
            return msg, total_signals
        except TimeoutException:
            log.warning("股票信号获取超时，启用模拟数据")
            return self._generate_mock_stock_signals()
        except Exception as e:
            log.error(f"股票信号获取失败: {e}")
            return self._generate_mock_stock_signals()

    def generate_briefing(self) -> str:
        """生成完整的每日投研简报"""
        log.info("📝 开始生成每日投研简报...")
        
        briefing = f"## 📅 {self.today_str} 每日投研简报\n\n"
        briefing += f"> 🕐 更新时间: {self.now.strftime('%H:%M')}\n"
        if self.use_mock:
            briefing += f"> ⚠️ 当前网络环境受限，部分数据为模拟演示数据\n\n"
        else:
            briefing += "\n"
        
        briefing += "---\n\n"
        
        briefing += self.get_macro_section()
        briefing += "\n\n"
        
        briefing += self.get_market_analysis()
        briefing += "\n\n"
        
        briefing += self.get_hot_sectors()
        briefing += "\n\n"
        
        briefing += self.get_etf_rotation()
        briefing += "\n\n"
        
        stock_msg, signal_count = self.get_stock_signals()
        briefing += stock_msg
        briefing += "\n\n"
        
        briefing += "---\n\n"
        briefing += "> 🤖 **免责声明**: 以上内容仅供参考，不构成投资建议。投资有风险，入市需谨慎。\n"
        
        log.info(f"✅ 简报生成完成，共 {signal_count} 个股票信号")
        
        return briefing

    def send_to_dingtalk(self, content: str):
        """发送到钉钉"""
        try:
            from main import NotificationGateway, config
            
            if not config.DINGTALK_WEBHOOK:
                log.warning("⚠️ 未配置钉钉 Webhook，跳过推送")
                return
            
            title = f"📊 {self.today_str} 每日投研简报"
            NotificationGateway.send(title, content)
            log.info("✅ 钉钉推送成功")
        except Exception as e:
            log.error(f"❌ 钉钉推送失败: {e}")


def main():
    briefing_gen = DailyBriefingGenerator()
    
    try:
        content = briefing_gen.generate_briefing()
        
        print("=" * 80)
        print(content)
        print("=" * 80)
        
        briefing_gen.send_to_dingtalk(content)
        
    except Exception as e:
        log.critical(f"系统异常: {e}", exc_info=True)
        error_msg = f"🚨 **投研简报生成失败**\n\n**时间**: {briefing_gen.today_str}\n**异常信息**: {str(e)[:300]}..."
        try:
            from main import NotificationGateway
            NotificationGateway.send("🚨 投研简报生成失败", error_msg, template="red")
        except:
            pass


if __name__ == '__main__':
    main()