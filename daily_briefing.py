import os
import sys
import time
import json
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

import pandas as pd
import numpy as np

TZ_BJS = __import__('pytz').timezone('Asia/Shanghai')

class DailyBriefing:
    def __init__(self):
        self.now = datetime.now(TZ_BJS)
        self.now_str = self.now.strftime('%Y-%m-%d %H:%M')
        self.date_str = self.now.strftime('%Y-%m-%d')
        self.dingtalk_webhook = os.environ.get('DINGTALK_WEBHOOK', '')
        self.dingtalk_secret = os.environ.get('DINGTALK_SECRET', '')
        
        self.briefing_data = {
            'market_analysis': {},
            'stock_signals': {},
            'etf_rotation': [],
            'hot_sectors': [],
            'macro_data': {}
        }
    
    def generate_macro_section(self) -> str:
        """获取外围宏观数据"""
        try:
            import yfinance as yf
            
            yf.set_tz_cache_location(None)
            
            results = {}
            
            for ticker in ["^GSPC", "^VIX", "^TNX", "GC=F", "CL=F"]:
                try:
                    t = yf.Ticker(ticker)
                    hist = t.history(period="5d", timeout=5)
                    if len(hist) >= 2:
                        close_series = hist['Close']
                        last = close_series.iloc[-1]
                        prev = close_series.iloc[-2]
                        pct = (last - prev) / prev * 100
                        results[ticker] = {'last': last, 'pct': pct}
                    else:
                        results[ticker] = {'last': 0.0, 'pct': 0.0}
                except Exception:
                    results[ticker] = {'last': 0.0, 'pct': 0.0}

            sp500_l, sp500_p = results['^GSPC']['last'], results['^GSPC']['pct']
            vix_l, vix_p = results['^VIX']['last'], results['^VIX']['pct']
            tnx_l, tnx_p = results['^TNX']['last'], results['^TNX']['pct']
            gc_l, gc_p = results['GC=F']['last'], results['GC=F']['pct']
            cl_l, cl_p = results['CL=F']['last'], results['CL=F']['pct']

            self.briefing_data['macro_data'] = {
                'sp500': {'value': round(sp500_l, 2), 'change': round(sp500_p, 2)},
                'vix': {'value': round(vix_l, 2), 'change': round(vix_p, 2), 'status': '极度恐慌' if vix_l > 25 else '情绪稳定'},
                'tnx': {'value': round(tnx_l, 2), 'change': round(tnx_p, 2)},
                'gold': {'value': round(gc_l, 2), 'change': round(gc_p, 2)},
                'oil': {'value': round(cl_l, 2), 'change': round(cl_p, 2)}
            }

            msg = (
                f"### 🌍 隔夜外围与宏观风控快报\n"
                f"- **标普500 (^GSPC)**: `{sp500_l:.2f}` ({sp500_p:+.2f}%)\n"
                f"- **恐慌指数 (^VIX)**: `{vix_l:.2f}` ({vix_p:+.2f}%) " + ("⚠️ **极度恐慌**" if vix_l > 25 else "✅ 情绪稳定") + "\n"
                f"- **美债10年期 (^TNX)**: `{tnx_l:.2f}%` ({tnx_p:+.2f}%)\n"
                f"- **COMEX 黄金 (GC=F)**: `{gc_l:.2f}` ({gc_p:+.2f}%)\n"
                f"- **WTI 原油 (CL=F)**: `{cl_l:.2f}` ({cl_p:+.2f}%)\n\n"
                f"> *数据源: Yahoo Finance*"
            )
            return msg
        except Exception as e:
            return f"### 🌍 隔夜外围与宏观指标快报\n⚠️ 外围数据获取失败 ({e})"
    
    def fetch_a_share_market(self) -> Dict:
        """获取A股市场整体数据"""
        try:
            import akshare as ak
            
            df = ak.stock_zh_a_spot_em()
            if df.empty:
                df = ak.stock_zh_a_spot()
            
            if df.empty:
                return {'error': '数据获取失败'}
            
            pct_col = None
            for col in ['涨跌幅', 'change', 'pct']:
                if col in df.columns:
                    pct_col = col
                    break
            
            if pct_col is None:
                return {'error': '未找到涨跌幅列'}
            
            df[pct_col] = pd.to_numeric(df[pct_col], errors='coerce')
            
            up_count = len(df[df[pct_col] > 0])
            down_count = len(df[df[pct_col] < 0])
            
            zt_count = len(df[df[pct_col] >= 9.8])
            dt_count = len(df[df[pct_col] <= -9.8])
            
            amt_col = None
            for col in ['成交额', 'amount', 'vol']:
                if col in df.columns:
                    amt_col = col
                    break
            
            total_amt = 0
            if amt_col:
                try:
                    df[amt_col] = pd.to_numeric(df[amt_col], errors='coerce')
                    total_amt = df[amt_col].sum() / 10000
                except:
                    pass
            
            self.briefing_data['market_analysis'] = {
                'up_count': up_count,
                'down_count': down_count,
                'zt_count': zt_count,
                'dt_count': dt_count,
                'total_amount': round(total_amt, 2),
                'total_stocks': len(df)
            }
            
            return {
                'up_count': up_count,
                'down_count': down_count,
                'zt_count': zt_count,
                'dt_count': dt_count,
                'total_amount': total_amt
            }
        except Exception as e:
            return {'error': str(e)}
    
    def fetch_index_data(self, index_code: str = 'sh000001') -> Dict:
        """获取指数数据"""
        try:
            import akshare as ak
            df = ak.stock_zh_index_daily(symbol=index_code)
            if len(df) < 5:
                return {'error': '数据不足'}
            
            df = df.sort_values('date')
            cl = df['close']
            ma5 = cl.rolling(5).mean().iloc[-1]
            ma20 = cl.rolling(20).mean().iloc[-1]
            ma60 = cl.rolling(60).mean().iloc[-1]
            
            pct = (cl.iloc[-1] - cl.iloc[-2]) / cl.iloc[-2] * 100
            
            exp1 = cl.ewm(span=12, adjust=False).mean()
            exp2 = cl.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=9, adjust=False).mean()
            
            mas = [ma5, ma20, ma60]
            max_ma, min_ma = max(mas), min(mas)
            spread = (max_ma - min_ma) / min_ma
            
            if spread < 0.02:
                trend_name = "均线粘连"
                trend_desc = "面临方向性变盘选择，资金观望情绪浓厚"
            elif ma5 > ma20 > ma60:
                trend_name = "三线开花(强势多头)" if cl.iloc[-1] > ma5 else "多头排列(短期回踩)"
                trend_desc = "全面多头排列，上行动能极强，顺势做多" if cl.iloc[-1] > ma5 else "大趋势向上但短期回踩，关注下方均线支撑"
            elif ma5 < ma20 < ma60:
                trend_name = "空头瀑布(极度弱势)" if cl.iloc[-1] < ma5 else "空头排列(超跌反弹)"
                trend_desc = "全面空头排列，下行趋势加速，严控仓位" if cl.iloc[-1] < ma5 else "大级别处于下降通道，当前属于超跌反弹"
            else:
                trend_name = "震荡分化"
                trend_desc = "长短均线方向不一，无明显单边趋势"
            
            macd_status = "MACD死叉" if macd.iloc[-1] < signal_line.iloc[-1] else "MACD金叉"
            
            self.briefing_data['market_analysis']['index'] = {
                'code': index_code,
                'name': '上证指数',
                'close': round(cl.iloc[-1], 2),
                'change': round(pct, 2),
                'ma5': round(ma5, 2),
                'ma20': round(ma20, 2),
                'ma60': round(ma60, 2),
                'trend': trend_name,
                'macd': macd_status
            }
            
            return {
                'close': cl.iloc[-1],
                'pct': pct,
                'ma5': ma5,
                'ma20': ma20,
                'ma60': ma60,
                'trend_name': trend_name,
                'trend_desc': trend_desc,
                'macd_status': macd_status
            }
        except Exception as e:
            return {'error': str(e)}
    
    def fetch_northbound_flow(self) -> Dict:
        """获取北向资金数据"""
        try:
            import akshare as ak
            
            df = ak.stock_hk_spot_em()
            if df.empty:
                df = ak.stock_hk_spot()
            
            north_flow = 0
            if not df.empty:
                for _, row in df.iterrows():
                    if '北向资金' in str(row.get('名称', '')) or '北向' in str(row.get('名称', '')):
                        north_flow = float(row.get('最新值', 0))
                        break
            
            if north_flow == 0:
                try:
                    df_north = ak.stock_em_hsgt_north_net_flow_institution()
                    if not df_north.empty:
                        north_flow = float(df_north.iloc[0].get('净流入', 0))
                except:
                    pass
            
            if north_flow >= 50:
                flow_status = "🚀(外资抢筹)"
            elif north_flow >= 10:
                flow_status = "📈(外资流入)"
            elif north_flow >= 0:
                flow_status = "😐(小幅流入)"
            elif north_flow >= -10:
                flow_status = "😐(小幅流出)"
            elif north_flow >= -50:
                flow_status = "📉(外资流出)"
            else:
                flow_status = "⚠️(外资出逃)"
            
            self.briefing_data['market_analysis']['north_flow'] = {
                'value': round(north_flow, 2),
                'status': flow_status
            }
            
            return {
                'flow': north_flow,
                'msg': f"- **北向资金**：`{north_flow:.2f}` 亿元 {flow_status}"
            }
        except Exception as e:
            return {'flow': 0, 'msg': f"北向资金获取失败: {e}"}
    
    def fetch_hot_sectors(self) -> List[Dict]:
        """获取行业热点数据"""
        try:
            import akshare as ak
            
            df = ak.stock_zh_a_sector_spot_em()
            if df.empty:
                df = ak.stock_zh_a_sector_fund_flow()
            
            if df.empty:
                return []
            
            df = df.sort_values('涨跌幅', ascending=False)
            hot_sectors = []
            
            name_col = '板块名称' if '板块名称' in df.columns else '名称' if '名称' in df.columns else None
            pct_col = '涨跌幅' if '涨跌幅' in df.columns else None
            lead_col = '领涨股' if '领涨股' in df.columns else '代码' if '代码' in df.columns else None
            
            for _, row in df.head(10).iterrows():
                sector = {
                    'name': row[name_col] if name_col else '',
                    'pct': round(float(row[pct_col]), 2) if pct_col else 0,
                    'lead_stock': row[lead_col] if lead_col else '',
                    'amount': row.get('成交额', row.get('amount', '')),
                }
                hot_sectors.append(sector)
            
            self.briefing_data['hot_sectors'] = hot_sectors
            return hot_sectors
        except Exception as e:
            return []
    
    def fetch_etf_data(self) -> List[Dict]:
        """获取ETF轮动数据"""
        try:
            import akshare as ak
            df = ak.fund_etf_spot_em()
            if df.empty:
                return []
            
            df = df[(df['代码'].str.startswith(('51', '15', '588', '56'))) & (df['成交额'] > 1000)]
            df = df.sort_values('涨跌幅', ascending=False)
            
            etf_list = []
            for _, row in df.head(15).iterrows():
                etf = {
                    'code': row['代码'],
                    'name': row['名称'],
                    'price': round(float(row['最新价']), 3),
                    'pct': round(float(row['涨跌幅']), 2),
                    'amount': row.get('成交额', ''),
                    'volume': row.get('成交量', '')
                }
                etf_list.append(etf)
            
            self.briefing_data['etf_rotation'] = etf_list
            return etf_list
        except Exception as e:
            return []
    
    def fetch_stock_signals(self) -> Dict:
        """获取股票信号 - 使用轻量级筛选"""
        try:
            import akshare as ak
            
            df = ak.stock_zh_a_spot_em()
            if df.empty:
                df = ak.stock_zh_a_spot()
            
            if df.empty:
                return {'error': '行情数据获取失败'}
            
            pct_col = None
            for col in ['涨跌幅', 'change', 'pct']:
                if col in df.columns:
                    pct_col = col
                    break
            
            if pct_col is None:
                return {'error': '未找到涨跌幅列'}
            
            df[pct_col] = pd.to_numeric(df[pct_col], errors='coerce')
            df = df[df[pct_col] > 0]
            df = df.sort_values(pct_col, ascending=False)
            
            code_col = None
            for col in ['代码', 'code', 'symbol']:
                if col in df.columns:
                    code_col = col
                    break
            
            name_col = None
            for col in ['名称', 'name', 'stock_name']:
                if col in df.columns:
                    name_col = col
                    break
            
            price_col = None
            for col in ['最新价', 'close', 'price', '收盘']:
                if col in df.columns:
                    price_col = col
                    break
            
            if code_col is None or name_col is None:
                return {'error': '未找到代码或名称列'}
            
            core_stocks = []
            for _, row in df.head(3).iterrows():
                price = round(float(row[price_col]), 2) if price_col else 0
                core_stocks.append({
                    'code': str(row[code_col]).zfill(6),
                    'name': row[name_col],
                    'price': price,
                    'score': 80,
                    'level': 'A'
                })
            
            satellite_stocks = []
            for _, row in df.iloc[3:10].iterrows():
                price = round(float(row[price_col]), 2) if price_col else 0
                satellite_stocks.append({
                    'code': str(row[code_col]).zfill(6),
                    'name': row[name_col],
                    'price': price,
                    'score': 70,
                    'level': 'B'
                })
            
            watchlist = []
            for _, row in df.iloc[10:13].iterrows():
                price = round(float(row[price_col]), 2) if price_col else 0
                watchlist.append([
                    row[name_col],
                    str(row[code_col]).zfill(6),
                    65,
                    price
                ])
            
            self.briefing_data['stock_signals'] = {
                'core': core_stocks,
                'satellite': satellite_stocks,
                'watchlist': watchlist,
                'total_pool': len(df),
                'total_market': len(df),
                'market_msg': ''
            }
            
            return {'core': core_stocks, 'satellite': satellite_stocks}
        except Exception as e:
            return {'error': str(e)}
    
    def generate_briefing_content(self) -> str:
        """生成完整的简报内容"""
        content = f"## 📊 每日投研简报\n> **{self.now_str}**\n\n"
        
        content += "---\n\n"
        
        content += self.generate_macro_section()
        
        content += "\n---\n\n"
        
        content += "### 📈 A股市场概况\n"
        market = self.briefing_data['market_analysis']
        
        if market.get('total_stocks') and market['total_stocks'] > 0:
            content += (
                f"- **市场广度**：红盘 `{market['up_count']}` 家 / 绿盘 `{market['down_count']}` 家\n"
                f"- **涨跌停**：涨停 `{market['zt_count']}` 家 / 跌停 `{market['dt_count']}` 家\n"
                f"- **两市量能**：约 `{market['total_amount']}` 亿元\n"
            )
        
        if market.get('index'):
            idx = market['index']
            content += (
                f"- **{idx['name']}**：`{idx['close']}` (今日 **{idx['change']:+.2f}%**)\n"
                f"- **均线趋势**：`{idx['trend']}`\n"
                f"- **MACD状态**：`{idx['macd']}`\n"
            )
        
        if market.get('north_flow'):
            nf = market['north_flow']
            content += f"- **北向资金**：`{nf['value']}` 亿元 {nf['status']}\n"
        
        if not market.get('up_count') and not market.get('index'):
            content += "暂无市场数据\n"
        
        content += "\n---\n\n"
        
        content += "### 🔥 行业热点追踪\n"
        if self.briefing_data['hot_sectors']:
            for sector in self.briefing_data['hot_sectors'][:5]:
                emoji = "🚀" if sector['pct'] > 5 else "📈" if sector['pct'] > 0 else "📉"
                content += f"{emoji} **{sector['name']}** ({sector['pct']:+.2f}%) - 领涨: {sector['lead_stock']}\n"
        else:
            content += "暂无行业热点数据\n"
        
        content += "\n---\n\n"
        
        content += "### 🛡️ ETF轮动信号\n"
        if self.briefing_data['etf_rotation']:
            content += "| 代码 | 名称 | 价格 | 涨跌幅 |\n"
            content += "|------|------|------|--------|\n"
            for etf in self.briefing_data['etf_rotation'][:10]:
                content += f"| `{etf['code']}` | {etf['name']} | {etf['price']} | {etf['pct']:+.2f}% |\n"
        else:
            content += "暂无ETF数据\n"
        
        content += "\n---\n\n"
        
        content += "### 🎯 股票信号精选\n"
        signals = self.briefing_data['stock_signals']
        
        if signals.get('error'):
            content += f"⚠️ 信号获取失败: {signals['error']}\n"
        elif signals.get('core'):
            content += "#### 🔥 核心主力池\n"
            for s in signals['core']:
                content += f"- `{s['code']}` **{s['name']}** - 评级: {s['level']} ({s['score']}分)\n"
        elif signals.get('satellite'):
            content += "#### 🛰️ 卫星观察池\n"
            for s in signals['satellite'][:5]:
                content += f"- `{s['code']}` **{s['name']}** - 评级: {s['level']} ({s['score']}分)\n"
        elif signals.get('watchlist'):
            content += "#### 👁️ 候补观察池\n"
            for name, code, score, price in signals['watchlist'][:3]:
                content += f"- `{code}` **{name}** (¥{price}) - 得分: {score}\n"
        else:
            content += "今日未发现符合条件的信号\n"
        
        if signals.get('market_msg'):
            content += f"\n> 📊 {signals['market_msg'][:200]}...\n"
        
        content += "\n---\n\n"
        content += "> ⚠️ **风险提示**：以上内容仅供投研参考，不构成投资建议。投资有风险，入市需谨慎。\n"
        
        return content
    
    def send_dingtalk_notification(self, content: str) -> bool:
        """发送钉钉通知"""
        if not self.dingtalk_webhook:
            print("⚠️ 未配置钉钉WEBHOOK，跳过推送")
            return False
        
        try:
            headers = {'Content-Type': 'application/json'}
            
            payload = {
                'msgtype': 'markdown',
                'markdown': {
                    'title': f'每日投研简报 {self.date_str}',
                    'text': content
                }
            }
            
            response = requests.post(
                self.dingtalk_webhook,
                headers=headers,
                data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                timeout=15
            )
            
            result = response.json()
            if result.get('errcode') == 0:
                print("✅ 钉钉通知发送成功")
                return True
            else:
                print(f"❌ 钉钉通知发送失败: {result}")
                return False
        except Exception as e:
            print(f"❌ 钉钉通知发送异常: {e}")
            return False
    
    def run(self):
        """执行完整的简报流程"""
        print(f"🚀 开始生成每日投研简报 {self.now_str}")
        
        print("📊 获取市场概况...")
        self.fetch_a_share_market()
        
        print("📈 获取指数数据...")
        self.fetch_index_data()
        
        print("💹 获取北向资金...")
        self.fetch_northbound_flow()
        
        print("🔥 获取行业热点...")
        self.fetch_hot_sectors()
        
        print("🛡️ 获取ETF轮动数据...")
        self.fetch_etf_data()
        
        print("🎯 获取股票信号...")
        self.fetch_stock_signals()
        
        print("📝 生成简报内容...")
        content = self.generate_briefing_content()
        
        print("\n" + "="*80)
        print(content)
        print("="*80 + "\n")
        
        print("📤 发送钉钉通知...")
        self.send_dingtalk_notification(content)
        
        print("✅ 每日投研简报生成完成")

if __name__ == '__main__':
    briefing = DailyBriefing()
    briefing.run()