import os
import sys
import time
import logging
import threading
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)


class TimeoutException(Exception):
    pass


def run_with_timeout(func, timeout=30, default=None):
    result = [default]
    exception = [None]
    
    def target():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        log.warning(f"⚠️ 操作超时 ({timeout}s): {func.__name__}")
        return default
    
    if exception[0] is not None:
        raise exception[0]
    
    return result[0]


from main import (
    config, TZ_BJS, NotificationGateway, get_signals, send_dingtalk,
    generate_macro_section, fetch_hot_sectors, fetch_spot, fetch_index,
    extract_market_context, Config, Cols, _DATA_PROXY,
    _today_str, load_pushed_state, save_pushed_state
)

C = Cols()


def fetch_etf_rotation():
    try:
        import akshare as ak
        etf_codes = {
            '510300': '沪深300ETF',
            '510500': '中证500ETF',
            '159915': '创业板ETF',
            '510050': '上证50ETF',
            '512880': '证券ETF',
            '512480': '券商ETF',
            '512010': '医药ETF',
            '512690': '酒ETF',
            '513100': '纳指ETF',
            '159995': '芯片ETF',
        }
        
        etf_data = []
        for code, name in etf_codes.items():
            try:
                df = run_with_timeout(
                    lambda c=code: ak.fund_etf_hist_em(
                        symbol=f"{c}.SH" if c.startswith('5') else f"{c}.SZ", 
                        period="daily", 
                        start_date=(datetime.now(TZ_BJS) - timedelta(days=60)).strftime('%Y%m%d')
                    ),
                    timeout=10
                )
                
                if df is not None and not df.empty:
                    df['日期'] = pd.to_datetime(df['日期'])
                    df = df.sort_values('日期')
                    
                    close = df['收盘'].astype(float)
                    volume = df['成交量'].astype(float)
                    
                    if len(close) >= 20:
                        ma5 = close.rolling(5).mean().iloc[-1]
                        ma20 = close.rolling(20).mean().iloc[-1]
                        
                        pct_chg = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100 if len(close) >= 2 else 0
                        vol_ratio = volume.iloc[-1] / volume.iloc[-5:].mean() if len(volume) >= 5 else 1
                        
                        trend = '📈 上升' if close.iloc[-1] > ma20 else '📉 下降'
                        
                        etf_data.append({
                            'code': code,
                            'name': name,
                            'price': round(close.iloc[-1], 2),
                            'pct_chg': round(pct_chg, 2),
                            'vol_ratio': round(vol_ratio, 2),
                            'trend': trend,
                        })
            except Exception as e:
                log.warning(f"获取 ETF {code} 数据失败: {e}")
        
        etf_data.sort(key=lambda x: x['pct_chg'], reverse=True)
        return etf_data
    except Exception as e:
        log.error(f"获取 ETF 轮动数据失败: {e}")
        return []


def fetch_sector_hotspots():
    try:
        hot_map = run_with_timeout(fetch_hot_sectors, timeout=30)
        
        if hot_map:
            sector_counts = {}
            for code, sector in hot_map.items():
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            top_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            
            import akshare as ak
            sector_details = []
            for sector_name, count in top_sectors:
                try:
                    df = run_with_timeout(
                        lambda s=sector_name: ak.stock_board_industry_cons_em(symbol=s),
                        timeout=10
                    )
                    
                    if df is not None and not df.empty:
                        pct_col = next((c for c in df.columns if '涨跌幅' in c), None)
                        if pct_col:
                            avg_pct = df[pct_col].astype(float).mean()
                            leader = df.iloc[0]
                            code_col = next((c for c in df.columns if '代码' in c), None)
                            name_col = next((c for c in df.columns if '名称' in c), None)
                            leader_code = str(leader[code_col]) if code_col else ''
                            leader_name = leader[name_col] if name_col else ''
                            leader_pct = float(leader[pct_col]) if pct_col else 0
                            
                            sector_details.append({
                                'name': sector_name,
                                'stock_count': count,
                                'avg_pct': round(avg_pct, 2),
                                'leader_code': leader_code,
                                'leader_name': leader_name,
                                'leader_pct': round(leader_pct, 2),
                            })
                except Exception as e:
                    log.warning(f"获取板块 {sector_name} 详情失败: {e}")
                    sector_details.append({
                        'name': sector_name,
                        'stock_count': count,
                        'avg_pct': 0,
                        'leader_code': '',
                        'leader_name': '',
                        'leader_pct': 0,
                    })
            
            return sector_details
    except Exception as e:
        log.error(f"获取行业热点数据失败: {e}")
    
    try:
        import akshare as ak
        df = run_with_timeout(lambda: ak.stock_board_industry_name_em(), timeout=15)
        
        if df is not None and not df.empty:
            df['涨跌幅'] = pd.to_numeric(df['涨跌幅'], errors='coerce')
            top_sectors = df.nlargest(8, '涨跌幅')
            
            sector_details = []
            for _, row in top_sectors.iterrows():
                sector_name = row['板块名称']
                pct = round(float(row['涨跌幅']), 2)
                sector_details.append({
                    'name': sector_name,
                    'stock_count': 0,
                    'avg_pct': pct,
                    'leader_code': '',
                    'leader_name': '',
                    'leader_pct': pct,
                })
            
            return sector_details
    except Exception as e:
        log.error(f"备用接口获取行业热点失败: {e}")
    
    return []


def generate_briefing():
    now_ts = datetime.now(TZ_BJS)
    now_str = now_ts.strftime('%Y-%m-%d %H:%M')
    today_str = now_ts.strftime('%Y-%m-%d')
    
    briefing = f"## 📊 每日投研简报\n> **{today_str} {now_ts.strftime('%H:%M')}**\n\n"
    
    briefing += "---\n\n"
    briefing += "## 🌍 宏观与外围市场\n"
    macro_content = run_with_timeout(generate_macro_section, timeout=30, default="### 🌍 隔夜外围与宏观指标快报\n⚠️ 外围数据获取超时（网络受限）")
    briefing += macro_content + "\n\n"
    
    briefing += "---\n\n"
    briefing += "## 📈 A股市场全景分析\n"
    
    try:
        df_raw = run_with_timeout(fetch_spot, timeout=60)
        if df_raw is None:
            raise TimeoutException("fetch_spot timeout")
            
        c_conf = Config()
        df_clean, market_ok, market_msg, index_ret, market_overheated, market_regime, vol_surge, market_temp = extract_market_context(df_raw, c_conf)
        briefing += f"{market_msg}\n\n"
    except TimeoutException:
        log.error("市场分析数据获取超时")
        briefing += "⚠️ 市场分析数据获取超时（网络受限）\n\n"
    except Exception as e:
        log.error(f"获取市场分析失败: {e}")
        briefing += f"⚠️ 市场分析数据获取失败: {e}\n\n"
    
    briefing += "---\n\n"
    briefing += "## 🔥 行业热点追踪\n"
    sector_hotspots = fetch_sector_hotspots()
    if sector_hotspots:
        for i, sector in enumerate(sector_hotspots):
            pct_sign = '+' if sector['avg_pct'] > 0 else ''
            leader_info = f" | 领涨: {sector['leader_name']}({sector['leader_code']}) {pct_sign}{sector['leader_pct']}%" if sector['leader_name'] else ''
            briefing += f"{i+1}. **{sector['name']}** {pct_sign}{sector['avg_pct']}% {leader_info}\n"
        briefing += "\n"
    else:
        briefing += "⚠️ 行业热点数据获取失败\n\n"
    
    briefing += "---\n\n"
    briefing += "## 🧩 ETF轮动监控\n"
    etf_rotation = fetch_etf_rotation()
    if etf_rotation:
        briefing += "| ETF名称 | 代码 | 现价 | 涨跌幅 | 量比 | 趋势 |\n"
        briefing += "|---------|------|------|--------|------|------|\n"
        for etf in etf_rotation:
            pct_sign = '+' if etf['pct_chg'] > 0 else ''
            briefing += f"| {etf['name']} | `{etf['code']}` | ¥{etf['price']} | {pct_sign}{etf['pct_chg']}% | {etf['vol_ratio']} | {etf['trend']} |\n"
        briefing += "\n"
    else:
        briefing += "⚠️ ETF轮动数据获取失败\n\n"
    
    briefing += "---\n\n"
    briefing += "## 🎯 精选股票信号\n"
    
    try:
        signals_result = run_with_timeout(get_signals, timeout=300)
        if signals_result is None:
            raise TimeoutException("get_signals timeout")
        
        signals, watchlist, pushed, pool_size, m_msg, total_market = signals_result
        
        if signals and any(signals.values()):
            total_signals = sum(len(sigs) for sigs in signals.values())
            briefing += f"**扫描结果**: 全市场 `{total_market}` 只，筛选 `{pool_size}` 只，信号 `{total_signals}` 只\n\n"
            
            if signals.get('Core'):
                briefing += "### 🔥 核心主力池\n"
                for s in signals['Core']:
                    warn_msg = " ⚡ 创业板" if str(s.code).startswith('300') else ""
                    briefing += f"- **{s.name}** ({s.code}){warn_msg}: ¥{s.price} ({s.pct_chg}) | 评分: **{s.score:.1f}**\n"
                    briefing += f"  止损: ¥{s.stop_loss} | 目标: ¥{s.target1}\n\n"
            
            if signals.get('Satellite'):
                briefing += "### 🛰️ 卫星观察池\n"
                for s in signals['Satellite'][:5]:
                    briefing += f"- **{s.name}** ({s.code}): ¥{s.price} ({s.pct_chg}) | 评分: **{s.score:.1f}**\n"
                if len(signals['Satellite']) > 5:
                    briefing += f"  ... 还有 {len(signals['Satellite']) - 5} 只\n"
                briefing += "\n"
        else:
            briefing += "✅ 今日未发现符合条件的信号，建议保持观望\n\n"
        
        if watchlist:
            briefing += "### 👁️ 候补观察池\n"
            for name, code, score, price in watchlist[:3]:
                briefing += f"- **{name}** ({code}): ¥{price} | 评分: {score}\n"
            briefing += "\n"
    except TimeoutException:
        log.error("股票信号获取超时")
        briefing += "⚠️ 股票信号获取超时（网络受限，跳过全量扫描）\n\n"
    except Exception as e:
        log.error(f"获取股票信号失败: {e}")
        briefing += f"⚠️ 股票信号获取失败: {e}\n\n"
    
    briefing += "---\n\n"
    briefing += "## 💡 操作建议\n"
    briefing += "> 📌 以上内容仅供参考，不构成投资建议。投资有风险，入市需谨慎。\n"
    briefing += f"> 🤖 生成时间: {now_str} | Hermes AI量化系统"
    
    return briefing


def main():
    try:
        log.info("🚀 每日投研简报生成引擎启动...")
        
        briefing = generate_briefing()
        
        log.info(f"📝 简报生成完成，共 {len(briefing)} 字符")
        
        if config.DINGTALK_WEBHOOK:
            NotificationGateway.send('📊 每日投研简报', briefing, template='blue')
            log.info("✅ 钉钉通知发送成功")
        else:
            log.warning("⚠️ 未配置 DINGTALK_WEBHOOK，跳过钉钉推送")
        
        print("\n" + "="*80)
        print(briefing)
        print("="*80 + "\n")
        
    except Exception as e:
        log.critical(f"每日投研简报生成失败: {e}", exc_info=True)
        error_msg = f"🚨 **每日投研简报生成失败**\n\n**时间**: {_today_str()}\n**异常**: {str(e)[:300]}..."
        if config.DINGTALK_WEBHOOK:
            NotificationGateway.send("🚨 投研简报告警", error_msg, template="red")
    finally:
        try:
            _DATA_PROXY.cleanup()
        except:
            pass


if __name__ == '__main__':
    main()