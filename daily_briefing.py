import os
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
os.environ['NO_PROXY'] = '*'

import json
import logging
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

import sys
sys.path.insert(0, '/workspace')

from main import (
    AppConfig, NotificationGateway, fetch_spot, fetch_index, fetch_hot_sectors,
    fetch_northbound_flow, extract_market_context, Config,
    fetch_core_pool, get_signals, send_dingtalk, save_pushed_state, _today_str, TZ_BJS, Cols
)

config = AppConfig()
C = Cols()

def generate_macro_section_fallback():
    """生成宏观数据报告（离线模式）"""
    return "### 🌍 隔夜外围与宏观风控快报\n⚠️ 外部数据接口暂不可用，已切换至离线模式。建议关注国内政策面和资金面变化。"

def safe_call_with_timeout(func, timeout=10, fallback=None, *args, **kwargs):
    """带超时的安全调用"""
    import threading
    
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        log.warning(f"调用 {func.__name__} 超时，使用备选方案")
        return fallback
    
    if exception[0] is not None:
        log.warning(f"调用 {func.__name__} 失败: {exception[0]}")
        return fallback
    
    return result[0]

class DailyBriefing:
    def __init__(self):
        self.now = datetime.now(TZ_BJS)
        self.today_str = self.now.strftime('%Y-%m-%d')
        self.briefing_content = ""
        self.use_fallback = False
    
    def generate_market_analysis(self):
        """生成市场分析报告"""
        log.info("📊 正在生成市场分析报告...")
        try:
            df_raw = safe_call_with_timeout(fetch_spot, timeout=60, fallback=None)
            if df_raw is None:
                raise Exception("无法获取行情数据")
            
            c_conf = Config()
            df_clean, market_ok, market_msg, idx_ret, market_overheated, market_regime, vol_surge = extract_market_context(df_raw, c_conf)
            return market_msg
        except Exception as e:
            log.error(f"生成市场分析失败: {e}")
            self.use_fallback = True
            return f"### 📊 A股深度诊断\n⚠️ 市场分析获取失败: {str(e)}\n\n> 当前处于离线模式，建议参考本地缓存数据进行分析。"
    
    def generate_stock_signals(self):
        """生成股票信号报告"""
        log.info("🎯 正在生成股票信号报告...")
        try:
            result = safe_call_with_timeout(get_signals, timeout=120, fallback=None)
            
            if result is None:
                return "\n### 📈 股票信号\n无法获取股票信号数据。"
            
            sigs, watch, pushed, pool_size, m_msg, total_mkt = result
            
            if sigs is None or (isinstance(sigs, list) and len(sigs) == 0) or (isinstance(sigs, dict) and not any(sigs.values())):
                return "\n### 📈 股票信号\n今日未发现符合条件的股票信号，建议观望。"
            
            content = "\n### 📈 股票信号\n"
            for category, signals in sigs.items():
                content += f"#### {category} ({len(signals)}只)\n"
                for sig in signals[:5]:
                    content += f"- **{sig.name}** (`{sig.code}`): 评分 {sig.score:.1f}分 | 现价 ¥{sig.price} | {sig.level.split(']')[0]}]\n"
                    content += f"  止损: ¥{sig.stop_loss} | 目标: ¥{sig.target1}\n\n"
            
            return content
        except Exception as e:
            log.error(f"生成股票信号失败: {e}")
            return f"\n### 📈 股票信号\n⚠️ 获取失败: {str(e)}"
    
    def generate_etf_rotation(self):
        """生成ETF轮动报告"""
        log.info("🔄 正在生成ETF轮动报告...")
        try:
            df_raw = safe_call_with_timeout(fetch_spot, timeout=60, fallback=None)
            if df_raw is None:
                return "\n### 📦 ETF轮动\n无法获取行情数据。"
            
            etf_mask = df_raw[C.S_CODE].astype(str).str.startswith(('51', '15', '588', '56'))
            etf_df = df_raw[etf_mask].copy()
            
            if etf_df.empty:
                return "\n### 📦 ETF轮动\n今日无ETF数据。"
            
            etf_df[C.S_PCT] = pd.to_numeric(etf_df[C.S_PCT], errors='coerce')
            
            top_etf = etf_df.nlargest(5, C.S_PCT)
            bottom_etf = etf_df.nsmallest(3, C.S_PCT)
            
            content = "\n### 📦 ETF轮动\n"
            content += "#### 🏆 领涨ETF\n"
            for _, row in top_etf.iterrows():
                content += f"- **{row[C.S_NAME]}** ({row[C.S_CODE]}): +{row[C.S_PCT]:.2f}%\n"
            
            content += "\n#### 📉 领跌ETF\n"
            for _, row in bottom_etf.iterrows():
                content += f"- **{row[C.S_NAME]}** ({row[C.S_CODE]}): {row[C.S_PCT]:.2f}%\n"
            
            return content
        except Exception as e:
            log.error(f"生成ETF轮动失败: {e}")
            return f"\n### 📦 ETF轮动\n⚠️ 获取失败: {str(e)}"
    
    def generate_hot_sectors(self):
        """生成行业热点报告"""
        log.info("🌋 正在生成行业热点报告...")
        try:
            hot_map = safe_call_with_timeout(fetch_hot_sectors, timeout=30, fallback=None)
            if not hot_map:
                return "\n### 🔥 行业热点\n今日无热点板块数据。"
            
            from collections import Counter
            sec_counts = Counter(hot_map.values())
            top_sectors = sec_counts.most_common(8)
            
            content = "\n### 🔥 行业热点\n"
            for sector, count in top_sectors:
                content += f"- **{sector}**: {count}只成分股\n"
            
            content += "\n> 注：以上为今日涨幅居前的热点板块，可关注板块内强势个股。"
            return content
        except Exception as e:
            log.error(f"生成行业热点失败: {e}")
            return f"\n### 🔥 行业热点\n⚠️ 获取失败: {str(e)}"
    
    def generate_macro_section(self):
        """生成宏观数据报告"""
        log.info("🌍 正在生成宏观数据报告...")
        return generate_macro_section_fallback()
    
    def generate_briefing(self):
        """生成完整的每日投研简报"""
        log.info("📝 正在生成完整的每日投研简报...")
        
        sections = [
            self.generate_macro_section(),
            self.generate_market_analysis(),
            self.generate_hot_sectors(),
            self.generate_etf_rotation(),
            self.generate_stock_signals()
        ]
        
        header = f"## 📊 每日投研简报\n> **{self.now.strftime('%Y-%m-%d %H:%M')}**\n\n"
        footer = "\n---\n\n> 🤖 本报告由AI量化系统自动生成，仅供参考，不构成投资建议。"
        
        self.briefing_content = header + "\n\n".join(sections) + footer
        return self.briefing_content
    
    def send_to_dingtalk(self):
        """发送简报到钉钉"""
        log.info("📤 正在发送简报到钉钉...")
        try:
            NotificationGateway.send('📊 每日投研简报', self.briefing_content)
            log.info("✅ 简报发送成功！")
            return True
        except Exception as e:
            log.error(f"发送钉钉失败: {e}")
            return False

def main():
    log.info("🚀 启动每日投研简报生成系统...")
    
    briefing = DailyBriefing()
    
    try:
        content = briefing.generate_briefing()
        print(content)
        
        if config.DINGTALK_WEBHOOK:
            briefing.send_to_dingtalk()
        else:
            log.warning("⚠️ 未配置钉钉Webhook，仅打印到控制台")
            
    except Exception as e:
        log.error(f"生成简报失败: {e}", exc_info=True)
        error_msg = f"🚨 **每日投研简报生成失败**\n\n**时间**: {_today_str()}\n**异常信息**: {str(e)[:300]}..."
        try:
            NotificationGateway.send("🚨 简报生成失败", error_msg, template="red")
        except:
            pass

if __name__ == '__main__':
    main()