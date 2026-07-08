#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日投研简报生成器
- 市场分析（大盘趋势、市场广度、北向资金、外围市场）
- 股票信号（核心池、观察池）
- ETF轮动（宽基ETF、行业ETF强弱分析）
- 行业热点（领涨板块、资金流向）
- 钉钉推送
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from collections import Counter

import numpy as np
import pandas as pd
import requests
import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    AppConfig, Config, Cols, Signal, NotificationGateway,
    fetch_spot, fetch_index, fetch_hist, fetch_hot_sectors, fetch_northbound_flow,
    extract_market_context, generate_macro_section, get_ma_trend,
    AShareTechnicals, apply_scoring, vectorized_prescreen,
    TZ_BJS, C, log, config
)

ETF_UNIVERSE = {
    '宽基指数': [
        {'code': '510300', 'name': '沪深300ETF', 'index': 'sh000300'},
        {'code': '510500', 'name': '中证500ETF', 'index': 'sh000905'},
        {'code': '510850', 'name': '中证A500ETF', 'index': 'sh000852'},
        {'code': '159915', 'name': '创业板ETF', 'index': 'sz399006'},
        {'code': '510180', 'name': '上证180ETF', 'index': 'sh000010'},
        {'code': '510050', 'name': '上证50ETF', 'index': 'sh000016'},
        {'code': '159919', 'name': '沪深300ETF(易方达)', 'index': 'sh000300'},
        {'code': '588000', 'name': '科创50ETF', 'index': 'sh000688'},
    ],
    '行业主题': [
        {'code': '512480', 'name': '半导体ETF', 'index': ''},
        {'code': '515030', 'name': '新能源车ETF', 'index': ''},
        {'code': '512690', 'name': '酒ETF', 'index': ''},
        {'code': '512170', 'name': '医疗ETF', 'index': ''},
        {'code': '512880', 'name': '证券ETF', 'index': ''},
        {'code': '512200', 'name': '房地产ETF', 'index': ''},
        {'code': '512660', 'name': '军工ETF', 'index': ''},
        {'code': '159995', 'name': '芯片ETF', 'index': ''},
        {'code': '515790', 'name': '光伏ETF', 'index': ''},
        {'code': '516160', 'name': '新能源ETF', 'index': ''},
        {'code': '512010', 'name': '医药ETF', 'index': ''},
        {'code': '515050', 'name': '5GETF', 'index': ''},
        {'code': '512580', 'name': '环保ETF', 'index': ''},
        {'code': '512980', 'name': '传媒ETF', 'index': ''},
        {'code': '512680', 'name': '银行ETF', 'index': ''},
        {'code': '512070', 'name': '非银ETF', 'index': ''},
    ],
    '大宗商品/海外': [
        {'code': '518880', 'name': '黄金ETF', 'index': ''},
        {'code': '159985', 'name': '豆粕ETF', 'index': ''},
        {'code': '513100', 'name': '纳指ETF', 'index': ''},
        {'code': '513500', 'name': '标普500ETF', 'index': ''},
        {'code': '513030', 'name': '德国DAX', 'index': ''},
        {'code': '159941', 'name': '纳指ETF(易方达)', 'index': ''},
    ],
}

class ETFRotationAnalyzer:
    """ETF轮动分析师"""
    
    def __init__(self):
        self.now = datetime.now(TZ_BJS)
    
    def _get_etf_spot_data(self) -> pd.DataFrame:
        """获取ETF实时行情数据"""
        try:
            df_raw = fetch_spot()
            if df_raw is None or df_raw.empty:
                return pd.DataFrame()
            
            all_etf_codes = []
            for category, etfs in ETF_UNIVERSE.items():
                for etf in etfs:
                    all_etf_codes.append(etf['code'])
            
            etf_df = df_raw[df_raw[C.S_CODE].astype(str).str.zfill(6).isin(all_etf_codes)].copy()
            return etf_df
        except Exception as e:
            log.warning(f"获取ETF行情失败: {e}")
            return pd.DataFrame()
    
    def _calc_etf_momentum(self, code: str, name: str) -> dict:
        """计算单只ETF的动量指标"""
        try:
            end_s = self.now.strftime('%Y%m%d')
            start_s = (self.now - timedelta(days=120)).strftime('%Y%m%d')
            hist = fetch_hist(code, start_s, end_s)
            
            if hist is None or len(hist) < 20:
                return None
            
            close = hist[C.H_CLOSE].astype(float)
            
            ret_1d = (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close) >= 2 else 0
            ret_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
            ret_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0
            ret_60d = (close.iloc[-1] / close.iloc[-61] - 1) * 100 if len(close) >= 61 else 0
            
            ma5 = close.rolling(5).mean().iloc[-1]
            ma20 = close.rolling(20).mean().iloc[-1]
            ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20
            
            ma_trend = "多头" if (ma5 > ma20 > ma60) else "空头" if (ma5 < ma20 < ma60) else "震荡"
            
            vol = hist[C.H_VOL].astype(float)
            vol_ratio = vol.iloc[-1] / vol.rolling(20).mean().iloc[-1] if len(vol) >= 20 else 1.0
            
            momentum_score = (
                ret_1d * 0.2 + 
                ret_5d * 0.3 + 
                ret_20d * 0.3 + 
                ret_60d * 0.2
            )
            
            if ma_trend == "多头":
                momentum_score += 5
            elif ma_trend == "空头":
                momentum_score -= 5
            
            return {
                'code': code,
                'name': name,
                'price': float(close.iloc[-1]),
                'ret_1d': ret_1d,
                'ret_5d': ret_5d,
                'ret_20d': ret_20d,
                'ret_60d': ret_60d,
                'ma_trend': ma_trend,
                'vol_ratio': vol_ratio,
                'momentum_score': momentum_score,
            }
        except Exception as e:
            log.debug(f"计算ETF {code} 动量失败: {e}")
            return None
    
    def analyze_rotation(self) -> dict:
        """分析ETF轮动情况"""
        log.info("🔄 开始分析ETF轮动...")
        
        all_results = []
        
        for category, etfs in ETF_UNIVERSE.items():
            for etf in etfs:
                result = self._calc_etf_momentum(etf['code'], etf['name'])
                if result:
                    result['category'] = category
                    all_results.append(result)
        
        if not all_results:
            return {'strong': [], 'weak': [], 'by_category': {}}
        
        df = pd.DataFrame(all_results)
        df = df.sort_values('momentum_score', ascending=False)
        
        strong = df.head(5).to_dict('records')
        weak = df.tail(5).iloc[::-1].to_dict('records')
        
        by_category = {}
        for cat in ETF_UNIVERSE.keys():
            cat_df = df[df['category'] == cat]
            if not cat_df.empty:
                by_category[cat] = cat_df.sort_values('momentum_score', ascending=False).to_dict('records')
        
        log.info(f"✅ ETF轮动分析完成，强势ETF: {[e['name'] for e in strong]}")
        
        return {
            'strong': strong,
            'weak': weak,
            'by_category': by_category,
            'all': df.to_dict('records'),
        }
    
    def format_rotation_report(self, rotation_data: dict) -> str:
        """格式化ETF轮动报告"""
        if not rotation_data.get('strong'):
            return "### 🔄 ETF轮动分析\n⚠️ ETF数据获取失败，暂无法提供轮动分析。\n"
        
        lines = ["### 🔄 ETF轮动分析\n"]
        
        lines.append("**🔥 强势ETF Top5（动量领先）**\n")
        for i, etf in enumerate(rotation_data['strong'], 1):
            lines.append(
                f"{i}. **{etf['name']}** (`{etf['code']}`) "
                f"¥{etf['price']:.3f} "
                f"| 日涨跌 {etf['ret_1d']:+.2f}% "
                f"| 5日 {etf['ret_5d']:+.2f}% "
                f"| 20日 {etf['ret_20d']:+.2f}% "
                f"| 60日 {etf['ret_60d']:+.2f}% "
                f"| 趋势: `{etf['ma_trend']}`"
            )
        
        lines.append("\n**❄️ 弱势ETF Top5（动量落后）**\n")
        for i, etf in enumerate(rotation_data['weak'], 1):
            lines.append(
                f"{i}. **{etf['name']}** (`{etf['code']}`) "
                f"¥{etf['price']:.3f} "
                f"| 日涨跌 {etf['ret_1d']:+.2f}% "
                f"| 5日 {etf['ret_5d']:+.2f}% "
                f"| 20日 {etf['ret_20d']:+.2f}% "
                f"| 趋势: `{etf['ma_trend']}`"
            )
        
        lines.append("\n**📊 分类别强弱一览**\n")
        for cat, etfs in rotation_data.get('by_category', {}).items():
            if not etfs:
                continue
            top3 = etfs[:3]
            bottom3 = etfs[-3:][::-1]
            top_names = " > ".join([f"{e['name']}({e['momentum_score']:+.1f})" for e in top3])
            bottom_names = " < ".join([f"{e['name']}({e['momentum_score']:+.1f})" for e in bottom3])
            lines.append(f"- **{cat}**：\n  - 强势: {top_names}\n  - 弱势: {bottom_names}")
        
        lines.append("\n> 💡 **轮动策略参考**：强势品种可关注回调买入机会，弱势品种规避或考虑轮出。")
        
        return "\n".join(lines) + "\n"


class SectorHeatAnalyzer:
    """行业热点分析师"""
    
    def __init__(self):
        self.now = datetime.now(TZ_BJS)
    
    def analyze_sectors(self) -> dict:
        """分析行业热点"""
        log.info("🌋 开始分析行业热点...")
        
        try:
            hot_map = fetch_hot_sectors()
            if not hot_map:
                return {'top_sectors': [], 'sector_details': []}
            
            sector_counts = Counter(hot_map.values())
            top_sectors = sector_counts.most_common(10)
            
            sector_details = []
            try:
                import akshare as ak
                df = ak.stock_board_industry_name_em()
                if df is not None and not df.empty:
                    name_col = next((c for c in df.columns if '板块名称' in c or 'name' in c.lower()), None)
                    pct_col = next((c for c in df.columns if '涨跌幅' in c or 'pct' in c.lower()), None)
                    
                    if name_col and pct_col:
                        df[pct_col] = pd.to_numeric(df[pct_col], errors='coerce')
                        sorted_df = df.sort_values(pct_col, ascending=False)
                        
                        for _, row in sorted_df.head(10).iterrows():
                            sector_name = str(row[name_col])
                            sector_pct = float(row[pct_col]) if pd.notna(row[pct_col]) else 0
                            stock_count = sector_counts.get(sector_name, 0)
                            sector_details.append({
                                'name': sector_name,
                                'pct': sector_pct,
                                'stock_count': stock_count,
                            })
            except Exception as e:
                log.debug(f"获取行业涨幅榜失败: {e}")
                for sector_name, count in top_sectors:
                    sector_details.append({
                        'name': sector_name,
                        'pct': 0,
                        'stock_count': count,
                    })
            
            log.info(f"✅ 行业热点分析完成，领涨板块: {[s['name'] for s in sector_details[:5]]}")
            
            return {
                'top_sectors': top_sectors,
                'sector_details': sector_details,
            }
        except Exception as e:
            log.warning(f"行业热点分析失败: {e}")
            return {'top_sectors': [], 'sector_details': []}
    
    def format_sector_report(self, sector_data: dict) -> str:
        """格式化行业热点报告"""
        if not sector_data.get('sector_details'):
            return "### 🌋 行业热点分析\n⚠️ 行业数据获取失败，暂无法提供热点分析。\n"
        
        lines = ["### 🌋 行业热点分析\n"]
        
        lines.append("**🚀 今日领涨行业 Top10**\n")
        for i, sector in enumerate(sector_data['sector_details'], 1):
            pct_str = f"{sector['pct']:+.2f}%" if sector['pct'] != 0 else "数据暂缺"
            count_str = f"成分股 {sector['stock_count']} 只" if sector['stock_count'] > 0 else ""
            lines.append(
                f"{i}. **{sector['name']}** - {pct_str} {count_str}"
            )
        
        if len(sector_data['sector_details']) >= 3:
            lines.append("\n**💡 热点解读**")
            top3_names = [s['name'] for s in sector_data['sector_details'][:3]]
            lines.append(f"- 今日市场主线集中在 **{'、'.join(top3_names)}** 等方向")
            lines.append(f"- 建议关注领涨板块中的龙头标的，注意追高风险")
            lines.append(f"- 可结合ETF轮动分析，选择对应的行业ETF参与")
        
        return "\n".join(lines) + "\n"


class DailyBriefingGenerator:
    """每日投研简报生成器"""
    
    def __init__(self):
        self.now = datetime.now(TZ_BJS)
        self.now_str = self.now.strftime('%Y-%m-%d %H:%M')
        self.c_conf = Config()
    
    def _get_market_analysis(self) -> tuple[str, dict]:
        """获取市场分析"""
        log.info("📊 开始市场分析...")
        
        try:
            df_raw = fetch_spot()
            if df_raw is None or df_raw.empty:
                return "⚠️ 市场数据获取失败", {}
            
            df_clean, m_ok, m_msg, idx_ret, m_overheated, m_regime, vol_surge, m_temp = \
                extract_market_context(df_raw, self.c_conf)
            
            market_info = {
                'ok': m_ok,
                'regime': m_regime,
                'overheated': m_overheated,
                'vol_surge': vol_surge,
                'temp': m_temp,
                'index_ret': idx_ret,
                'total_stocks': len(df_clean),
            }
            
            return m_msg, market_info
        except Exception as e:
            log.error(f"市场分析失败: {e}")
            return f"⚠️ 市场分析失败: {e}", {}
    
    def _get_stock_signals(self, market_info: dict) -> tuple[dict, list]:
        """获取股票信号"""
        log.info("🎯 开始股票信号扫描...")
        
        try:
            from main import get_signals as main_get_signals
            signals, watchlist, _, _, _, _ = main_get_signals()
            return signals, watchlist
        except Exception as e:
            log.warning(f"股票信号获取失败: {e}")
            return {}, []
    
    def _format_stock_signals(self, signals: dict, watchlist: list) -> str:
        """格式化股票信号"""
        lines = ["### 🎯 股票信号\n"]
        
        has_core = bool(signals.get('Core'))
        has_satellite = bool(signals.get('Satellite'))
        
        if not has_core and not has_satellite and not watchlist:
            lines.append("✅ 今日未发现高置信度信号，建议空仓防守或轻仓观望。\n")
            return "\n".join(lines)
        
        if has_core:
            lines.append("**🔥 核心主力池（高置信度）**\n")
            for i, s in enumerate(signals['Core'][:5], 1):
                lines.append(
                    f"{i}. **{s.name}** (`{s.code}`) "
                    f"¥{s.price} ({s.pct_chg}) "
                    f"| 评分: **{s.score}分** "
                    f"| 目标: ¥{s.target1} "
                    f"| 止损: ¥{s.stop_loss}"
                )
            lines.append("")
        
        if has_satellite:
            lines.append("**🛰️ 卫星观察池（观察跟踪）**\n")
            for i, s in enumerate(signals['Satellite'][:5], 1):
                lines.append(
                    f"{i}. **{s.name}** (`{s.code}`) "
                    f"¥{s.price} ({s.pct_chg}) "
                    f"| 评分: **{s.score}分**"
                )
            lines.append("")
        
        if watchlist:
            lines.append("**👁️ 候补观察池**\n")
            for name, code, score, price in watchlist[:5]:
                lines.append(
                    f"- `{code}` **{name}** ¥{price} | 得分: {score}"
                )
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_briefing(self) -> str:
        """生成完整的每日投研简报"""
        log.info(f"📰 开始生成 {self.now_str} 每日投研简报...")
        
        market_msg, market_info = self._get_market_analysis()
        
        etf_analyzer = ETFRotationAnalyzer()
        rotation_data = etf_analyzer.analyze_rotation()
        etf_report = etf_analyzer.format_rotation_report(rotation_data)
        
        sector_analyzer = SectorHeatAnalyzer()
        sector_data = sector_analyzer.analyze_sectors()
        sector_report = sector_analyzer.format_sector_report(sector_data)
        
        signals, watchlist = self._get_stock_signals(market_info)
        stock_report = self._format_stock_signals(signals, watchlist)
        
        content = f"## 📰 每日投研简报\n> **{self.now_str}**\n\n"
        
        content += "---\n\n"
        content += market_msg
        content += "\n\n---\n\n"
        content += stock_report
        content += "\n---\n\n"
        content += etf_report
        content += "\n---\n\n"
        content += sector_report
        
        content += "\n\n---\n\n"
        content += "> ⚠️ **风险提示**：本简报由AI量化系统自动生成，仅供参考，不构成投资建议。股市有风险，投资需谨慎。"
        
        log.info("✅ 每日投研简报生成完成")
        return content
    
    def send_to_dingtalk(self, content: str) -> bool:
        """发送到钉钉"""
        try:
            title = f"📰 每日投研简报 - {self.now.strftime('%Y-%m-%d')}"
            NotificationGateway.send(title, content, template="blue")
            log.info("✅ 每日投研简报已推送至钉钉/飞书")
            return True
        except Exception as e:
            log.error(f"❌ 推送失败: {e}")
            return False


def main():
    """主函数"""
    log.info("=" * 60)
    log.info("📰 每日投研简报系统启动")
    log.info("=" * 60)
    
    generator = DailyBriefingGenerator()
    
    content = generator.generate_briefing()
    
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'daily_briefing_output.md')
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        log.info(f"📝 简报已保存到: {output_file}")
    except Exception as e:
        log.warning(f"保存简报文件失败: {e}")
    
    success = generator.send_to_dingtalk(content)
    
    log.info("=" * 60)
    if success:
        log.info("✅ 每日投研简报任务完成")
    else:
        log.warning("⚠️ 简报生成完成但推送失败")
    log.info("=" * 60)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
