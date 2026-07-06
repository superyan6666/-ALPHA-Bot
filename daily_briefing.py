import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz

TZ_BJS = pytz.timezone('Asia/Shanghai')

def load_env_config():
    env_file = os.path.join(os.path.dirname(__file__), 'config.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env_config()

from main import (
    config, log, NotificationGateway, Cols, AShareTechnicals,
    calc_target_price, get_ma_trend, Config
)

C = Cols()
LOCAL_DATA_PATH = '.quantbot_data/ashare_daily.parquet'

def load_local_data():
    """加载本地股票数据"""
    try:
        df = pd.read_parquet(LOCAL_DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        df['pctChg'] = pd.to_numeric(df['pctChg'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df['turn'] = pd.to_numeric(df['turn'], errors='coerce')
        return df
    except Exception as e:
        log.error(f"加载本地数据失败: {e}")
        return None

def get_latest_date(df):
    """获取最新日期"""
    if df is None or df.empty:
        return None
    return df['date'].max()

def get_market_overview(df, latest_date):
    """基于本地数据获取市场概览"""
    try:
        day_data = df[df['date'] == latest_date].copy()
        
        up_count = (day_data['pctChg'] > 0).sum()
        down_count = (day_data['pctChg'] < 0).sum()
        zt_count = (day_data['pctChg'] >= 9.0).sum()
        dt_count = (day_data['pctChg'] <= -9.0).sum()
        
        avg_pct = day_data['pctChg'].mean()
        total_amt = day_data['amount'].sum() / 1e8
        
        bread = up_count / (up_count + down_count) if (up_count + down_count) > 0 else 0.5
        
        if bread < 0.25:
            market_state = "🧊 **恐慌冰点**"
            advice = "仓位 10%-20%。系统性风险释放，多看少动。"
        elif avg_pct > 1.0 and bread > 0.6:
            market_state = "🔥 **强势多头**"
            advice = "仓位 60%-80%。赚钱效应极佳，跟随主线积极做多。"
        elif avg_pct < -1.0 and bread <= 0.4:
            market_state = "🐻 **弱势空头**"
            advice = "仓位 20%-30%。均线压制，控制回撤。"
        else:
            market_state = "⚖️ **震荡均衡**"
            advice = "仓位 40%-60%。重个股轻大盘。"
        
        msg = f"### 📊 A股市场概况\n"
        msg += f"- **数据日期**: `{latest_date.strftime('%Y-%m-%d')}`\n"
        msg += f"- **市场平均涨跌幅**: **{avg_pct:+.2f}%**\n"
        msg += f"- **综合判定**: {market_state}\n"
        msg += f"- **市场广度**: 红盘 `{up_count}` / 绿盘 `{down_count}` (涨停 `{zt_count}` / 跌停 `{dt_count}`)\n"
        msg += f"- **两市量能**: 约 `{total_amt:.0f}` 亿元\n\n"
        msg += f"**💡 仓位建议**: {advice}"
        
        return day_data, msg
    except Exception as e:
        log.error(f"市场概览获取失败: {e}")
        return None, f"⚠️ 市场数据获取失败: {e}"

def get_sector_hotspots(df, latest_date):
    """基于本地数据获取行业热点"""
    try:
        day_data = df[df['date'] == latest_date].copy()
        day_data = day_data[day_data['pctChg'] > 5].copy()
        
        if day_data.empty:
            return None, "无涨幅超过5%的股票"
        
        return [("强势股集群", len(day_data))], ""
    except Exception as e:
        log.error(f"行业热点获取失败: {e}")
        return None, str(e)

def get_etf_rotation_signals(df, latest_date, top_n=10):
    """基于本地数据获取ETF轮动信号"""
    try:
        day_data = df[df['date'] == latest_date].copy()
        
        etf_mask = day_data['code'].astype(str).str.match(r'^sh\.(51|588)\d{3}$|^sz\.(15|56)\d{3}$')
        etf_df = day_data[etf_mask].copy()
        
        if etf_df.empty:
            return None, "无ETF数据"
        
        etf_df = etf_df.sort_values(by='pctChg', ascending=False).head(top_n)
        
        results = []
        start_date = latest_date - timedelta(days=120)
        
        for _, row in etf_df.iterrows():
            code = row['code']
            hist = df[df['code'] == code]
            hist = hist[(hist['date'] >= start_date) & (hist['date'] <= latest_date)].sort_values('date')
            
            if len(hist) >= 60:
                try:
                    hist_for_engine = hist.rename(columns={
                        'open': 'open', 'high': 'high', 'low': 'low', 
                        'close': 'close', 'volume': 'volume', 'amount': 'amount'
                    }).copy()
                    engine = AShareTechnicals(hist_for_engine)
                    data = engine.get_features()
                    if data:
                        ma20 = data.get('ma20_val', 0)
                        close = data.get('close_val', 0)
                        adx = data.get('adx', 0)
                        results.append({
                            'code': code, 'name': code, 'price': row['close'],
                            'pct_chg': row['pctChg'], 'vol_ratio': 1.0,
                            'ma20': ma20, 'adx': adx, 'above_ma20': close > ma20
                        })
                except Exception as e:
                    log.warning(f"ETF {code} 分析失败: {e}")
        
        return results, ""
    except Exception as e:
        log.error(f"ETF轮动信号获取失败: {e}")
        return None, str(e)

def get_stock_signals(df, day_data, latest_date, top_n=5):
    """基于本地数据获取股票信号"""
    try:
        if day_data is None or day_data.empty:
            return None, "股票数据获取失败"
        
        day_data['code_num'] = day_data['code'].astype(str).str.extract(r'(\d{6})')[0].fillna('').str.zfill(6)
        
        stock_mask = ~day_data['code'].astype(str).str.startswith(('sh.688', 'sh.9', 'sz.4'))
        pool = day_data[stock_mask].copy()
        pool = pool[pool['pctChg'] >= -5.0].copy()
        
        if pool.empty:
            return None, "无符合条件的股票"
        
        pool = pool.sort_values(by='pctChg', ascending=False).head(50)
        
        results = []
        start_date = latest_date - timedelta(days=180)
        
        for _, row in pool.iterrows():
            code = row['code']
            hist = df[df['code'] == code]
            hist = hist[(hist['date'] >= start_date) & (hist['date'] <= latest_date)].sort_values('date')
            
            if len(hist) >= 120:
                try:
                    hist_for_engine = hist.rename(columns={
                        'open': 'open', 'high': 'high', 'low': 'low', 
                        'close': 'close', 'volume': 'volume', 'amount': 'amount'
                    }).copy()
                    engine = AShareTechnicals(hist_for_engine)
                    data = engine.get_features()
                    if data:
                        atr_val = data.get('atr_val', 0)
                        close_val = data.get('close_val', 0)
                        stop_loss = close_val - 1.5 * atr_val
                        target1 = calc_target_price(close_val, stop_loss, data)
                        
                        score = 50
                        if data.get('bull_rank', False): score += 15
                        if data.get('has_obv_break', False): score += 10
                        if data.get('has_chip_break', False): score += 10
                        if data.get('macd_divergence', False): score += 10
                        if data.get('extreme_shrink_vol', False): score += 5
                        if data.get('is_true_vcp', False): score += 10
                        
                        score = min(100, max(0, score))
                        
                        if score >= 70:
                            results.append({
                                'code': row['code_num'] if pd.notna(row['code_num']) else code,
                                'name': code, 'price': row['close'],
                                'pct_chg': row['pctChg'], 'score': score,
                                'stop_loss': round(stop_loss, 2), 'target1': target1,
                                'reasons': [
                                    f"趋势状态: {'多头' if data.get('bull_rank') else '震荡/空头'}",
                                    f"OBV突破: {'是' if data.get('has_obv_break') else '否'}",
                                    f"筹码突破: {'是' if data.get('has_chip_break') else '否'}",
                                    f"MACD背离: {'是' if data.get('macd_divergence') else '否'}"
                                ]
                            })
                except Exception as e:
                    log.warning(f"股票 {code} 分析失败: {e}")
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_n], ""
    except Exception as e:
        log.error(f"股票信号获取失败: {e}")
        return None, str(e)

def generate_briefing():
    """生成每日投研简报"""
    now = datetime.now(TZ_BJS)
    now_str = now.strftime('%Y-%m-%d %H:%M')
    
    sections = []
    sections.append(f"## 📅 每日投研简报\n> **{now_str}**\n\n")
    sections.append("---\n\n")
    
    sections.append("### 🌍 数据来源说明\n")
    sections.append("📦 使用本地数据湖（ashare_daily.parquet），包含历史行情数据\n\n")
    
    sections.append("---\n\n")
    
    df = load_local_data()
    if df is None:
        return "❌ 无法加载本地数据，请检查数据文件是否存在"
    
    latest_date = get_latest_date(df)
    if latest_date is None:
        return "❌ 无法获取数据日期"
    
    day_data, market_msg = get_market_overview(df, latest_date)
    sections.append(market_msg + "\n\n")
    
    sections.append("---\n\n")
    
    sections.append("### 🔥 行业热点排行\n")
    sectors, err = get_sector_hotspots(df, latest_date)
    if sectors:
        for sector, count in sectors:
            sections.append(f"- **{sector}**: {count} 只成分股\n")
    else:
        sections.append(f"⚠️ {err}\n")
    sections.append("\n")
    
    sections.append("---\n\n")
    
    sections.append("### 📈 ETF轮动监控\n")
    etf_signals, err = get_etf_rotation_signals(df, latest_date)
    if etf_signals:
        for etf in etf_signals:
            status = "🟢 站上MA20" if etf['above_ma20'] else "🔴 跌破MA20"
            sections.append(
                f"- **{etf['name']}** (`{etf['code']}`): ¥{etf['price']} ({etf['pct_chg']}%)\n"
                f"  - ADX: {etf['adx']:.1f} | {status}\n"
            )
    else:
        sections.append(f"⚠️ {err}\n")
    sections.append("\n")
    
    sections.append("---\n\n")
    
    sections.append("### 🎯 精选股票信号\n")
    stock_signals, err = get_stock_signals(df, day_data, latest_date)
    if stock_signals:
        for sig in stock_signals:
            level = '⭐⭐⭐⭐⭐ S级' if sig['score'] >= 85 else \
                    '⭐⭐⭐⭐ A级' if sig['score'] >= 75 else '⭐⭐⭐ B级'
            sections.append(
                f"#### {level} {sig['name']} (`{sig['code']}`)\n"
                f"- **现价**: ¥{sig['price']} ({sig['pct_chg']}%)\n"
                f"- **综合评分**: `{sig['score']}` 分\n"
                f"- **止损**: ¥{sig['stop_loss']} | **目标**: ¥{sig['target1']}\n"
                f"- **核心逻辑**:\n"
            )
            for reason in sig['reasons']:
                sections.append(f"  - {reason}\n")
            sections.append("\n")
    else:
        sections.append(f"暂无符合条件的精选股票信号。{err}\n")
    
    sections.append("---\n\n")
    sections.append("> 📝 **免责声明**: 以上内容仅供研究参考，不构成投资建议。市场有风险，投资需谨慎。")
    
    return ''.join(sections)

def main():
    log.info("🚀 每日投研简报生成引擎启动...")
    
    try:
        briefing = generate_briefing()
        
        log.info("📋 简报内容生成完成")
        print(briefing)
        
        if config.DINGTALK_WEBHOOK or config.FEISHU_WEBHOOK:
            NotificationGateway.send('📅 每日投研简报', briefing)
            log.info("✅ 简报已发送到钉钉/飞书")
        else:
            log.warning("⚠️ 未配置WEBHOOK，仅在本地打印简报")
        
    except Exception as e:
        log.critical(f"每日简报生成失败: {e}", exc_info=True)
        error_msg = f"🚨 **每日投研简报生成失败**\n\n**时间**: {datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M')}\n**异常**: {str(e)[:300]}..."
        try:
            NotificationGateway.send("🚨 每日简报失败", error_msg, template="red")
        except:
            pass

if __name__ == '__main__':
    main()
