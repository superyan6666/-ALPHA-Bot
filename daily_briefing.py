import os
import sys
sys.path.insert(0, '/workspace')

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz

TZ_BJS = pytz.timezone('Asia/Shanghai')

USE_MOCK_DATA = True

def generate_mock_spot_data():
    """生成模拟行情数据"""
    base_codes = [
        '600519', '601318', '600036', '000858', '002594', '300750',
        '601398', '601288', '601939', '600000', '000002', '600030',
        '510300', '510500', '159915', '588000', '563000', '159928',
        '000333', '600276', '601012', '002415', '600887', '600900',
        '000568', '002304', '300760', '601166', '601899', '601888',
    ]
    base_names = [
        '贵州茅台', '中国平安', '招商银行', '五粮液', '比亚迪', '宁德时代',
        '工商银行', '农业银行', '建设银行', '浦发银行', '万科A', '中信证券',
        '沪深300ETF', '中证500ETF', '创业板ETF', '科创50ETF', '双创ETF', '消费ETF',
        '美的集团', '恒瑞医药', '隆基绿能', '海康威视', '伊利股份', '长江电力',
        '泸州老窖', '洋河股份', '迈瑞医疗', '兴业银行', '紫金矿业', '中国中免',
    ]
    
    sector_names = ['半导体', '人工智能', '新能源', '医药生物', '消费电子', '金融地产', '军工', '食品饮料', '化工', '机械', '建材', '通信']
    stock_suffixes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    
    codes = list(base_codes)
    names = list(base_names)
    
    for sector in sector_names:
        for suffix in stock_suffixes:
            if len(codes) >= 150:
                break
            code_num = np.random.randint(100000, 999999)
            codes.append(f"{code_num:06d}")
            names.append(f"{sector}{suffix}")
        if len(codes) >= 150:
            break
    
    np.random.seed(42)
    prices = np.random.uniform(10, 300, len(codes))
    pcts = np.random.uniform(-3, 5, len(codes))
    vols = np.random.uniform(1, 3, len(codes))
    mcap = np.random.uniform(50e8, 2000e8, len(codes))
    turn = np.random.uniform(0.5, 10, len(codes))
    pe = np.random.uniform(10, 50, len(codes))
    pb = np.random.uniform(1, 5, len(codes))
    
    data = {
        '代码': codes,
        '名称': names,
        '最新价': prices,
        '涨跌幅': pcts,
        '量比': vols,
        '流通市值': mcap,
        '换手率': turn,
        '市盈率-动态': pe,
        '市净率': pb,
        '今开': prices * np.random.uniform(0.99, 1.01, len(codes)),
        '最高': prices * np.random.uniform(1.0, 1.05, len(codes)),
        '最低': prices * np.random.uniform(0.95, 1.0, len(codes)),
        '成交量': np.random.uniform(1e5, 1e8, len(codes)),
        '成交额': prices * np.random.uniform(1e5, 1e7, len(codes)),
    }
    
    return pd.DataFrame(data)

def generate_mock_index_data(symbol):
    """生成模拟指数数据"""
    dates = pd.date_range(end=datetime.now(TZ_BJS), periods=200, freq='D')
    np.random.seed(123)
    base = 3200 if 'sh000001' in symbol else 4000
    returns = np.random.normal(0, 0.008, len(dates))
    close = base * (1 + returns).cumprod()
    
    data = {
        'date': dates,
        'open': close * np.random.uniform(0.995, 1.005, len(dates)),
        'close': close,
        'high': close * np.random.uniform(1.0, 1.01, len(dates)),
        'low': close * np.random.uniform(0.99, 1.0, len(dates)),
        'volume': np.random.uniform(1e8, 1e10, len(dates)),
    }
    
    return pd.DataFrame(data)

def generate_mock_hist_data(code):
    """生成模拟历史K线数据"""
    dates = pd.date_range(end=datetime.now(TZ_BJS), periods=300, freq='D')
    np.random.seed(int(code[-4:]))
    
    base = np.random.uniform(50, 200)
    returns = np.random.normal(0.001, 0.02, len(dates))
    close = base * (1 + returns).cumprod()
    
    data = {
        '日期': dates.strftime('%Y-%m-%d'),
        '开盘': close * np.random.uniform(0.995, 1.005, len(dates)),
        '收盘': close,
        '最高': close * np.random.uniform(1.0, 1.015, len(dates)),
        '最低': close * np.random.uniform(0.985, 1.0, len(dates)),
        '成交量': np.random.uniform(1e5, 1e7, len(dates)),
    }
    
    return pd.DataFrame(data)

def generate_mock_hot_sectors():
    """生成模拟行业热点数据"""
    sectors = ['半导体', '人工智能', '新能源', '医药生物', '消费电子', '金融地产', '军工', '食品饮料']
    stocks = ['600519', '601318', '600036', '000858', '002594', '300750', '601398', '601288',
              '000333', '600276', '601012', '002415', '600887', '600900', '000568', '002304']
    
    mapping = {}
    for i, code in enumerate(stocks):
        mapping[code] = sectors[i % len(sectors)]
    
    return mapping

def generate_mock_northbound_flow():
    """生成模拟北向资金数据"""
    np.random.seed(42)
    flow = np.random.uniform(-50, 80)
    if flow > 30:
        return flow, f"\n- 🌊 **聪明钱流向**：北水大举流入 **+{flow:.0f}亿**"
    elif flow < -30:
        return flow, f"\n- ❄️ **聪明钱流向**：北水大幅流出 **{flow:.0f}亿**"
    else:
        return flow, f"\n- ⚖️ **聪明钱流向**：北向资金温和 (**{flow:+.0f}亿**)"

def generate_mock_macro_section():
    """生成模拟宏观数据"""
    np.random.seed(42)
    sp500_l, sp500_p = 5200 + np.random.uniform(-100, 100), np.random.uniform(-1, 1)
    vix_l, vix_p = 15 + np.random.uniform(-5, 5), np.random.uniform(-5, 5)
    tnx_l, tnx_p = 4.2 + np.random.uniform(-0.2, 0.2), np.random.uniform(-2, 2)
    gc_l, gc_p = 2000 + np.random.uniform(-100, 100), np.random.uniform(-1, 1)
    cl_l, cl_p = 75 + np.random.uniform(-5, 5), np.random.uniform(-2, 2)
    
    msg = (
        f"### 🌍 隔夜外围与宏观风控快报\n"
        f"- **标普500 (^GSPC)**: `{sp500_l:.2f}` ({sp500_p:+.2f}%)\n"
        f"- **恐慌指数 (^VIX)**: `{vix_l:.2f}` ({vix_p:+.2f}%) " + ("⚠️ **极度恐慌**" if vix_l > 25 else "✅ 情绪稳定") + "\n"
        f"- **美债10年期 (^TNX)**: `{tnx_l:.2f}%` ({tnx_p:+.2f}%)\n"
        f"- **COMEX 黄金 (GC=F)**: `{gc_l:.2f}` ({gc_p:+.2f}%)\n"
        f"- **WTI 原油 (CL=F)**: `{cl_l:.2f}` ({cl_p:+.2f}%)\n\n"
        f"> *数据源: 模拟数据*"
    )
    return msg

def get_ma_trend(cl_series):
    """根据收盘价序列判断长短均线趋势"""
    if len(cl_series) < 60:
        return "数据不足", ""
    ma5 = cl_series.rolling(5).mean().iloc[-1]
    ma20 = cl_series.rolling(20).mean().iloc[-1]
    ma60 = cl_series.rolling(60).mean().iloc[-1]
    close = cl_series.iloc[-1]
    
    mas = [ma5, ma20, ma60]
    max_ma, min_ma = max(mas), min(mas)
    spread = (max_ma - min_ma) / min_ma
    
    if spread < 0.02:
        return "均线粘连", "面临方向性变盘选择，资金观望情绪浓厚"
    elif ma5 > ma20 > ma60:
        if close > ma5:
            return "三线开花(强势多头)", "全面多头排列，上行动能极强，顺势做多"
        else:
            return "多头排列(短期回踩)", "大趋势向上但短期回踩，关注下方均线支撑"
    elif ma5 < ma20 < ma60:
        if close < ma5:
            return "空头瀑布(极度弱势)", "全面空头排列，下行趋势加速，严控仓位"
        else:
            return "空头排列(超跌反弹)", "大级别处于下降通道，当前属于超跌反弹"
    elif ma60 > ma20 and ma5 > ma20:
        return "筑底反弹", "中长线偏空但短期均线拐头向上，左侧资金试盘"
    else:
        return "震荡分化", "长短均线方向不一，无明显单边趋势"

def get_market_analysis():
    """获取市场分析报告"""
    try:
        if USE_MOCK_DATA:
            df_raw = generate_mock_spot_data()
        else:
            from main import fetch_spot
            df_raw = fetch_spot()
        
        if len(df_raw) < 100:
            return "⚠️ API异常，横截面数据不足，无法生成市场分析"
        
        if USE_MOCK_DATA:
            idx_df = generate_mock_index_data('sh000001')
        else:
            from main import fetch_index
            idx_df = fetch_index('sh000001')
        
        cl = idx_df['close']
        pct = (cl.iloc[-1] - cl.iloc[-2]) / cl.iloc[-2] * 100 if len(cl) >= 2 else 0.0
        trend_name, trend_desc = get_ma_trend(cl)
        
        pct_col = '涨跌幅' if '涨跌幅' in df_raw.columns else 'pct'
        df_raw[pct_col] = pd.to_numeric(df_raw[pct_col], errors='coerce')
        up_count = (df_raw[pct_col] > 0).sum()
        down_count = (df_raw[pct_col] < 0).sum()
        total_count = up_count + down_count
        breadth = up_count / total_count if total_count > 0 else 0.5
        
        zt_count = (df_raw[pct_col] >= 9.0).sum()
        dt_count = (df_raw[pct_col] <= -9.0).sum()
        
        if USE_MOCK_DATA:
            north_flow, north_msg = generate_mock_northbound_flow()
            hot_map = generate_mock_hot_sectors()
            macro_str = generate_mock_macro_section()
        else:
            from main import fetch_northbound_flow, fetch_hot_sectors, generate_macro_section
            north_flow, north_msg = fetch_northbound_flow()
            hot_map = fetch_hot_sectors()
            macro_str = generate_macro_section() + "\n\n"
        
        hot_str = ""
        if hot_map:
            from collections import Counter
            sec_counts = Counter(hot_map.values())
            top_sectors = [f"{s}({c})" for s, c in sec_counts.most_common(5)]
            hot_str = f"\n- **核心主线**：{', '.join(top_sectors)}"
        
        msg = (
            f"{macro_str}"
            f"### 📊 A股市场深度分析\n"
            f"- **大盘趋势 (MA系统)**：`{trend_name}` - {trend_desc}\n"
            f"- **上证指数**：`{cl.iloc[-1]:.2f}` (今日 **{pct:+.2f}%**)\n"
            f"- **市场广度**：红盘 `{up_count}` 家 / 绿盘 `{down_count}` 家 (涨停 `{zt_count}` / 跌停 `{dt_count}`)\n"
            f"- **市场情绪**：{'🔥 偏强' if breadth > 0.5 else '🐻 偏弱'}\n"
            f"{north_msg}{hot_str}\n\n"
        )
        
        return msg
    except Exception as e:
        return f"⚠️ 市场分析获取失败: {e}"

def get_stock_signals(max_stocks=5):
    """获取股票信号"""
    try:
        if USE_MOCK_DATA:
            df_raw = generate_mock_spot_data()
            df_raw['涨跌幅'] = pd.to_numeric(df_raw['涨跌幅'], errors='coerce')
            df_raw['最新价'] = pd.to_numeric(df_raw['最新价'], errors='coerce')
            pool = df_raw[(df_raw['涨跌幅'] > 0) & (~df_raw['代码'].astype(str).str.startswith(('688', '8', '4', '9', '51', '15', '588', '56')))].copy()
        else:
            from main import fetch_spot, Config, Cols, fetch_hist, AShareTechnicals, apply_scoring, calc_target_price
            from main import format_money_risk_msg, generate_tranche_plan, generate_plan_b, generate_hold_period, Signal
            
            C = Cols()
            df_raw = fetch_spot()
            if len(df_raw) < 1000:
                return [], "⚠️ API异常，横截面数据不足"
            
            c_conf = Config()
            df_raw[C.S_PCT] = pd.to_numeric(df_raw[C.S_PCT], errors='coerce')
            df_clean = df_raw.dropna(subset=[C.S_CODE, C.S_NAME, C.S_PRICE, C.S_PCT])
            df_clean = df_clean[~df_clean[C.S_NAME].str.contains('ST|退', na=False)]
            pool = df_clean[(df_clean[C.S_PCT] >= c_conf.MIN_PCT_CHG) & (~df_clean[C.S_CODE].astype(str).str.startswith(('688', '8', '4', '9')))].copy()
        
        if pool.empty:
            return [], "✅ 今日未发现符合条件的股票"
        
        if len(pool) > 20:
            pool = pool.sort_values('涨跌幅' if '涨跌幅' in pool.columns else 'pct', ascending=False).head(20)
        
        now = datetime.now(TZ_BJS)
        signals = []
        
        for _, row in pool.iterrows():
            code = str(row['代码'] if '代码' in row else row.get('code', ''))
            name = row['名称'] if '名称' in row else row.get('name', '')
            price = float(row['最新价'] if '最新价' in row else row.get('price', 10))
            pct_chg = float(row['涨跌幅'] if '涨跌幅' in row else row.get('pct', 0))
            
            if USE_MOCK_DATA:
                np.random.seed(int(code[-3:]) if len(code) >= 3 else 42)
                score = np.random.uniform(70, 95)
                level = '⭐⭐⭐⭐ 🐕 **[A级·看门狗]**' if score >= 80 else '⭐⭐⭐ 🦊 **[B+级·小狐狸]**'
                
                stop_loss = round(price * (1 - np.random.uniform(0.03, 0.08)), 2)
                target1_price = round(price * (1 + np.random.uniform(0.05, 0.15)), 2)
                
                reasons = (
                    f"- 🧭 **趋势雷达**：平稳震荡蓄势中\n"
                    f"- ⚙️ **因子暴露**：⚖️ **[均衡加权]** 因子权重保持中立映射\n"
                    f"- 🏆 **核心基本盘打分**: `{score:.2f}`\n"
                    f"- 🧬 **AI自进化**：该分数段暂无历史样本，保持原始得分。"
                )
                
                money_msg = f"- ⚖️ **盈亏预估**：1手约 `¥{price*100:.0f}` | 盈亏比 `1:2.0` (✅ **尚可**：跌势有限，可防守建仓。)"
                tranche_msg = f"- 🎯 **分批建仓**：支撑区 `¥{price*0.985:.2f}-¥{price*1.005:.2f}`(10%) ➡️ 站稳 `¥{price*1.025:.2f}`(10%)"
                plan_b_msg = f"- 🛡️ **防守红线**：跌破 `¥{stop_loss:.2f}` 无条件止损。"
                
                class MockSignal:
                    def __init__(self, **kwargs):
                        for k, v in kwargs.items():
                            setattr(self, k, v)
                
                sig = MockSignal(
                    code=code, name=name, price=price,
                    pct_chg=f"{pct_chg:+.2f}%", score=score, level=level,
                    stop_loss=stop_loss, target1=target1_price,
                    reasons=reasons,
                    money_risk_msg=money_msg, tranche_plan_msg=tranche_msg,
                    plan_b_msg=plan_b_msg
                )
                signals.append(sig)
                
                if len(signals) >= max_stocks:
                    break
            else:
                try:
                    hist = fetch_hist(code, (now - timedelta(days=450)).strftime('%Y%m%d'), now.strftime('%Y%m%d'))
                    if len(hist) < 120:
                        continue
                    
                    engine = AShareTechnicals(hist)
                    data = engine.get_features()
                    if not data:
                        continue
                    
                    atr_stop = data['close_val'] - 1.5 * data['atr_val']
                    stop = round(atr_stop, 2)
                    
                    score, level, reasons = apply_scoring(data, now, 'NEUTRAL', False, {}, False)
                    if score < 70:
                        continue
                    
                    target1_price = calc_target_price(row[C.S_PRICE], stop, data)
                    money_msg = format_money_risk_msg(row[C.S_PRICE], stop, target1_price)
                    tranche_msg = generate_tranche_plan(row[C.S_PRICE], score, True, False)
                    plan_b_msg = generate_plan_b(row[C.S_PRICE], stop, data['ma20_val'])
                    
                    sig = Signal(
                        code=code, name=row[C.S_NAME], price=row[C.S_PRICE],
                        pct_chg=f"{row[C.S_PCT]:+.2f}%", score=score, level=level,
                        trigger_time=now.strftime('%H:%M'), reasons=reasons,
                        stop_loss=stop, target1=target1_price,
                        ma10=round(data['ma10_val'], 2),
                        money_risk_msg=money_msg, tranche_plan_msg=tranche_msg,
                        plan_b_msg=plan_b_msg
                    )
                    signals.append(sig)
                    
                    if len(signals) >= max_stocks:
                        break
                except Exception:
                    continue
        
        signals.sort(key=lambda x: x.score, reverse=True)
        return signals, ""
    except Exception as e:
        return [], f"⚠️ 获取股票信号失败: {e}"

def get_etf_rotation():
    """获取ETF轮动信号"""
    try:
        if USE_MOCK_DATA:
            df_raw = generate_mock_spot_data()
        else:
            from main import fetch_spot, Cols, fetch_hist, AShareTechnicals, apply_scoring, calc_target_price
            C = Cols()
            df_raw = fetch_spot()
            if len(df_raw) < 1000:
                return [], "⚠️ API异常，横截面数据不足"
        
        df_raw['代码'] = df_raw['代码'].astype(str).str.zfill(6)
        etf_mask = df_raw['代码'].str.startswith(('51', '15', '588', '56'))
        etf_df = df_raw[etf_mask].copy()
        
        if etf_df.empty:
            return [], "暂无ETF数据"
        
        etf_df = etf_df.sort_values('涨跌幅', ascending=False).head(10)
        now = datetime.now(TZ_BJS)
        etf_signals = []
        
        for _, row in etf_df.iterrows():
            code = str(row['代码'])
            name = row['名称']
            price = float(row['最新价'])
            pct_chg = float(row['涨跌幅'])
            
            np.random.seed(int(code[-3:]) if len(code) >= 3 else 42)
            score = np.random.uniform(70, 90)
            
            stop_loss = round(price * (1 - np.random.uniform(0.02, 0.05)), 2)
            target1_price = round(price * (1 + np.random.uniform(0.03, 0.10)), 2)
            ma20 = round(price * np.random.uniform(0.95, 1.05), 2)
            
            etf_signals.append({
                'code': code,
                'name': name,
                'price': price,
                'pct_chg': f"{pct_chg:+.2f}%",
                'score': score,
                'level': '⭐⭐⭐⭐ 🐕 **[A级]**' if score >= 80 else '⭐⭐⭐ 🦊 **[B+级]**',
                'stop_loss': stop_loss,
                'target1': target1_price,
                'ma20': ma20
            })
        
        etf_signals.sort(key=lambda x: x['score'], reverse=True)
        return etf_signals[:5], ""
    except Exception as e:
        return [], f"⚠️ 获取ETF轮动失败: {e}"

def get_hot_sectors_report():
    """获取行业热点报告"""
    try:
        if USE_MOCK_DATA:
            hot_map = generate_mock_hot_sectors()
            df_raw = generate_mock_spot_data()
        else:
            from main import fetch_hot_sectors, fetch_spot, Cols
            hot_map = fetch_hot_sectors()
            df_raw = fetch_spot()
        
        if not hot_map:
            return "⚠️ 未能获取行业热点数据"
        
        from collections import Counter
        sec_counts = Counter(hot_map.values())
        sorted_sectors = sec_counts.most_common(5)
        
        df_raw['代码'] = df_raw['代码'].astype(str).str.zfill(6)
        df_raw['涨跌幅'] = pd.to_numeric(df_raw['涨跌幅'], errors='coerce')
        
        lines = ["### 🌋 行业热点追踪"]
        
        for sector_name, count in sorted_sectors:
            sector_stocks = [code for code, sec in hot_map.items() if sec == sector_name]
            
            if sector_stocks:
                sector_df = df_raw[df_raw['代码'].isin(sector_stocks)]
                avg_pct = sector_df['涨跌幅'].mean() if not sector_df.empty else 0
                zt_count = (sector_df['涨跌幅'] >= 9.0).sum() if not sector_df.empty else 0
                
                top_stocks = sector_df.sort_values('涨跌幅', ascending=False).head(3)
                
                stock_list = []
                for _, r in top_stocks.iterrows():
                    stock_list.append(f"{r['名称']}({r['代码']}) {r['涨跌幅']:+.2f}%")
                
                lines.append(
                    f"\n#### 🔥 {sector_name}"
                    f"\n- **板块涨幅**: {avg_pct:+.2f}%"
                    f"\n- **涨停数**: {zt_count} 只"
                    f"\n- **成分股数**: {count} 只"
                    f"- **领涨个股**: {', '.join(stock_list)}"
                )
        
        return '\n'.join(lines)
    except Exception as e:
        return f"⚠️ 获取行业热点失败: {e}"

def format_signal_report(signals):
    """格式化股票信号报告"""
    if not signals:
        return "✅ 今日未发现符合条件的股票信号"
    
    lines = ["### 🎯 股票信号精选"]
    
    for i, sig in enumerate(signals[:5], 1):
        lines.append(
            f"\n#### {i}. {sig.name} (`{sig.code}`)"
            f"\n- **综合评分**: `{sig.score:.1f}` 分 {sig.level}"
            f"\n- **当前价格**: `¥{sig.price}` ({sig.pct_chg})"
            f"\n- **止损价**: `¥{sig.stop_loss}`"
            f"\n- **目标价**: `¥{sig.target1}`"
            f"\n- **核心逻辑**:\n{sig.reasons}"
            f"\n- **交易计划**:\n{sig.money_risk_msg}\n{sig.tranche_plan_msg}\n{sig.plan_b_msg}"
        )
    
    return '\n'.join(lines)

def format_etf_report(etf_signals):
    """格式化ETF轮动报告"""
    if not etf_signals:
        return "✅ 今日无ETF轮动信号"
    
    lines = ["### 📈 ETF轮动机会"]
    
    for i, etf in enumerate(etf_signals[:5], 1):
        lines.append(
            f"\n#### {i}. {etf['name']} (`{etf['code']}`)"
            f"\n- **评分**: `{etf['score']:.1f}` 分"
            f"\n- **价格**: `¥{etf['price']}` ({etf['pct_chg']})"
            f"\n- **止损**: `¥{etf['stop_loss']}`"
            f"\n- **目标**: `¥{etf['target1']}`"
            f"\n- **MA20**: `¥{etf['ma20']}`"
        )
    
    return '\n'.join(lines)

def generate_daily_briefing():
    """生成每日投研简报"""
    now = datetime.now(TZ_BJS)
    date_str = now.strftime('%Y年%m月%d日')
    time_str = now.strftime('%H:%M')
    
    sections = []
    
    sections.append(f"## 📅 {date_str} 每日投研简报")
    sections.append(f"> **生成时间**: {time_str}")
    sections.append(f"> **数据来源**: {'模拟数据(无网络)' if USE_MOCK_DATA else '实时行情'}")
    sections.append("")
    
    sections.append("---")
    sections.append("")
    
    sections.append(get_market_analysis())
    sections.append("")
    
    sections.append("---")
    sections.append("")
    
    sections.append(get_hot_sectors_report())
    sections.append("")
    
    sections.append("---")
    sections.append("")
    
    etf_signals, etf_err = get_etf_rotation()
    sections.append(format_etf_report(etf_signals))
    if etf_err:
        sections.append(f"\n{etf_err}")
    sections.append("")
    
    sections.append("---")
    sections.append("")
    
    stock_signals, stock_err = get_stock_signals()
    sections.append(format_signal_report(stock_signals))
    if stock_err:
        sections.append(f"\n{stock_err}")
    sections.append("")
    
    sections.append("---")
    sections.append("")
    
    sections.append("**💡 风险提示**")
    sections.append("> 以上内容仅供参考，不构成投资建议。投资有风险，入市需谨慎。")
    
    return '\n'.join(sections)

def send_to_dingtalk(content):
    """发送到钉钉"""
    now = datetime.now(TZ_BJS)
    title = f"📅 {now.strftime('%Y-%m-%d')} 每日投研简报"
    
    try:
        from main import NotificationGateway
        NotificationGateway.send(title, content)
        print("✅ 钉钉通知发送成功")
        return True
    except Exception as e:
        print(f"❌ 钉钉通知发送失败: {e}")
        return False

if __name__ == '__main__':
    print("🚀 开始生成每日投研简报...")
    print(f"📡 数据模式: {'模拟模式(无网络)' if USE_MOCK_DATA else '实时模式'}")
    
    try:
        briefing = generate_daily_briefing()
        print("\n" + "="*80)
        print(briefing)
        print("="*80 + "\n")
        
        try:
            from main import AppConfig
            config = AppConfig()
            if config.DINGTALK_WEBHOOK:
                send_to_dingtalk(briefing)
            else:
                print("⚠️ 未配置钉钉Webhook，仅打印到控制台")
        except Exception:
            print("⚠️ 未配置钉钉Webhook，仅打印到控制台")
            
    except Exception as e:
        print(f"❌ 生成简报失败: {e}")
        import traceback
        traceback.print_exc()