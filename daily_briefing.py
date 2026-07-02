import os
import time
import json
import logging
from datetime import datetime, timedelta
from collections import Counter

import requests
import numpy as np
import pandas as pd
import pytz

requests.adapters.DEFAULT_RETRIES = 3
_session = requests.Session()

import akshare as ak
ak_api = ak

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

TZ_BJS = pytz.timezone('Asia/Shanghai')

DINGTALK_WEBHOOK = os.environ.get('DINGTALK_WEBHOOK', '')
NOTIFY_SEC_KEYWORD = os.environ.get('NOTIFY_SEC_KEYWORD', 'Hermes')

ETF_UNIVERSE = {
    '510300': '沪深300ETF',
    '510500': '中证500ETF',
    '512100': '中证1000ETF',
    '159915': '创业板ETF',
    '588000': '科创50ETF',
    '510050': '上证50ETF',
    '512880': '证券ETF',
    '512690': '酒ETF',
    '512010': '医药ETF',
    '512660': '军工ETF',
    '512400': '有色金属ETF',
    '515210': '钢铁ETF',
    '512980': '传媒ETF',
    '515030': '新能源车ETF',
    '512480': '半导体ETF',
    '516160': '新能源ETF',
    '512760': '芯片ETF',
    '159995': '芯片ETF',
    '513050': '中概互联ETF',
    '513100': '纳指ETF',
    '518880': '黄金ETF',
    '511010': '国债ETF',
}

def _today_str() -> str:
    return datetime.now(TZ_BJS).strftime('%Y-%m-%d')

def send_dingtalk_notification(title: str, content: str) -> bool:
    if not DINGTALK_WEBHOOK:
        log.warning("未配置 DINGTALK_WEBHOOK，跳过钉钉推送")
        return False
    
    headers = {"Content-Type": "application/json"}
    
    final_text = content
    if NOTIFY_SEC_KEYWORD and NOTIFY_SEC_KEYWORD not in final_text:
        final_text = f"### {NOTIFY_SEC_KEYWORD}\n\n{final_text}"
    
    payload = {
        'msgtype': 'markdown',
        'markdown': {
            'title': title,
            'text': final_text
        }
    }
    
    for attempt in range(2):
        try:
            res = _session.post(DINGTALK_WEBHOOK, json=payload, headers=headers, timeout=10)
            res.raise_for_status()
            res_dict = res.json()
            if res_dict.get('errcode', 0) != 0:
                log.error(f"钉钉推送接口拒绝: {res_dict}")
                return False
            log.info("✅ 钉钉推送成功")
            return True
        except Exception as e:
            if attempt == 1:
                log.error(f"钉钉推送失败: {e}")
                return False
            time.sleep(1)
    return False

def get_index_data(symbol: str) -> pd.DataFrame:
    try:
        df = ak.stock_zh_index_daily_tx(symbol=symbol)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
    except Exception as e:
        log.debug(f"新浪指数接口波动 ({symbol}): {e}")
    try:
        df = ak.stock_zh_index_daily_em(symbol=symbol)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            return df
    except Exception as e:
        log.debug(f"东方财富指数接口波动 ({symbol}): {e}")
    try:
        df = ak.stock_zh_index_daily(symbol=symbol)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            return df
    except Exception as e:
        log.warning(f"所有指数接口都失败 ({symbol}): {e}")
    return pd.DataFrame()

def calc_ma_trend(cl_series: pd.Series) -> tuple[str, str]:
    if len(cl_series) < 60:
        return "数据不足", "数据不足以判断趋势"
    ma5 = cl_series.rolling(5).mean().iloc[-1]
    ma10 = cl_series.rolling(10).mean().iloc[-1]
    ma20 = cl_series.rolling(20).mean().iloc[-1]
    ma60 = cl_series.rolling(60).mean().iloc[-1]
    close = cl_series.iloc[-1]
    
    mas = [ma5, ma20, ma60]
    max_ma, min_ma = max(mas), min(mas)
    spread = (max_ma - min_ma) / min_ma if min_ma > 0 else 0
    
    if spread < 0.02:
        return "均线粘连", "面临方向性变盘选择，资金观望情绪浓厚"
    elif ma5 > ma10 > ma20 > ma60:
        if close > ma5:
            return "三线开花(强势多头)", "全面多头排列，上行动能极强，顺势做多"
        else:
            return "多头排列(短期回踩)", "大趋势向上但短期回踩，关注下方均线支撑"
    elif ma5 < ma10 < ma20 < ma60:
        if close < ma5:
            return "空头瀑布(极度弱势)", "全面空头排列，下行趋势加速，严控仓位"
        else:
            return "空头排列(超跌反弹)", "大级别处于下降通道，当前属于超跌反弹"
    elif ma60 > ma20 and ma5 > ma20:
        return "筑底反弹", "中长线偏空但短期均线拐头向上，左侧资金试盘"
    else:
        return "震荡分化", "长短均线方向不一，无明显单边趋势"

def fetch_tencent_spot(codes: list[str]) -> pd.DataFrame:
    if not codes:
        return pd.DataFrame()
    batch_size = 50
    results = []
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        formatted = []
        for c in batch:
            c = str(c).zfill(6)
            if c.startswith('6'): formatted.append(f'sh{c}')
            elif c.startswith(('0', '3')): formatted.append(f'sz{c}')
            elif c.startswith(('8', '4', '9')): formatted.append(f'bj{c}')
            else: formatted.append(f'sh{c}')
        url = f"http://qt.gtimg.cn/q={','.join(formatted)}"
        try:
            resp = _session.get(url, timeout=5)
            resp.encoding = 'gbk'
            lines = resp.text.strip().split('\n')
            for line in lines:
                if '=' not in line: continue
                var, data = line.split('=', 1)
                parts = data.replace('"', '').replace(';', '').split('~')
                if len(parts) < 45: continue
                parsed = {
                    'name': parts[1],
                    'code': parts[2],
                    'price': float(parts[3]) if parts[3] else None,
                    'open': float(parts[5]) if parts[5] else None,
                    'high': float(parts[33]) if parts[33] else None,
                    'low': float(parts[34]) if parts[34] else None,
                    'pct': float(parts[32]) if parts[32] else None,
                    'vol': float(parts[36]) if parts[36] else None,
                    'amount': float(parts[37]) * 10000 if parts[37] else None,
                    'turnover': float(parts[38]) if parts[38] else 2.0,
                    'pe': float(parts[39]) if parts[39] else -1.0,
                    'volume_ratio': float(parts[49]) if len(parts) > 49 and parts[49] else 1.0,
                }
                results.append(parsed)
        except Exception as e:
            log.debug(f"腾讯批量获取失败: {e}")
    return pd.DataFrame(results)

def get_market_analysis() -> dict:
    log.info("📊 开始大盘市场分析...")
    result = {
        'sh_close': 0, 'sh_pct': 0, 'sh_trend': '', 'sh_trend_desc': '',
        'sz_close': 0, 'sz_pct': 0, 'sz_trend': '', 'sz_trend_desc': '',
        'cyb_close': 0, 'cyb_pct': 0, 'cyb_trend': '', 'cyb_trend_desc': '',
        'up_count': 0, 'down_count': 0, 'zt_count': 0, 'dt_count': 0,
        'total_amt': 0, 'breadth': 0.5, 'regime': 'NEUTRAL',
        'north_flow': 0, 'north_msg': '',
    }
    
    try:
        sh_df = get_index_data('sh000001')
        if not sh_df.empty and len(sh_df) >= 2:
            result['sh_close'] = float(sh_df['close'].iloc[-1])
            result['sh_pct'] = (sh_df['close'].iloc[-1] - sh_df['close'].iloc[-2]) / sh_df['close'].iloc[-2] * 100
            if len(sh_df) >= 60:
                result['sh_trend'], result['sh_trend_desc'] = calc_ma_trend(sh_df['close'])
            else:
                result['sh_trend'] = "数据不足"
                result['sh_trend_desc'] = "数据不足以判断趋势"
        
        sz_df = get_index_data('sz399001')
        if not sz_df.empty and len(sz_df) >= 2:
            result['sz_close'] = float(sz_df['close'].iloc[-1])
            result['sz_pct'] = (sz_df['close'].iloc[-1] - sz_df['close'].iloc[-2]) / sz_df['close'].iloc[-2] * 100
            if len(sz_df) >= 60:
                result['sz_trend'], result['sz_trend_desc'] = calc_ma_trend(sz_df['close'])
        
        cyb_df = get_index_data('sz399006')
        if not cyb_df.empty and len(cyb_df) >= 2:
            result['cyb_close'] = float(cyb_df['close'].iloc[-1])
            result['cyb_pct'] = (cyb_df['close'].iloc[-1] - cyb_df['close'].iloc[-2]) / cyb_df['close'].iloc[-2] * 100
            if len(cyb_df) >= 60:
                result['cyb_trend'], result['cyb_trend_desc'] = calc_ma_trend(cyb_df['close'])
    except Exception as e:
        log.warning(f"大盘指数分析失败: {e}")
    
    try:
        spot_df = ak.stock_zh_a_spot_em()
        if spot_df is not None and not spot_df.empty:
            pct_col = next((c for c in spot_df.columns if '涨跌幅' in c), None)
            amt_col = next((c for c in spot_df.columns if '成交额' in c), None)
            if pct_col:
                spot_df[pct_col] = pd.to_numeric(spot_df[pct_col], errors='coerce')
                result['up_count'] = int((spot_df[pct_col] > 0).sum())
                result['down_count'] = int((spot_df[pct_col] < 0).sum())
                result['zt_count'] = int((spot_df[pct_col] >= 9.0).sum())
                result['dt_count'] = int((spot_df[pct_col] <= -9.0).sum())
                total = result['up_count'] + result['down_count']
                result['breadth'] = result['up_count'] / total if total > 0 else 0.5
            if amt_col:
                result['total_amt'] = float(pd.to_numeric(spot_df[amt_col], errors='coerce').sum()) / 1e8
            log.info("✅ 东方财富实时行情获取成功")
    except Exception as e1:
        log.debug(f"东方财富实时行情失败: {e1}，尝试新浪/腾讯备用源...")
        try:
            sample_codes = ['600519', '601318', '000001', '000858', '300750', 
                           '600036', '601012', '002594', '601899', '000333']
            sample_df = fetch_tencent_spot(sample_codes)
            if not sample_df.empty:
                sample_df['pct'] = pd.to_numeric(sample_df['pct'], errors='coerce')
                up = int((sample_df['pct'] > 0).sum())
                total = len(sample_df)
                result['breadth'] = up / max(total, 1)
                log.info(f"⚠️ 使用样本股估算市场广度: {up}/{total}")
        except Exception as e2:
            log.warning(f"备用行情源也失败: {e2}")
    
    try:
        df = ak.stock_em_hsgt_north_net_flow_in(indicator="沪深港通")
        if df is not None and not df.empty:
            col = 'value' if 'value' in df.columns else df.columns[-1]
            result['north_flow'] = float(df.iloc[-1][col]) / 1e8
            if result['north_flow'] > 30:
                result['north_msg'] = f"北水大举流入 +{result['north_flow']:.0f}亿"
            elif result['north_flow'] < -30:
                result['north_msg'] = f"北水大幅流出 {result['north_flow']:.0f}亿"
            else:
                result['north_msg'] = f"北向资金温和 ({result['north_flow']:+.0f}亿)"
    except Exception as e:
        log.debug(f"北向资金获取失败: {e}")
        result['north_msg'] = "北向资金数据暂不可用"
    
    if result['breadth'] < 0.25:
        result['regime'] = 'PANIC'
    elif result['sh_trend'] and '多头' in result['sh_trend'] and result['breadth'] > 0.6:
        result['regime'] = 'BULL'
    elif '空头' in (result['sh_trend'] or '') and result['breadth'] <= 0.4:
        result['regime'] = 'BEAR'
    else:
        result['regime'] = 'NEUTRAL'
    
    return result

def get_stock_hist(code: str) -> pd.DataFrame:
    market_prefix = 'sh' if code.startswith(('6', '5', '9')) else 'sz'
    if code.startswith(('8', '4')):
        market_prefix = 'bj'
    symbol = f'{market_prefix}{code}'
    try:
        df = ak.stock_zh_a_daily(symbol=symbol, adjust='qfq')
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
    except Exception:
        pass
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                    start_date=(datetime.now() - timedelta(days=180)).strftime('%Y%m%d'),
                    end_date=datetime.now().strftime('%Y%m%d'),
                    adjust="qfq")
        if df is not None and not df.empty:
            col_map = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'}
            df = df.rename(columns=col_map)
            return df
    except Exception:
        pass
    return pd.DataFrame()

def get_etf_hist(code: str) -> pd.DataFrame:
    market_prefix = 'sh' if code.startswith(('5', '6', '9')) else 'sz'
    symbol = f'{market_prefix}{code}'
    try:
        df = ak.fund_etf_hist_sina(symbol=symbol)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
    except Exception:
        pass
    try:
        df = ak.fund_etf_hist_em(symbol=code, period="daily", adjust="qfq")
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            return df
    except Exception:
        pass
    return pd.DataFrame()

def get_etf_rotation() -> list[dict]:
    log.info("📈 开始ETF轮动分析...")
    etf_list = []
    
    for code, name in ETF_UNIVERSE.items():
        try:
            df = get_etf_hist(code)
            if df is None or df.empty or len(df) < 60:
                continue
            
            close = pd.to_numeric(df['close'], errors='coerce').dropna()
            vol = pd.to_numeric(df.get('volume', pd.Series(1, index=close.index)), errors='coerce').dropna()
            if len(close) < 60:
                continue
            
            ma5 = close.rolling(5).mean().iloc[-1]
            ma10 = close.rolling(10).mean().iloc[-1]
            ma20 = close.rolling(20).mean().iloc[-1]
            ma60 = close.rolling(60).mean().iloc[-1]
            
            pct_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
            pct_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0
            pct_60d = (close.iloc[-1] / close.iloc[-61] - 1) * 100 if len(close) >= 61 else 0
            
            score = 0
            trend = ""
            
            if ma5 > ma10 > ma20 > ma60:
                score += 40
                trend = "强势多头"
            elif ma5 > ma20 > ma60:
                score += 30
                trend = "多头排列"
            elif ma5 < ma10 < ma20 < ma60:
                score += 0
                trend = "空头瀑布"
            elif ma5 < ma20 < ma60:
                score += 10
                trend = "弱势空头"
            else:
                score += 20
                trend = "震荡整理"
            
            if pct_5d > 0:
                score += min(pct_5d * 2, 20)
            else:
                score += max(pct_5d * 2, -15)
            
            if pct_20d > 0:
                score += min(pct_20d * 1.5, 25)
            else:
                score += max(pct_20d * 1.5, -20)
            
            if pct_60d > 0:
                score += min(pct_60d, 15)
            else:
                score += max(pct_60d * 0.5, -15)
            
            last_close = float(close.iloc[-1])
            last_pct = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100 if len(close) >= 2 else 0
            
            etf_list.append({
                'code': code,
                'name': name,
                'price': round(last_close, 3),
                'pct': round(last_pct, 2),
                'pct_5d': round(pct_5d, 2),
                'pct_20d': round(pct_20d, 2),
                'pct_60d': round(pct_60d, 2),
                'trend': trend,
                'score': int(round(score)),
                'above_ma20': last_close > ma20,
                'above_ma60': last_close > ma60,
            })
        except Exception as e:
            log.debug(f"ETF {code}({name}) 分析失败: {e}")
            continue
    
    etf_list.sort(key=lambda x: x['score'], reverse=True)
    return etf_list

def get_industry_hotspots() -> tuple[list[dict], list[dict]]:
    log.info("🔥 开始行业热点分析...")
    
    up_sectors = []
    down_sectors = []
    
    try:
        df = ak.stock_board_industry_name_em()
        if df is not None and not df.empty:
            name_col = next((c for c in df.columns if '板块名称' in c or '板块' in c), None)
            pct_col = next((c for c in df.columns if '涨跌幅' in c), None)
            if name_col and pct_col:
                df[pct_col] = pd.to_numeric(df[pct_col], errors='coerce')
                df_sorted = df.sort_values(pct_col, ascending=False)
                
                for _, row in df_sorted.head(10).iterrows():
                    try:
                        cons = ak.stock_board_industry_cons_em(symbol=row[name_col])
                        leader_name = ""
                        leader_pct = 0
                        if cons is not None and not cons.empty:
                            c_name_col = next((c for c in cons.columns if '名称' in c), None)
                            c_pct_col = next((c for c in cons.columns if '涨跌幅' in c), None)
                            if c_name_col and c_pct_col:
                                cons[c_pct_col] = pd.to_numeric(cons[c_pct_col], errors='coerce')
                                top_stock = cons.nlargest(1, c_pct_col).iloc[0]
                                leader_name = str(top_stock[c_name_col])
                                leader_pct = float(top_stock[c_pct_col])
                        
                        up_sectors.append({
                            'name': row[name_col],
                            'pct': round(float(row[pct_col]), 2),
                            'leader': leader_name,
                            'leader_pct': round(leader_pct, 2),
                        })
                    except Exception:
                        up_sectors.append({
                            'name': row[name_col],
                            'pct': round(float(row[pct_col]), 2),
                            'leader': '',
                            'leader_pct': 0,
                        })
                
                for _, row in df_sorted.tail(5).iterrows():
                    down_sectors.append({
                        'name': row[name_col],
                        'pct': round(float(row[pct_col]), 2),
                    })
                return up_sectors, down_sectors
    except Exception as e:
        log.debug(f"东方财富行业板块分析失败: {e}")
    
    try:
        log.info("尝试同花顺行业板块备用源...")
        df = ak.stock_board_industry_name_ths()
        if df is not None and not df.empty:
            name_col = next((c for c in df.columns if '板块' in c or 'name' in c.lower()), None)
            pct_col = next((c for c in df.columns if '涨跌' in c or 'pct' in c.lower()), None)
            if name_col and pct_col:
                df[pct_col] = pd.to_numeric(df[pct_col], errors='coerce')
                df_sorted = df.sort_values(pct_col, ascending=False)
                
                for _, row in df_sorted.head(10).iterrows():
                    up_sectors.append({
                        'name': str(row[name_col]),
                        'pct': round(float(row[pct_col]), 2),
                        'leader': '',
                        'leader_pct': 0,
                    })
                
                for _, row in df_sorted.tail(5).iterrows():
                    down_sectors.append({
                        'name': str(row[name_col]),
                        'pct': round(float(row[pct_col]), 2),
                    })
                return up_sectors, down_sectors
    except Exception as e:
        log.debug(f"同花顺行业板块也失败: {e}")
    
    log.warning("所有行业板块数据源都失败")
    return up_sectors, down_sectors

def get_stock_signals(top_n: int = 10) -> list[dict]:
    log.info("🎯 开始股票信号扫描...")
    signals = []
    
    core_pool = [
        '600519', '601318', '000858', '000333', '600036', '601166',
        '002594', '601012', '601899', '300750', '600276', '300760',
        '002415', '600030', '600900', '000568', '002304', '600887',
        '601398', '601288', '601939', '601988', '600000', '601328',
        '000001', '000002', '300015', '300059', '600104', '600690',
        '601668', '601857', '601088', '600028', '000651', '002475',
        '600438', '600585', '601111', '000157', '002142',
        '002271', '300122', '600809', '300274', '002714', '603288',
    ]
    
    spot_df = None
    
    try:
        log.info("获取新浪实时行情...")
        spot_all = ak.stock_zh_a_spot()
        if spot_all is not None and not spot_all.empty:
            code_col = next((c for c in spot_all.columns if '代码' in c), None)
            name_col = next((c for c in spot_all.columns if '名称' in c), None)
            pct_col = next((c for c in spot_all.columns if '涨跌幅' in c), None)
            price_col = next((c for c in spot_all.columns if '最新价' in c), None)
            vol_col = next((c for c in spot_all.columns if '成交量' in c), None)
            amt_col = next((c for c in spot_all.columns if '成交额' in c), None)
            
            if all([code_col, name_col, pct_col, price_col]):
                spot_all['_code_clean'] = spot_all[code_col].astype(str).str.replace(r'^(sh|sz|bj)', '', regex=True)
                spot_filtered = spot_all[spot_all['_code_clean'].isin(core_pool)].copy()
                spot_df = pd.DataFrame({
                    'code': spot_filtered['_code_clean'],
                    'name': spot_filtered[name_col],
                    'price': pd.to_numeric(spot_filtered[price_col], errors='coerce'),
                    'pct': pd.to_numeric(spot_filtered[pct_col], errors='coerce'),
                    'volume': pd.to_numeric(spot_filtered.get(vol_col, 0), errors='coerce') if vol_col else 0,
                    'amount': pd.to_numeric(spot_filtered.get(amt_col, 0), errors='coerce') if amt_col else 0,
                })
                log.info(f"✅ 新浪实时行情获取成功，共 {len(spot_df)} 只核心股票")
    except Exception as e:
        log.debug(f"新浪实时行情失败: {e}")
    
    if spot_df is None or spot_df.empty:
        try:
            log.info("尝试腾讯实时行情...")
            spot_df = fetch_tencent_spot(core_pool)
        except Exception as e:
            log.debug(f"腾讯实时行情也失败: {e}")
    
    if spot_df is None or spot_df.empty:
        try:
            log.info("尝试东方财富实时行情...")
            spot_em = ak.stock_zh_a_spot_em()
            if spot_em is not None and not spot_em.empty:
                code_col = next((c for c in spot_em.columns if '代码' in c), None)
                name_col = next((c for c in spot_em.columns if '名称' in c), None)
                pct_col = next((c for c in spot_em.columns if '涨跌幅' in c), None)
                price_col = next((c for c in spot_em.columns if '最新价' in c), None)
                if all([code_col, name_col, pct_col, price_col]):
                    spot_df = pd.DataFrame({
                        'code': spot_em[code_col].astype(str).str.zfill(6),
                        'name': spot_em[name_col],
                        'price': pd.to_numeric(spot_em[price_col], errors='coerce'),
                        'pct': pd.to_numeric(spot_em[pct_col], errors='coerce'),
                    })
                    spot_df = spot_df[spot_df['code'].isin(core_pool)].reset_index(drop=True)
        except Exception as e:
            log.debug(f"东方财富行情也失败: {e}")
    
    if spot_df is None or spot_df.empty:
        log.warning("所有实时行情源都失败，无法生成股票信号")
        return signals
    
    spot_df['pct'] = pd.to_numeric(spot_df['pct'], errors='coerce')
    spot_df['price'] = pd.to_numeric(spot_df['price'], errors='coerce')
    
    mask = (spot_df['pct'] > -3.0) & (spot_df['pct'] < 7.0) & (spot_df['price'] > 2) & (spot_df['price'] < 200)
    filtered = spot_df[mask].copy()
    if filtered.empty:
        return signals
    
    candidates = filtered.nlargest(30, 'pct')
    
    for _, row in candidates.iterrows():
        code = str(row['code']).zfill(6)
        name = str(row.get('name', code))
        try:
            hist_df = get_stock_hist(code)
            if hist_df is None or hist_df.empty or len(hist_df) < 30:
                continue
            
            close = pd.to_numeric(hist_df['close'], errors='coerce').dropna()
            high = pd.to_numeric(hist_df.get('high', close), errors='coerce').dropna()
            low = pd.to_numeric(hist_df.get('low', close), errors='coerce').dropna()
            vol = pd.to_numeric(hist_df.get('volume', pd.Series(1, index=close.index)), errors='coerce').dropna()
            
            if len(close) < 30:
                continue
            
            ma5 = close.rolling(5).mean()
            ma10 = close.rolling(10).mean()
            ma20 = close.rolling(20).mean()
            ma60 = close.rolling(60).mean() if len(close) >= 60 else ma20
            
            last_close = close.iloc[-1]
            last_ma5 = ma5.iloc[-1]
            last_ma20 = ma20.iloc[-1]
            last_ma60 = ma60.iloc[-1]
            
            if last_close < last_ma5:
                continue
            if last_close < last_ma20 * 0.95:
                continue
            
            atr = pd.Series(0.0, index=close.index)
            for i in range(1, len(close)):
                tr = max(high.iloc[i] - low.iloc[i], 
                         abs(high.iloc[i] - close.iloc[i-1]), 
                         abs(low.iloc[i] - close.iloc[i-1]))
                atr.iloc[i] = tr
            atr14 = atr.rolling(14).mean().iloc[-1]
            
            if atr14 <= 0 or atr14 / last_close > 0.08:
                continue
            
            stop_loss = round(last_close - 1.5 * atr14, 2)
            target1 = round(last_close + 2.0 * atr14, 2)
            
            score = 50
            reasons = []
            
            if last_ma5 > last_ma20 > last_ma60:
                score += 20
                reasons.append("多头排列(MA5>MA20>MA60)")
            elif last_ma5 > last_ma20:
                score += 10
                reasons.append("短期均线多头")
            
            if len(vol) >= 20:
                vol_ma5 = vol.rolling(5).mean().iloc[-1]
                vol_ma20 = vol.rolling(20).mean().iloc[-1]
                if vol.iloc[-1] > vol_ma5 * 1.2:
                    score += 10
                    reasons.append("放量突破")
                elif vol.iloc[-1] < vol_ma20 * 0.7:
                    score -= 5
                    reasons.append("缩量整理")
            
            rsi = 50.0
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi_series = 100 - (100 / (1 + rs))
            rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
            
            if 40 < rsi < 70:
                score += 10
                reasons.append(f"RSI健康({rsi:.0f})")
            elif rsi >= 75:
                score -= 10
                reasons.append(f"RSI超买({rsi:.0f})")
            
            pct = float(row.get('pct', 0))
            
            score = max(0, min(100, score))
            
            signals.append({
                'code': code,
                'name': name,
                'price': round(last_close, 2),
                'pct': round(pct, 2),
                'score': score,
                'stop_loss': stop_loss,
                'target1': target1,
                'reasons': '、'.join(reasons) if reasons else '形态良好',
                'ma20': round(last_ma20, 2),
                'atr_pct': round(atr14 / last_close * 100, 1),
            })
            
            if len(signals) >= top_n * 2:
                break
        except Exception as e:
            log.debug(f"股票 {code}({name}) 分析失败: {e}")
            continue
    
    signals.sort(key=lambda x: x['score'], reverse=True)
    return signals[:top_n]

def generate_briefing_content() -> str:
    now_str = datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M')
    
    market = get_market_analysis()
    time.sleep(1)
    etfs = get_etf_rotation()
    time.sleep(1)
    up_sectors, down_sectors = get_industry_hotspots()
    time.sleep(1)
    signals = get_stock_signals(top_n=8)
    
    regime_emoji = {
        'BULL': '🔥',
        'BEAR': '🐻',
        'PANIC': '🧊',
        'NEUTRAL': '⚖️',
    }.get(market['regime'], '⚖️')
    
    regime_name = {
        'BULL': '强势多头',
        'BEAR': '弱势空头',
        'PANIC': '恐慌冰点',
        'NEUTRAL': '震荡均衡',
    }.get(market['regime'], '震荡均衡')
    
    content = f"# 📊 每日投研简报\n> **{now_str}**\n\n"
    
    content += "## 🌡️ 大盘温度与市场状态\n\n"
    content += f"- **综合判定**：{regime_emoji} **{regime_name}**\n"
    content += f"- **上证指数**：`{market['sh_close']:.2f}` ({market['sh_pct']:+.2f}%) — {market['sh_trend']}\n"
    content += f"- **深证成指**：`{market['sz_close']:.2f}` ({market['sz_pct']:+.2f}%) — {market['sz_trend']}\n"
    content += f"- **创业板指**：`{market['cyb_close']:.2f}` ({market['cyb_pct']:+.2f}%) — {market['cyb_trend']}\n"
    content += f"- **市场广度**：红盘 `{market['up_count']}` 家 / 绿盘 `{market['down_count']}` 家\n"
    content += f"- **涨跌停**：涨停 `{market['zt_count']}` 家 / 跌停 `{market['dt_count']}` 家\n"
    content += f"- **两市成交**：约 `{market['total_amt']:.0f}` 亿元\n"
    content += f"- **北向资金**：{market['north_msg']}\n\n"
    
    content += "## 🔥 行业热点与主线\n\n"
    if up_sectors:
        content += "### 领涨板块 Top 10\n\n"
        for i, s in enumerate(up_sectors[:10], 1):
            leader_str = f" 龙头: **{s['leader']}** ({s['leader_pct']:+.2f}%)" if s['leader'] else ""
            content += f"{i}. **{s['name']}** — 涨幅 `{s['pct']:+.2f}%`{leader_str}\n"
        content += "\n"
    else:
        content += "⚠️ 行业热点数据暂不可用\n\n"
    
    if down_sectors:
        content += "### 领跌板块\n\n"
        for s in down_sectors:
            content += f"- **{s['name']}** — 跌幅 `{s['pct']:+.2f}%`\n"
        content += "\n"
    
    content += "## 📈 ETF轮动与配置建议\n\n"
    if etfs:
        top_bull = [e for e in etfs if e['score'] >= 60][:5]
        if top_bull:
            content += "### 🚀 强势多头ETF (建议关注)\n\n"
            content += "| 代码 | 名称 | 现价 | 日涨跌 | 5日涨 | 20日涨 | 60日涨 | 趋势 |\n"
            content += "|------|------|------|--------|-------|--------|--------|------|\n"
            for e in top_bull:
                content += f"| {e['code']} | {e['name']} | ¥{e['price']} | {e['pct']:+.2f}% | {e['pct_5d']:+.2f}% | {e['pct_20d']:+.2f}% | {e['pct_60d']:+.2f}% | {e['trend']} |\n"
            content += "\n"
        
        bottom_bear = [e for e in etfs if e['score'] < 30][-5:]
        if bottom_bear:
            content += "### ⚠️ 弱势ETF (建议规避)\n\n"
            content += "| 代码 | 名称 | 现价 | 日涨跌 | 20日涨 | 趋势 |\n"
            content += "|------|------|------|--------|--------|------|\n"
            for e in bottom_bear:
                content += f"| {e['code']} | {e['name']} | ¥{e['price']} | {e['pct']:+.2f}% | {e['pct_20d']:+.2f}% | {e['trend']} |\n"
            content += "\n"
        
        content += "### 💡 ETF配置建议\n\n"
        if market['regime'] == 'BULL':
            content += "当前市场处于**多头行情**，建议：\n"
            content += "- 仓位：**70%-80%** 权益类 ETF\n"
            content += "- 配置：优先宽基 + 强势行业主题\n"
            content += "- 策略：趋势跟随，回调至 MA20 加仓\n"
        elif market['regime'] == 'BEAR':
            content += "当前市场处于**空头行情**，建议：\n"
            content += "- 仓位：**20%-30%** 权益类 ETF，配置防御性资产\n"
            content += "- 配置：黄金ETF、国债ETF + 少量宽基\n"
            content += "- 策略：严控仓位，反弹减仓，不抄底\n"
        elif market['regime'] == 'PANIC':
            content += "当前市场处于**恐慌冰点**，建议：\n"
            content += "- 仓位：**10%-20%** 极轻仓试错\n"
            content += "- 配置：以现金、黄金、国债为主\n"
            content += "- 策略：多看少动，等待情绪修复信号\n"
        else:
            content += "当前市场处于**震荡均衡**，建议：\n"
            content += "- 仓位：**40%-60%** 灵活配置\n"
            content += "- 配置：宽基打底 + 行业轮动\n"
            content += "- 策略：高抛低吸，不追涨杀跌\n"
        content += "\n"
    else:
        content += "⚠️ ETF轮动数据暂不可用\n\n"
    
    content += "## 🎯 股票精选信号\n\n"
    if signals:
        content += "### 精选潜力股 Top 8\n\n"
        for i, s in enumerate(signals, 1):
            risk_pct = ((s['price'] - s['stop_loss']) / s['price']) * 100
            reward_pct = ((s['target1'] - s['price']) / s['price']) * 100
            rr_ratio = reward_pct / max(risk_pct, 0.1)
            
            star_level = "⭐⭐⭐⭐⭐" if s['score'] >= 85 else "⭐⭐⭐⭐" if s['score'] >= 75 else "⭐⭐⭐"
            
            content += f"#### {i}. {s['name']} (`{s['code']}`) {star_level}\n\n"
            content += f"- **当前价**：`¥{s['price']}` ({s['pct']:+.2f}%) | 综合评分：**{s['score']}分**\n"
            content += f"- **目标价**：`¥{s['target1']}` (潜在收益 `+{reward_pct:.1f}%`)\n"
            content += f"- **止损价**：`¥{s['stop_loss']}` (风险 `-{risk_pct:.1f}%`)\n"
            content += f"- **盈亏比**：`1:{rr_ratio:.1f}` | ATR波动：`{s['atr_pct']}%`\n"
            content += f"- **触发逻辑**：{s['reasons']}\n\n"
        
        content += "> ⚠️ **风险提示**：以上信号仅供参考，不构成投资建议。请结合自身风险承受能力审慎决策，严格执行止损纪律。\n\n"
    else:
        content += "✅ 今日未发现形态完全符合安全边际的标的，建议空仓防守。\n\n"
    
    content += "---\n\n"
    content += f"*本简报由 AI 量化系统自动生成，数据来源：东方财富、AkShare。生成时间：{now_str}*"
    
    return content

def main():
    log.info("🚀 每日投研简报生成开始...")
    
    try:
        content = generate_briefing_content()
        
        with open('daily_briefing_output.md', 'w', encoding='utf-8') as f:
            f.write(content)
        log.info("📝 简报已保存到 daily_briefing_output.md")
        
        title = "📊 每日投研简报"
        success = send_dingtalk_notification(title, content)
        
        if success:
            log.info("🎉 每日投研简报已成功推送至钉钉")
        else:
            log.warning("⚠️ 钉钉推送失败，请检查配置")
            
    except Exception as e:
        log.error(f"❌ 生成简报失败: {e}", exc_info=True)
        error_msg = f"# ❌ 每日投研简报生成失败\n\n**错误信息**：{e}\n\n请检查数据源状态或稍后重试。"
        send_dingtalk_notification("❌ 简报生成失败", error_msg)

if __name__ == '__main__':
    main()
