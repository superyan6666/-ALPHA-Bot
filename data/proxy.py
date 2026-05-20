
"""
数据获取代理模块
"""
import os
import time
import random
import json
import logging
from urllib.parse import urlparse
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Tuple, Set, Any
import pandas as pd
import numpy as np
import requests

from core.config import Cols as C, Config

# 日志配置
log = logging.getLogger(__name__)

# WAF 旁路 Session 类
class WAFBypassSession(requests.Session):
    def request(self, method, url, **kwargs):
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        
        whitelist_domains = ('eastmoney.com', 'dfcfw.com', 'sina.com.cn', 
                           'sinajs.cn', 'money.163.com', '10jqka.com.cn', 
                           'tushare.pro')
        needs_patch = any(hostname == d or hostname.endswith('.' + d) 
                         for d in whitelist_domains)
        
        if needs_patch:
            log.debug(f"[WAF Patch] 注入浏览器 UA -> {hostname}")
            headers = kwargs.get('headers', {})
            if not isinstance(headers, dict):
                headers = dict(headers)
            if 'User-Agent' not in headers and 'user-agent' not in headers:
                headers['User-Agent'] = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                     'AppleWebKit/537.36 (KHTML, like Gecko) '
                                     'Chrome/124.0.0.0 Safari/537.36')
            kwargs['headers'] = headers
        else:
            log.debug(f"[WAF Patch] 原生放行 -> {hostname}")
            
        kwargs['timeout'] = kwargs.get('timeout', 15.0)
        return super().request(method, url, **kwargs)

requests.Session = WAFBypassSession

# 重试装饰器
def retry(times=4, delay=2, exceptions=(Exception,)):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt < times - 1:
                        log.debug(f"抓取受挫，将在 {delay * (2 ** attempt)}s 后重试: {e}")
                        time.sleep(delay * (2 ** attempt))
                    else:
                        raise
        return wrapper
    return decorator


class DataProxy:
    """数据获取多源路由层 (Fallback Waterfall)"""
    def __init__(self, config=None):
        self.config = config or Config()
        self.bs_logged_in = False
        self.ts_pro = None
        
        import os
        ts_token = os.getenv('TUSHARE_TOKEN')
        if ts_token:
            try:
                import tushare as ts
                ts.set_token(ts_token)
                self.ts_pro = ts.pro_api()
                log.info("🌟 成功挂载 Tushare Pro 机构级数据核心")
            except Exception as e:
                log.warning(f"Tushare 初始化失败: {e}")

    def __del__(self):
        if self.bs_logged_in:
            try:
                import baostock as bs
                bs.logout()
            except:
                pass

    def _login_baostock(self):
        import baostock as bs
        if not self.bs_logged_in:
            bs.login()
            self.bs_logged_in = True

    # ---- [1. Historical Data] ----
    def _fetch_hist_tushare(self, code, start, end):
        if not self.ts_pro:
            return None
        try:
            import tushare as ts
            ts_code = f"{code}.SH" if code.startswith(('6', '5')) else f"{code}.SZ"
            start_fmt = f"{start[:4]}{start[4:6]}{start[6:]}"
            end_fmt = f"{end[:4]}{end[4:6]}{end[6:]}"
            asset_type = 'FD' if code.startswith(('51', '15', '588', '56')) else 'E'
            df_adj = ts.pro_bar(ts_code=ts_code, api=self.ts_pro, 
                              start_date=start_fmt, end_date=end_fmt, 
                              adj='qfq', asset=asset_type)
            if df_adj is None or df_adj.empty:
                return None
            
            df_adj = df_adj.rename(columns={'trade_date': C.H_DATE, 
                                          'open': C.H_OPEN, 'close': C.H_CLOSE, 
                                          'high': C.H_HIGH, 'low': C.H_LOW, 
                                          'vol': C.H_VOL})
            df_adj[C.H_DATE] = pd.to_datetime(df_adj[C.H_DATE]).dt.strftime('%Y-%m-%d')
            df_adj = df_adj.sort_values(C.H_DATE).reset_index(drop=True)
            for col in [C.H_OPEN, C.H_CLOSE, C.H_HIGH, C.H_LOW, C.H_VOL]:
                df_adj[col] = pd.to_numeric(df_adj[col], errors='coerce')
            return df_adj[list(self.config.HIST_COLS)]
        except Exception as e:
            log.debug(f"[Tier 1 Tushare] 获取历史失败: {e}")
            return None

    def _get_tushare_fundamentals_df(self) -> pd.DataFrame:
        if not self.ts_pro:
            return pd.DataFrame()
        try:
            for days_back in range(1, 10):
                trade_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
                df = self.ts_pro.daily_basic(trade_date=trade_date)
                if df is not None and not df.empty:
                    df[C.S_CODE] = df['ts_code'].str.slice(0, 6)
                    df[C.S_TURN] = df.get('turnover_rate_f', 
                                       df.get('turnover_rate', 2.0)).astype(float)
                    df[C.S_VR] = df.get('volume_ratio', 1.0).astype(float)
                    df[C.S_PE] = df.get('pe_ttm', -1.0).astype(float)
                    df[C.S_PB] = df.get('pb', 2.0).astype(float)
                    df[C.S_MCAP] = df.get('circ_mv', 0.0).astype(float) * 10000
                    return df[[C.S_CODE, C.S_TURN, C.S_VR, C.S_PE, C.S_PB, C.S_MCAP]]
        except Exception as e:
            log.debug(f"Tushare 向量化获取基本面失败: {e}")
        return pd.DataFrame()

    def _fetch_hist_baostock(self, code, start, end):
        import baostock as bs
        if bs is None:
            return None
        self._login_baostock()
        try:
            prefix = 'sh.' if code.startswith('6') else 'sz.'
            start_fmt = f"{start[:4]}-{start[4:6]}-{start[6:]}"
            end_fmt = f"{end[:4]}-{end[4:6]}-{end[6:]}"
            rs = bs.query_history_k_data_plus(prefix + code,
                "date,open,close,high,low,volume",
                start_date=start_fmt, end_date=end_fmt,
                frequency="d", adjustflag="1")
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            if not data_list:
                return None
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            df = df.rename(columns={'date': C.H_DATE, 'open': C.H_OPEN, 
                                  'close': C.H_CLOSE, 'high': C.H_HIGH, 
                                  'low': C.H_LOW, 'volume': C.H_VOL})
            for col in [C.H_OPEN, C.H_CLOSE, C.H_HIGH, C.H_LOW, C.H_VOL]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df[list(self.config.HIST_COLS)]
        except Exception as e:
            log.debug(f"[Tier 2 BaoStock] 获取历史失败: {e}")
            return None

    @retry(times=3, delay=2)
    def _fetch_hist_akshare(self, code, start, end):
        import akshare as ak
        try:
            df = ak.stock_zh_a_hist(symbol=code, period='daily', 
                                  start_date=start, end_date=end, adjust='qfq')
            if df is not None and not df.empty:
                return df[list(self.config.HIST_COLS)].copy()
        except Exception as e:
            try:
                df = ak.stock_zh_a_hist_tx(symbol=code, 
                                        start_date=start, end_date=end, 
                                        adjust='qfq')
                if df is not None and not df.empty:
                    col_map = {'日期': C.H_DATE, '开盘': C.H_OPEN, 
                             '收盘': C.H_CLOSE, '最高': C.H_HIGH, 
                             '最低': C.H_LOW, '成交量': C.H_VOL}
                    df = df.rename(columns=col_map)
                    return df[list(self.config.HIST_COLS)].copy()
            except Exception:
                pass
            raise ValueError(f'akshare history empty for {code}')
            
    def get_hist(self, code, start, end) -> pd.DataFrame:
        df = self._fetch_hist_tushare(code, start, end)
        if df is not None:
            return df
        df = self._fetch_hist_baostock(code, start, end)
        if df is not None:
            return df
        return self._fetch_hist_akshare(code, start, end)

    # ---- [2. Spot Data (实时横截面)] ----
    def _fetch_spot_qmt(self):
        return None

    def _fetch_spot_efinance(self):
        try:
            import efinance as ef
            df = ef.stock.get_realtime_quotes()
            if df is not None and not df.empty:
                rename_map = {'代码': C.S_CODE, '名称': C.S_NAME, '最新价': C.S_PRICE,
                             '涨跌幅': C.S_PCT, '今开': C.S_OPEN, '最高': C.S_HIGH,
                             '最低': C.S_LOW, '成交量': C.S_VOL, '成交额': C.S_AMT,
                             '换手率': C.S_TURN, '市盈率-动态': C.S_PE, 
                             '市净率': C.S_PB, '量比': C.S_VR}
                df = df.rename(columns=rename_map)
                return df
        except Exception as e:
            log.debug(f"[Tier 2 efinance] 获取实时行情失败: {e}")
        return None

    @retry(times=3, delay=5)
    def _fetch_spot_akshare(self):
        import akshare as ak
        try:
            time.sleep(random.uniform(1.0, 3.0))
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                return df
        except Exception as e:
            log.warning(f"行情主接口异常: {e}，正在启动新浪备用源执行优雅降级...")
            df = ak.stock_zh_a_spot()
            if df is not None and not df.empty:
                rename_map = {'代码': C.S_CODE, '名称': C.S_NAME, '最新价': C.S_PRICE,
                            '涨跌幅': C.S_PCT, '今开': C.S_OPEN, '最高': C.S_HIGH,
                            '最低': C.S_LOW, '成交量': C.S_VOL, '成交额': C.S_AMT}
                df = df.rename(columns=rename_map)
                
                funds_df = self._get_tushare_fundamentals_df()
                if not funds_df.empty:
                    df = pd.merge(df, funds_df, on=C.S_CODE, how='left')
                    df[C.S_TURN] = df[C.S_TURN].fillna(2.0)
                    df[C.S_MCAP] = df[C.S_MCAP].fillna(100e8)
                    df[C.S_PE] = df[C.S_PE].fillna(-1.0)
                    df[C.S_PB] = df[C.S_PB].fillna(2.0)
                    df[C.S_VR] = df[C.S_VR].fillna(1.0)
                    log.info("💎 已通过 Tushare 成功向量化修复 Sina 备用源缺失的 PE/VR 等基本面数据。")
                else:
                    fallback_defaults = {C.S_TURN: 2.0, C.S_MCAP: 100e8, 
                                      C.S_PE: -1.0, C.S_PB: 2.0, C.S_VR: 1.0}
                    for col, val in fallback_defaults.items():
                        if col not in df.columns:
                            df[col] = val
                return df
            raise ValueError('spot_empty')

    def _fetch_spot_tushare_fallback(self) -> pd.DataFrame:
        """极寒时刻的终极兜底：当全市场实时接口死掉，用日频历史伪装截面"""
        if not self.ts_pro:
            raise ValueError("Tushare 未配置，终极兜底失败！网络严重受阻。")
        log.warning("⚠️ [FALLBACK] 所有实时数据源失效，正尝试使用 Tushare 日终历史充当伪装截面数据！")
        
        try:
            for days_back in range(0, 10):
                trade_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
                df = self.ts_pro.daily(trade_date=trade_date)
                if df is not None and not df.empty:
                    df_basic = self.ts_pro.daily_basic(trade_date=trade_date)
                    if df_basic is not None and not df_basic.empty:
                        df = pd.merge(df, df_basic, on='ts_code', how='left')
                    
                    df[C.S_CODE] = df['ts_code'].str.slice(0, 6)
                    df[C.S_PRICE] = df['close']
                    df[C.S_OPEN] = df['open']
                    df[C.S_HIGH] = df['high']
                    df[C.S_LOW] = df['low']
                    df[C.S_VOL] = df['vol']
                    df[C.S_AMT] = df['amount'] * 1000
                    df[C.S_PCT] = 0.0  # 强制置零，防止使用T-1的涨跌幅误导下游策略
                    df[C.S_TURN] = df.get('turnover_rate', df.get('turnover_rate_f', 2.0))
                    df[C.S_PE] = df.get('pe_ttm', -1.0)
                    df[C.S_PB] = df.get('pb', 2.0)
                    df[C.S_MCAP] = df.get('circ_mv', 0.0) * 10000
                    df[C.S_VR] = df.get('volume_ratio', 1.0)
                    df[C.S_NAME] = "TS_" + df[C.S_CODE]
                    df['DATA_MODE'] = 'T+1_FALLBACK'
                    log.warning(f"⚠️ [FALLBACK_SUCCESS] 成功拉取 {trade_date} Tushare数据作为伪装截面。注意时效性！")
                    return df
            raise ValueError("Tushare returned empty daily data across 10 days.")
        except Exception as e:
            log.error(f"❌ [FATAL] Tushare 终极兜底方案也失败，彻底断网: {e}")
            raise e

    def get_spot(self) -> pd.DataFrame:
        df = self._fetch_spot_qmt()
        if df is not None:
            return df
        df = self._fetch_spot_efinance()
        if df is not None:
            return df
        try:
            df = self._fetch_spot_akshare()
            if df is not None:
                return df
        except Exception as e:
            log.debug(f"akshare spot failed: {e}")
        return self._fetch_spot_tushare_fallback()

    # ---- [3. Index & Context] ----
    @retry(times=4, delay=2)
    def get_index(self, symbol: str) -> pd.DataFrame:
        import akshare as ak
        try:
            df = ak.stock_zh_index_daily_tx(symbol=symbol)
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
                return df
        except Exception as e:
            log.warning(f"腾讯指数接口波动 ({symbol}): {e}，切换东方财富源...")
        
        try:
            df = ak.stock_zh_index_daily_em(symbol=symbol)
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
                return df
        except Exception as e:
            log.warning(f"东方财富指数接口也波动 ({symbol}): {e}，切换 Baostock 源...")
            
        import baostock as bs
        if bs is not None:
            self._login_baostock()
            bs_symbol = 'sh.' + symbol if symbol.startswith('0') else 'sz.' + symbol
            start_fmt = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            rs = bs.query_history_k_data_plus(bs_symbol, 
                                             "date,open,close,high,low,volume", 
                                             start_date=start_fmt, frequency="d")
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            if data_list:
                df = pd.DataFrame(data_list, columns=rs.fields)
                df = df.rename(columns={'date': 'date', 'open': 'open', 
                                      'close': 'close', 'high': 'high', 
                                      'low': 'low', 'volume': 'volume'})
                for col in ['open', 'close', 'high', 'low', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
        raise ValueError(f'index_empty_{symbol}')

    @retry(times=3, delay=2)
    def get_core_pool(self) -> set:
        import akshare as ak
        pool = set()
        
        # 1. 优先使用 Akshare 常规源
        try:
            for idx in ["000300", "000905", "000852", "399006"]:
                df = ak.index_stock_cons(symbol=idx)
                if df is not None and not df.empty:
                    col = next((c for c in df.columns if '代码' in c), None)
                    if col:
                        pool.update(df[col].astype(str).str.zfill(6).tolist())
            if pool:
                return pool
        except Exception as e:
            log.warning(f"Akshare 获取核心成分股池失败: {e}，尝试切换 Tushare 备用源...")

        # 2. 降级使用 Tushare
        if self.ts_pro:
            try:
                for idx in ["399300.SZ", "000905.SH", "000852.SH", "399006.SZ"]:
                    df = self.ts_pro.index_weight(
                        index_code=idx, 
                        start_date=(datetime.now() - timedelta(days=60)).strftime('%Y%m%d')
                    )
                    if not df.empty:
                        pool.update(df['con_code'].str.slice(0, 6).tolist())
                if pool:
                    return pool
            except Exception as e:
                log.debug(f"Tushare 获取成分股失败: {e}")
                
        # 3. 终极兜底：静态核心50池，防止全市场扫描导致性能爆炸
        log.warning("⚠️ 核心池所有动态接口失效，已降级为静态核心50股票池！")
        return {"600519", "601318", "600036", "601166", "000858", "002594", 
              "000333", "600276", "601012", "601899", "601888", "603288", 
              "002415", "600030", "600887", "600900", "000568", "002304", 
              "002714", "300750", "300760", "600438", "601398", "601288", 
              "601939", "601988", "600000", "601328", "601138", "002475", 
              "000001", "000002", "300015", "300059", "600104", "600690", 
              "601668", "601816", "601857", "601088", "600028", "601066", 
              "600585", "601111", "000157", "000651", "002142", "002271", 
              "300122", "600809"}

    @retry(times=2, delay=2)
    def get_hot_sectors(self) -> dict:
        import akshare as ak
        hot_stocks = {}
        try:
            df = ak.stock_board_industry_name_em()
            if df is not None and not df.empty:
                top_sectors = df.nlargest(5, '涨跌幅')['板块名称'].tolist()
                log.info(f"🌋 (东方财富) 今日领涨主线板块抓取成功: {', '.join(top_sectors)}")
                for sector in top_sectors:
                    try:
                        time.sleep(0.5) 
                        cons = ak.stock_board_industry_cons_em(symbol=sector)
                        if cons is not None and not cons.empty:
                            col = next((c for c in cons.columns if '代码' in c), None)
                            if col:
                                for code in cons[col].astype(str).str.zfill(6).tolist():
                                    hot_stocks[code] = sector
                    except Exception:
                        pass
                if hot_stocks:
                    return hot_stocks
        except Exception as e:
            log.warning(f"东方财富主线板块榜单获取失败(可能被云端拦截): {e}，正在切换同花顺备用源...")

        try:
            df = ak.stock_board_industry_name_ths()
            if df is not None and not df.empty:
                name_col = next((c for c in df.columns if '板块' in c or 'name' in c.lower()), None)
                pct_col = next((c for c in df.columns if '涨跌' in c or 'pct' in c.lower()), None)
                if name_col and pct_col:
                    df[pct_col] = pd.to_numeric(df[pct_col], errors='coerce')
                    top_sectors = df.nlargest(5, pct_col)[name_col].tolist()
                    log.info(f"🌋 (同花顺) 今日领涨主线板块抓取成功: {', '.join(top_sectors)}")
                    for sector in top_sectors:
                        try:
                            time.sleep(0.5)
                            cons = ak.stock_board_industry_cons_ths(symbol=sector)
                            if cons is not None and not cons.empty:
                                col = next((c for c in cons.columns if '代码' in c or 'code' in c.lower()), None)
                                if col:
                                    for code in cons[col].astype(str).str.zfill(6).tolist():
                                        hot_stocks[code] = sector
                        except Exception as e:
                            log.debug(f"同花顺获取板块【{sector}】成分股跳过: {e}")
                    if hot_stocks:
                        return hot_stocks
        except Exception as e:
            log.warning(f"同花顺板块备用源获取也失败: {e}")
            
        return hot_stocks

    @retry(times=2, delay=2)
    def get_northbound_flow(self) -> tuple[float, str]:
        import akshare as ak
        try:
            df = ak.stock_em_hsgt_north_net_flow_in(indicator="沪深港通")
            if df is not None and not df.empty:
                col = 'value' if 'value' in df.columns else df.columns[-1]
                today_flow = float(df.iloc[-1][col]) / 1e8
                if today_flow > 30:
                    return today_flow, f"\n- 🌊 **聪明钱流向**：北水大举流入 **+{today_flow:.0f}亿**"
                elif today_flow < -30:
                    return today_flow, f"\n- ❄️ **聪明钱流向**：北水大幅流出 **{today_flow:.0f}亿**"
                else:
                    return today_flow, f"\n- ⚖️ **聪明钱流向**：北向资金温和 (**{today_flow:+.0f}亿**)"
        except Exception:
            pass
        return 0.0, ""

    def get_etf_spot(self) -> pd.DataFrame:
        """获取ETF实时行情"""
        import akshare as ak
        try:
            df = ak.fund_etf_fund_info_em()
            if df is not None and not df.empty:
                return df
        except Exception as e:
            log.debug(f"ETF实时行情获取失败: {e}")
        return pd.DataFrame()
    
    def get_etf_hist(self, symbol: str, days: int = 250) -> pd.DataFrame:
        """获取ETF历史净值数据"""
        import akshare as ak
        try:
            end = datetime.now().strftime('%Y%m%d')
            start = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            df = ak.fund_etf_hist_em(symbol=symbol, period='daily', 
                                    start_date=start, end_date=end)
            if df is not None and not df.empty:
                col_map = {'日期': C.H_DATE, '开盘': C.H_OPEN, '收盘': C.H_CLOSE, 
                         '最高': C.H_HIGH, '最低': C.H_LOW, '成交量': C.H_VOL}
                df = df.rename(columns=col_map)
                return df
        except Exception as e:
            log.debug(f"ETF历史净值获取失败 {symbol}: {e}")
        return pd.DataFrame()
    
    def get_convertible_bonds(self) -> pd.DataFrame:
        """获取可转债实时数据"""
        import akshare as ak
        try:
            df = ak.bond_zh_cov()
            if df is not None and not df.empty:
                return df
        except Exception as e:
            log.debug(f"可转债数据获取失败: {e}")
        return pd.DataFrame()
    
    def get_southbound_flow(self) -> tuple[float, str]:
        """获取南向资金流向"""
        import akshare as ak
        try:
            df = ak.stock_em_hsgt_south_net_flow_in(indicator="沪深港通")
            if df is not None and not df.empty:
                col = 'value' if 'value' in df.columns else df.columns[-1]
                today_flow = float(df.iloc[-1][col]) / 1e8
                if today_flow > 20:
                    return today_flow, f"南向流入 +{today_flow:.0f}亿"
                elif today_flow < -20:
                    return today_flow, f"南向流出 {today_flow:.0f}亿"
                else:
                    return today_flow, f"南向温和 {today_flow:+.0f}亿"
        except Exception:
            pass
        return 0.0, ""
    
    def get_hk_spot(self) -> pd.DataFrame:
        """获取港股实时行情"""
        import akshare as ak
        try:
            df = ak.stock_hk_spot_em()
            if df is not None and not df.empty:
                return df
        except Exception as e:
            log.debug(f"港股实时行情获取失败: {e}")
        return pd.DataFrame()

