"""
ETF轮动策略流水线
"""
import time
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from core.config import StrategyType, ETFConfig

log = logging.getLogger(__name__)


@dataclass
class ETFSignal:
    """ETF信号"""
    code: str
    name: str
    price: float
    pct_chg: float
    score: int
    level: str
    reasons: str
    rank_20d: float
    atr_pct: float
    strategy_type: str = "etf"


class ETFTechnicals:
    """ETF技术特征提取"""
    
    def __init__(self, df: pd.DataFrame, benchmark_ret: float = 0.0):
        self.df = df.copy()
        close = self.df['close']
        high, low, vol = self.df['high'], self.df['low'], self.df['volume']
        
        for span in (5, 10, 20, 60, 120):
            self.df[f'MA{span}'] = close.rolling(span).mean()
        self.df['MA5_V'] = vol.rolling(5).mean()
        self.df['MA20_V'] = vol.rolling(20).mean()
        
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        self.df['DIF'] = ema12 - ema26
        self.df['DEA'] = self.df['DIF'].ewm(span=9, adjust=False).mean()
        self.df['MACD'] = (self.df['DIF'] - self.df['DEA']) * 2
        
        self.df['ATR'], self.df['ADX'] = self._calc_atr_adx(self.df)
        
        delta = close.diff()
        gain = delta.clip(lower=0.0).rolling(14).mean()
        loss = (-delta.clip(upper=0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        self.df['RSI14'] = 100 - (100 / (1 + rs))
        
        self.df['PCT_CHG'] = close.pct_change() * 100
        self.df['OBV'] = np.where(close > close.shift(), vol, np.where(close < close.shift(), -vol, 0)).cumsum()
        
        for span in (5, 10, 20):
            self.df[f'BB_MID{span}'] = close.rolling(span).mean()
            self.df[f'BB_STD{span}'] = close.rolling(span).std()
        
        self.today = self.df.iloc[-1]
        self.yest = self.df.iloc[-2]
        self.benchmark_ret = benchmark_ret
    
    def _calc_atr_adx(self, hist, period=14):
        high, low, close = hist['high'], hist['low'], hist['close']
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        up, dn = high.diff(), -low.diff()
        plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
        minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
        plus_di = 100 * pd.Series(plus_dm, index=hist.index).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=hist.index).rolling(period).mean() / atr
        denom = (plus_di + minus_di).replace(0, np.nan)
        dx = ((plus_di - minus_di).abs() / denom * 100)
        return atr, dx.rolling(period).mean()
    
    def get_features(self) -> Optional[dict]:
        df, today = self.df, self.today
        if pd.isna(today['ATR']) or today['ATR'] <= 1e-5:
            return None
        
        min_1y, max_1y = df['low'].min(), df['high'].max()
        rng = max_1y - min_1y
        if rng <= 0:
            return None
        
        price_pct = (today['close'] - min_1y) / rng
        
        rsi = float(today.get('RSI14', 50))
        if pd.isna(rsi) or rsi > 90:
            return None
        
        surge_5d = (today['close'] / df['close'].iloc[-6] - 1) * 100 if len(df) >= 6 else 0.0
        surge_20d = (today['close'] / df['close'].iloc[-21] - 1) * 100 if len(df) >= 21 else 0.0
        
        vcp_amp, is_true_vcp = self._calc_vcp_quality(df)
        
        red_days = 0
        for i in range(1, 4):
            if df['close'].iloc[-i] > df['open'].iloc[-i]:
                red_days += 1
            else:
                break
        
        vol_ratio = today['volume'] / today['MA20_V'] if pd.notna(today['MA20_V']) and today['MA20_V'] > 0 else 1.0
        
        rank_20d = self._calc_rank_pct()
        rank_60d = self._calc_rank_pct(window=60)
        
        bb_width = self._calc_bollinger_width(20)
        bb_position = (today['close'] - today['BB_MID20']) / (2 * today['BB_STD20']) if pd.notna(today['BB_STD20']) and today['BB_STD20'] > 0 else 0
        
        macd_hist = float(today['DIF']) - float(today['DEA'])
        
        return {
            'price_pct': price_pct,
            'adx': float(today['ADX']),
            'bull_rank': bool(today['MA20'] > today['MA60']),
            'bull_rank_long': bool(today['MA20'] > today['MA120']),
            'rsi': rsi,
            'surge_5d': surge_5d,
            'surge_20d': surge_20d,
            'vcp_amp': vcp_amp,
            'is_true_vcp': is_true_vcp,
            'red_days': red_days,
            'vol_ratio': float(vol_ratio),
            'dist_ma20': (today['close'] / today['MA20'] - 1) * 100,
            'dist_ma60': (today['close'] / today['MA60'] - 1) * 100,
            'macd_dea': float(today['DEA']),
            'macd_hist': macd_hist,
            'ma10_val': float(today['MA10']),
            'ma20_val': float(today['MA20']),
            'ma60_val': float(today['MA60']),
            'atr_val': float(today['ATR']),
            'close_val': float(today['close']),
            'atr_pct': (float(today['ATR']) / float(today['close'])) * 100,
            'rs_rating': ((today['close'] / df['close'].iloc[-60] - 1) * 100 - self.benchmark_ret) if len(df) >= 60 else 0.0,
            'rank_20d': rank_20d,
            'rank_60d': rank_60d,
            'has_obv_break': bool(df['OBV'].iloc[-1] > df['OBV'].iloc[-21:-1].max()),
            'pct_chg': float(today['PCT_CHG']),
            'bb_width': bb_width,
            'bb_position': bb_position,
            'volume_5d_avg': float(tol['MA5_V']) if 'MA5_V' in dir() else float(vol.rolling(5).mean().iloc[-1]),
        }
    
    def _calc_vcp_quality(self, df) -> tuple:
        if len(df) < 31:
            return 0.5, False
        segments = []
        for i in [(-31, -21), (-21, -11), (-11, -1)]:
            seg = df.iloc[i[0]:i[1]]
            low = seg['low'].min()
            if low > 0:
                amp = (seg['high'].max() - low) / low
                segments.append(amp)
        if len(segments) < 3:
            return segments[-1] if segments else 0.5, False
        is_vcp = segments[0] > segments[1] > segments[2]
        return segments[-1], is_vcp
    
    def _calc_rank_pct(self, window: int = 20) -> float:
        if len(self.df) < max(250, window * 2):
            return 0.5
        ret = (self.df['close'].iloc[-1] / self.df['close'].iloc[-window-1] - 1) * 100 if len(self.df) >= window + 1 else 0.0
        returns = []
        for i in range(window, len(self.df) - window):
            r = (self.df['close'].iloc[i] / self.df['close'].iloc[i-window] - 1) * 100
            returns.append(r)
        if not returns:
            return 0.5
        rank = sum(1 for r in returns if r < ret)
        return rank / len(returns)
    
    def _calc_bollinger_width(self, window: int = 20) -> float:
        if len(self.df) < window:
            return 0.0
        mid = self.df[f'BB_MID{window}']
        std = self.df[f'BB_STD{window}']
        if pd.isna(std.iloc[-1]) or std.iloc[-1] == 0:
            return 0.0
        return (4 * std.iloc[-1] / mid.iloc[-1]) * 100


class ETFPipeline:
    """ETF轮动策略流水线"""
    
    def __init__(self, data_lake, config: ETFConfig = None):
        self._data_lake = data_lake
        self._config = config or ETFConfig()
    
    @property
    def name(self) -> str:
        return "ETF轮动策略"
    
    def is_enabled(self) -> bool:
        return self._config.enabled
    
    def run(self, shared_context: dict = None) -> 'PipelineResult':
        from pipelines.base import PipelineResult
        
        if shared_context is None:
            shared_context = {}
        
        signals = []
        watchlist = []
        market_msg = ""
        
        try:
            etf_spot = self._data_lake.fetch_etf_spot()
            
            if etf_spot.empty:
                return PipelineResult(
                    signals=signals,
                    watchlist=watchlist,
                    market_msg="无ETF数据",
                    meta_info={}
                )
            
            market_regime = shared_context.get('market_regime', 'NEUTRAL')
            win_stats = shared_context.get('win_stats', {})
            
            etf_candidates = self._filter_etf(etf_spot)
            
            if not etf_candidates:
                return PipelineResult(
                    signals=signals,
                    watchlist=watchlist,
                    market_msg="无符合条件的ETF",
                    meta_info={}
                )
            
            scored_etfs = self._score_etfs(etf_candidates, market_regime, win_stats)
            
            scored_etfs.sort(key=lambda x: x[1], reverse=True)
            
            for code, name, price, pct, score, reasons, rank, atr_pct in scored_etfs:
                if score >= 70:
                    level = self._get_level(score)
                    signals.append(ETFSignal(
                        code=code, name=name, price=price, pct_chg=pct,
                        score=score, level=level, reasons=reasons,
                        rank_20d=rank, atr_pct=atr_pct
                    ))
                elif score >= 60:
                    watchlist.append((code, name, score))
            
            market_msg = self._generate_market_msg(scored_etfs)
            
        except Exception as e:
            log.error(f"ETF策略执行失败: {e}")
            market_msg = f"执行失败: {str(e)}"
        
        return PipelineResult(
            signals=signals,
            watchlist=watchlist,
            market_msg=market_msg,
            meta_info={"strategy": StrategyType.ETF, "count": len(signals)}
        )
    
    def _filter_etf(self, etf_spot) -> List[Dict]:
        candidates = []
        
        for _, row in etf_spot.iterrows():
            try:
                code = str(row.get('代码', ''))
                if not code:
                    continue
                
                volume = float(row.get('成交量', 0))
                if volume < self._config.min_volume:
                    continue
                    
                price = float(row.get('最新价', 0))
                if price <= 0:
                    continue
                    
                pct_chg = float(row.get('涨跌幅', 0))
                if pct_chg < -10 or pct_chg > 10:
                    continue
                
                name = str(row.get('名称', code))
                
                candidates.append({
                    'code': code,
                    'name': name,
                    'price': price,
                    'pct_chg': pct_chg,
                    'volume': volume
                })
            except Exception:
                continue
        
        return candidates
    
    def _score_etfs(self, candidates: List[Dict], market_regime: str, 
                   win_stats: dict) -> List[tuple]:
        scored = []
        
        def process_etf(etf):
            try:
                hist = self._data_lake.fetch_etf_hist(etf['code'])
                
                if hist is None or hist.empty or len(hist) < 60:
                    return None
                
                hist.columns = [c.lower() for c in hist.columns]
                
                benchmark_ret = 0.0
                tech = ETFTechnicals(hist, benchmark_ret)
                features = tech.get_features()
                
                if features is None:
                    return None
                
                score, reasons = self._apply_scoring(features, market_regime)
                
                if score < 55:
                    return None
                
                bucket = self._get_bucket(score)
                if bucket in win_stats and win_stats[bucket]['total'] >= 5:
                    wr = win_stats[bucket]['win'] / win_stats[bucket]['total']
                    score = int(score * (0.8 + 0.4 * wr))
                
                score = max(0, min(score, 100))
                
                return (
                    etf['code'], etf['name'], etf['price'], etf['pct_chg'],
                    score, reasons, features['rank_20d'], features['atr_pct']
                )
            except Exception as e:
                log.debug(f"处理ETF {etf['code']} 失败: {e}")
                return None
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_etf, c) for c in candidates]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    scored.append(result)
        
        scored.sort(key=lambda x: x[4], reverse=True)
        return scored
    
    def _apply_scoring(self, features: dict, market_regime: str) -> tuple:
        score = 45
        reasons = []
        
        f_mom = 1.3 if market_regime == 'BULL' else 0.8 if market_regime == 'BEAR' else 1.0
        
        if features['rank_20d'] >= 0.75:
            score += int(15 * f_mom)
            reasons.append("20日强势Top25%")
        elif features['rank_20d'] >= 0.5:
            score += int(8 * f_mom)
            reasons.append("20日排名靠前")
        elif features['rank_20d'] < 0.25:
            score += int(8 * f_mom * 0.5)
            reasons.append("超跌反弹机会")
        
        if features['rank_60d'] >= 0.6:
            score += int(10 * f_mom)
            reasons.append("中期强势")
        
        if features['bull_rank']:
            score += int(12 * f_mom)
            reasons.append("均线多头")
        
        if features['bull_rank_long']:
            score += int(8 * f_mom)
            reasons.append("长期上升趋势")
        
        if features['vol_ratio'] >= 1.5:
            score += int(10 * f_mom)
            reasons.append("量能放大")
        
        if features['has_obv_break']:
            score += int(12 * f_mom)
            reasons.append("资金持续流入")
        
        if features['rsi'] < 40:
            score += int(8 * f_mom * 0.7)
            reasons.append("RSI超卖")
        elif 40 <= features['rsi'] <= 65:
            score += int(6 * f_mom)
            reasons.append("RSI健康")
        
        if features['is_true_vcp']:
            score += int(8 * f_mom)
            reasons.append("VCP收敛")
        
        if features['macd_hist'] > 0 and features['macd_hist'] > features['macd_hist'] * 0.5:
            score += int(6 * f_mom)
            reasons.append("MACD红柱")
        
        if features['surge_5d'] > 15:
            score -= 10
            reasons.append("⚠️短期过热")
        
        if features['rsi'] > 75:
            score -= 8
            reasons.append("⚠️RSI过热")
        
        if features['price_pct'] < 0.2:
            score += 5
            reasons.append("低位配置价值")
        
        if features['bb_width'] < 5:
            score += 8
            reasons.append("布林收窄变盘在即")
        
        score = min(score, 100)
        
        return score, " | ".join(reasons)
    
    def _get_level(self, score: int) -> str:
        if score >= 80:
            return "⭐⭐⭐⭐⭐ 📊 **[S级·强势ETF]**"
        elif score >= 70:
            return "⭐⭐⭐⭐ 📈 **[A级·优选ETF]**"
        elif score >= 60:
            return "⭐⭐⭐ 📊 **[B+级·可关注]**"
        return "⭐⭐ 📉 **[观望级]**"
    
    def _get_bucket(self, score: int) -> str:
        if score >= 85:
            return '85-100'
        elif score >= 80:
            return '80-85'
        elif score >= 75:
            return '75-80'
        elif score >= 70:
            return '70-75'
        return '<70'
    
    def _generate_market_msg(self, scored_etfs: List) -> str:
        if not scored_etfs:
            return "今日暂无符合条件的ETF信号"
        
        top = scored_etfs[0]
        qualified = sum(1 for e in scored_etfs if e[4] >= 60)
        bullish = sum(1 for e in scored_etfs if e[4] >= 70)
        
        return (f"ETF轮动: {top[1]}({top[3]:+.2f}%) "
                f"评分{top[4]}分 | {qualified}只达标({bullish}只强势)")


from pipelines.base import PipelineResult
