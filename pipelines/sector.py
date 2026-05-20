"""
行业轮动策略流水线
"""
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

from core.config import StrategyType, SectorConfig

log = logging.getLogger(__name__)


@dataclass
class SectorSignal:
    """行业信号"""
    name: str
    code: str
    pct_chg: float
    ret_20d: float
    ret_5d: float
    volume_ratio: float
    score: int
    level: str
    reasons: str
    signal_type: str
    strategy_type: str = "sector"


class SectorPipeline:
    """行业轮动策略流水线"""
    
    def __init__(self, data_lake, config: SectorConfig = None):
        self._data_lake = data_lake
        self._config = config or SectorConfig()
    
    @property
    def name(self) -> str:
        return "行业轮动策略"
    
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
            import akshare as ak
            
            df = ak.stock_board_industry_name_em()
            if df is None or df.empty:
                return PipelineResult(
                    signals=signals,
                    watchlist=watchlist,
                    market_msg="无行业数据",
                    meta_info={}
                )
            
            market_regime = shared_context.get('market_regime', 'NEUTRAL')
            hot_sectors = shared_context.get('hot_sectors', {})
            
            sector_list = self._analyze_sectors(df, market_regime, hot_sectors)
            
            scored_sectors = self._score_sectors(sector_list, market_regime)
            
            scored_sectors.sort(key=lambda x: x[4], reverse=True)
            
            for name, code, pct, ret_20d, score, reasons, signal_type in scored_sectors:
                if score >= 65:
                    level = self._get_level(score)
                    signals.append(SectorSignal(
                        name=name, code=code, pct_chg=pct,
                        ret_20d=ret_20d, ret_5d=0, volume_ratio=0,
                        score=score, level=level, reasons=reasons,
                        signal_type=signal_type
                    ))
                elif score >= 55:
                    watchlist.append((name, signal_type, score))
            
            market_msg = self._generate_market_msg(scored_sectors)
            
        except Exception as e:
            log.error(f"行业轮动策略执行失败: {e}")
            market_msg = f"执行失败: {str(e)}"
        
        return PipelineResult(
            signals=signals,
            watchlist=watchlist,
            market_msg=market_msg,
            meta_info={"strategy": StrategyType.SECTOR, "count": len(signals)}
        )
    
    def _analyze_sectors(self, df, market_regime: str, hot_sectors: dict) -> List[Dict]:
        """分析各行业"""
        sector_list = []
        
        for _, row in df.iterrows():
            try:
                name = str(row.get('板块名称', ''))
                code = str(row.get('板块代码', ''))
                pct = float(row.get('涨跌幅', 0))
                
                if not code:
                    continue
                
                try:
                    hist = self._data_lake.fetch_index(code)
                    if hist is None or hist.empty or len(hist) < 25:
                        continue
                    
                    hist.columns = [c.lower() for c in hist.columns]
                    
                    ret_20d = (hist['close'].iloc[-1] / hist['close'].iloc[-21] - 1) * 100 if len(hist) >= 21 else 0
                    ret_5d = (hist['close'].iloc[-1] / hist['close'].iloc[-6] - 1) * 100 if len(hist) >= 6 else 0
                    
                    vol_now = hist['volume'].iloc[-5:].mean() if len(hist) >= 5 else hist['volume'].iloc[-1]
                    vol_before = hist['volume'].iloc[-20:-5].mean() if len(hist) >= 20 else vol_now
                    vol_ratio = vol_now / vol_before if vol_before > 0 else 1.0
                    
                    rsi = self._calc_rsi(hist['close'])
                    
                    adx = self._calc_adx(hist)
                    
                    sector_list.append({
                        'name': name,
                        'code': code,
                        'pct': pct,
                        'ret_20d': ret_20d,
                        'ret_5d': ret_5d,
                        'vol_ratio': vol_ratio,
                        'rsi': rsi,
                        'adx': adx,
                        'volume': float(hist['volume'].iloc[-1]) if len(hist) >= 1 else 0,
                        'close': float(hist['close'].iloc[-1]) if len(hist) >= 1 else 0,
                        'is_hot': name in hot_sectors.values()
                    })
                except Exception as e:
                    log.debug(f"分析行业 {name} 失败: {e}")
                    continue
                    
            except Exception:
                continue
        
        return sector_list
    
    def _calc_rsi(self, close, period=14) -> float:
        """计算RSI"""
        import numpy as np
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0
    
    def _calc_adx(self, hist) -> float:
        """计算ADX"""
        import numpy as np
        high, low, close = hist['high'], hist['low'], hist['close']
        period = 14
        
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        up, dn = high.diff(), -low.diff()
        plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
        minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
        plus_di = 100 * pd.Series(plus_dm, index=hist.index).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=hist.index).rolling(period).mean() / atr
        denom = (plus_di + minus_di).replace(0, np.nan)
        dx = ((plus_di - minus_di).abs() / denom * 100)
        adx = dx.rolling(period).mean()
        return float(adx.iloc[-1]) if not adx.empty else 25.0
    
    def _score_sectors(self, sectors: List[Dict], market_regime: str) -> List[tuple]:
        """对行业打分"""
        scored = []
        
        for sector in sectors:
            try:
                score, reasons, signal_type = self._apply_scoring(sector, market_regime)
                
                if score < 50:
                    continue
                
                scored.append((
                    sector['name'], sector['code'], sector['pct'],
                    sector['ret_20d'], score, reasons, signal_type
                ))
            except Exception:
                continue
        
        scored.sort(key=lambda x: x[4], reverse=True)
        return scored
    
    def _apply_scoring(self, sector: Dict, market_regime: str) -> tuple:
        score = 45
        reasons = []
        signal_type = ""
        
        pct = sector['pct']
        ret_20d = sector['ret_20d']
        ret_5d = sector['ret_5d']
        vol_ratio = sector['vol_ratio']
        rsi = sector.get('rsi', 50)
        adx = sector.get('adx', 25)
        is_hot = sector.get('is_hot', False)
        
        f_mom = 1.3 if market_regime == 'BULL' else 0.8 if market_regime == 'BEAR' else 1.0
        
        if ret_20d > self._config.momentum_threshold and vol_ratio > 1.2:
            score += int(25 * f_mom)
            reasons.append("主升动量")
            signal_type = "主升"
        elif ret_20d > self._config.momentum_threshold:
            score += int(15 * f_mom)
            reasons.append("中期强势")
            signal_type = "强势"
        
        if ret_5d < self._config.reversal_threshold and pct > 2:
            score += int(20 * f_mom)
            reasons.append("超跌反弹")
            signal_type = "反弹"
        elif ret_5d < -5 and pct > 0:
            score += int(12 * f_mom)
            reasons.append("缩量反弹")
            signal_type = "弱反弹"
        
        if vol_ratio > 1.5:
            score += int(10 * f_mom)
            reasons.append("资金涌入")
        elif vol_ratio > 1.2:
            score += int(6 * f_mom)
            reasons.append("温和放量")
        
        if is_hot:
            score += int(8 * f_mom)
            reasons.append("主线热点")
        
        if pct > 3:
            score += int(8 * f_mom)
            reasons.append("今日强势")
        elif pct > 1:
            score += int(4 * f_mom)
            reasons.append("小幅走强")
        
        if rsi < 40:
            score += int(8 * f_mom * 0.7)
            reasons.append("技术超卖")
        elif 40 <= rsi <= 60:
            score += int(6 * f_mom)
            reasons.append("技术健康")
        
        if adx > 25:
            score += int(8 * f_mom)
            reasons.append("趋势确认")
        
        if pct < -3 and ret_5d < -8:
            score -= 10
            reasons.append("⚠️破位下行")
        
        if market_regime == 'BEAR' and signal_type == "反弹":
            score = int(score * 1.2)
            reasons.append("熊市反弹增强")
        elif market_regime == 'BULL' and signal_type == "主升":
            score = int(score * 1.1)
            reasons.append("牛市顺势增强")
        
        score = min(score, 100)
        
        return score, " | ".join(reasons), signal_type
    
    def _get_level(self, score: int) -> str:
        if score >= 80:
            return "🔥⭐⭐⭐⭐⭐ **[S级主线]**"
        elif score >= 70:
            return "📈⭐⭐⭐⭐ **[A级强势]**"
        elif score >= 60:
            return "📊⭐⭐⭐ **[B+级机会]**"
        return "📉⭐⭐ **[观望]**"
    
    def _generate_market_msg(self, scored_sectors: List) -> str:
        if not scored_sectors:
            return "今日暂无强势行业信号"
        
        main_signals = [s for s in scored_sectors if s[6] == "主升"]
        reversal_signals = [s for s in scored_sectors if s[6] == "反弹"]
        strong_signals = [s for s in scored_sectors if s[4] >= 70]
        
        msg_parts = []
        if main_signals:
            top = main_signals[0]
            msg_parts.append(f"主线:{top[0]}({top[3]:+.1f}%)")
        if reversal_signals:
            top = reversal_signals[0]
            msg_parts.append(f"反弹:{top[0]}({top[2]:+.1f}%)")
        if strong_signals:
            msg_parts.append(f"强势{len(strong_signals)}个")
        
        return " | ".join(msg_parts) if msg_parts else "行业轮动清淡"


import pandas as pd
from pipelines.base import PipelineResult
