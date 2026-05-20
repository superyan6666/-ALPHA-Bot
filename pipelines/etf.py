
"""
ETF轮动策略流水线
"""
from typing import List, Dict, Any
from dataclasses import dataclass

from core.config import StrategyType, ETFConfig
from pipelines.base import BasePipeline, PipelineResult


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


class ETFPipeline(BasePipeline):
    """ETF轮动策略流水线"""
    
    def __init__(self, data_lake, config: ETFConfig = None):
        self._data_lake = data_lake
        self._config = config or ETFConfig()
    
    @property
    def name(self) -> str:
        return "ETF轮动策略"
    
    def is_enabled(self) -> bool:
        return self._config.enabled
    
    def run(self, shared_context: dict) -> PipelineResult:
        """
        执行ETF轮动策略
        
        Args:
            shared_context: 共享上下文
            
        Returns:
            ETF策略结果
        """
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
            
            etf_candidates = self._filter_etf(etf_spot)
            
            scored_etfs = self._score_etfs(etf_candidates, shared_context)
            
            scored_etfs.sort(key=lambda x: x.score, reverse=True)
            
            for etf in scored_etfs[:self._config.max_positions]:
                if etf.score >= 60:
                    signals.append(etf)
                elif etf.score >= 50:
                    watchlist.append((etf.code, etf.name, etf.score))
            
            market_msg = self._generate_market_msg(scored_etfs)
            
        except Exception as e:
            market_msg = f"ETF策略执行失败: {str(e)}"
        
        return PipelineResult(
            signals=signals,
            watchlist=watchlist,
            market_msg=market_msg,
            meta_info={"strategy": StrategyType.ETF, "count": len(signals)}
        )
    
    def _filter_etf(self, etf_spot) -> List[Dict]:
        """筛选ETF候选"""
        candidates = []
        
        for _, row in etf_spot.iterrows():
            try:
                volume = float(row.get('成交量', 0))
                if volume < self._config.min_volume:
                    continue
                    
                price = float(row.get('最新价', 0))
                if price <= 0:
                    continue
                    
                pct_chg = float(row.get('涨跌幅', 0))
                if pct_chg < -10 or pct_chg > 10:
                    continue
                
                candidates.append({
                    'code': str(row.get('代码', '')),
                    'name': str(row.get('名称', '')),
                    'price': price,
                    'pct_chg': pct_chg,
                    'volume': volume
                })
            except Exception:
                continue
        
        return candidates
    
    def _score_etfs(self, candidates: List[Dict], context: dict) -> List[ETFSignal]:
        """对ETF打分"""
        scored = []
        
        for etf in candidates:
            try:
                hist = self._data_lake.fetch_etf_hist(etf['code'])
                
                if hist.empty or len(hist) < 60:
                    continue
                
                features = self._extract_features(hist)
                score, reasons = self._calculate_score(features, context)
                level = self._get_level(score)
                
                scored.append(ETFSignal(
                    code=etf['code'],
                    name=etf['name'],
                    price=etf['price'],
                    pct_chg=etf['pct_chg'],
                    score=score,
                    level=level,
                    reasons=reasons,
                    rank_20d=features.get('rank_20d', 0)
                ))
            except Exception:
                continue
        
        return scored
    
    def _extract_features(self, hist) -> Dict:
        """提取ETF特征"""
        close = hist['close']
        volume = hist['volume']
        
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        
        ret_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0
        ret_60d = (close.iloc[-1] / close.iloc[-61] - 1) * 100 if len(close) >= 61 else 0
        
        vol_ma20 = volume.rolling(20).mean()
        vol_ratio = volume.iloc[-1] / vol_ma20.iloc[-1] if vol_ma20.iloc[-1] > 0 else 1
        
        rank_20d = self._calc_rank(hist, 20)
        
        return {
            'ma20': ma20.iloc[-1],
            'ma60': ma60.iloc[-1],
            'bull_rank': close.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1],
            'ret_20d': ret_20d,
            'ret_60d': ret_60d,
            'vol_ratio': vol_ratio,
            'rank_20d': rank_20d,
            'close': close.iloc[-1]
        }
    
    def _calc_rank(self, hist, window: int) -> float:
        """计算收益率排名"""
        if len(hist) < window * 2:
            return 0.5
        
        current_ret = (hist['close'].iloc[-1] / hist['close'].iloc[-window-1] - 1) * 100
        returns = []
        
        for i in range(window, len(hist) - window):
            ret = (hist['close'].iloc[i] / hist['close'].iloc[i-window] - 1) * 100
            returns.append(ret)
        
        if not returns:
            return 0.5
        
        rank = sum(1 for r in returns if r < current_ret)
        return rank / len(returns)
    
    def _calculate_score(self, features: Dict, context: dict) -> tuple:
        """计算ETF分数"""
        score = 45
        reasons = []
        
        if features['rank_20d'] >= 0.75:
            score += 15
            reasons.append("20日涨幅排名前25%")
        elif features['rank_20d'] >= 0.5:
            score += 8
            reasons.append("20日涨幅排名前50%")
        
        if features['bull_rank']:
            score += 12
            reasons.append("均线多头排列")
        
        if features['vol_ratio'] >= 1.5:
            score += 10
            reasons.append("量能放大")
        
        m_regime = context.get('market_regime', 'NEUTRAL')
        if m_regime == 'BULL' and features['ret_60d'] > 0:
            score = int(score * 1.2)
            reasons.append("牛市顺势增强")
        elif m_regime == 'BEAR' and features['rank_20d'] < 0.3:
            score += 8
            reasons.append("超跌反弹机会")
        
        return min(score, 100), " | ".join(reasons)
    
    def _get_level(self, score: int) -> str:
        """获取等级标签"""
        if score >= 80:
            return "⭐⭐⭐⭐⭐ [S级强势ETF]"
        elif score >= 70:
            return "⭐⭐⭐⭐ [A级优选ETF]"
        elif score >= 60:
            return "⭐⭐⭐ [B+级可关注]"
        else:
            return "⭐⭐ [观望级]"
    
    def _generate_market_msg(self, scored_etfs: List[ETFSignal]) -> str:
        """生成市场消息"""
        if not scored_etfs:
            return "今日暂无符合条件的ETF信号"
        
        top = scored_etfs[0]
        bullish = sum(1 for e in scored_etfs if e.score >= 60)
        
        return (f"今日强势ETF: {top.name}({top.pct_chg:+.2f}%) "
                f"评分{top.score}分 | {bullish}只ETF达到关注标准")

