"""
可转债策略流水线
"""
from typing import List, Dict, Any
from dataclasses import dataclass

from core.config import StrategyType, CBConfig
from pipelines.base import BasePipeline, PipelineResult


@dataclass
class CBSignal:
    """可转债信号"""
    code: str
    name: str
    cb_price: float
    premium_rt: float
    score: int
    level: str
    reasons: str
    double_low: float


class CBPipeline(BasePipeline):
    """可转债策略流水线"""
    
    def __init__(self, data_lake, config: CBConfig = None):
        self._data_lake = data_lake
        self._config = config or CBConfig()
    
    @property
    def name(self) -> str:
        return "可转债策略"
    
    def is_enabled(self) -> bool:
        return self._config.enabled
    
    def run(self, shared_context: dict) -> PipelineResult:
        """
        执行可转债策略
        
        Args:
            shared_context: 共享上下文
            
        Returns:
            可转债策略结果
        """
        signals = []
        watchlist = []
        market_msg = ""
        
        try:
            cb_spot = self._data_lake.fetch_convertible_bonds()
            
            if cb_spot.empty:
                return PipelineResult(
                    signals=signals,
                    watchlist=watchlist,
                    market_msg="无可转债数据",
                    meta_info={}
                )
            
            cb_candidates = self._filter_cb(cb_spot)
            
            scored_cbs = self._score_cbs(cb_candidates, shared_context)
            
            scored_cbs.sort(key=lambda x: x.score, reverse=True)
            
            for cb in scored_cbs[:self._config.max_positions]:
                if cb.score >= 65:
                    signals.append(cb)
                elif cb.score >= 55:
                    watchlist.append((cb.code, cb.name, cb.score))
            
            market_msg = self._generate_market_msg(scored_cbs)
            
        except Exception as e:
            market_msg = f"可转债策略执行失败: {str(e)}"
        
        return PipelineResult(
            signals=signals,
            watchlist=watchlist,
            market_msg=market_msg,
            meta_info={"strategy": StrategyType.CONVERTIBLE_BOND, "count": len(signals)}
        )
    
    def _filter_cb(self, cb_spot) -> List[Dict]:
        """筛选可转债候选"""
        candidates = []
        
        for _, row in cb_spot.iterrows():
            try:
                cb_price = float(row.get('最新价', 0))
                if cb_price <= 0 or cb_price > 150:
                    continue
                
                premium_rt = float(row.get('转股溢价率', 100))
                if premium_rt > self._config.max_premium:
                    continue
                
                scale = float(row.get('剩余规模', 0))
                if scale < self._config.min_scale:
                    continue
                
                rating = str(row.get('债券评级', 'AAA'))
                if self._is_low_rating(rating):
                    continue
                
                candidates.append({
                    'code': str(row.get('代码', '')),
                    'name': str(row.get('名称', '')),
                    'cb_price': cb_price,
                    'premium_rt': premium_rt,
                    'bond_rt': float(row.get('纯债溢价率', 0)),
                    'scale': scale,
                    'rating': rating,
                    'stock_pct': float(row.get('正股涨跌幅', 0))
                })
            except Exception:
                continue
        
        return candidates
    
    def _is_low_rating(self, rating: str) -> bool:
        """检查评级是否过低"""
        low_ratings = {'A+', 'A', 'A-', 'BBB', 'BB', 'B', 'C', 'D'}
        return rating in low_ratings
    
    def _score_cbs(self, candidates: List[Dict], context: dict) -> List[CBSignal]:
        """对可转债打分"""
        scored = []
        
        for cb in candidates:
            try:
                score, reasons = self._calculate_score(cb, context)
                level = self._get_level(score)
                double_low = cb['cb_price'] + cb['premium_rt'] * 0.5
                
                scored.append(CBSignal(
                    code=cb['code'],
                    name=cb['name'],
                    cb_price=cb['cb_price'],
                    premium_rt=cb['premium_rt'],
                    score=score,
                    level=level,
                    reasons=reasons,
                    double_low=double_low
                ))
            except Exception:
                continue
        
        return scored
    
    def _calculate_score(self, cb: Dict, context: dict) -> tuple:
        """计算可转债分数"""
        score = 45
        reasons = []
        
        double_low = cb['cb_price'] + cb['premium_rt'] * 0.5
        if double_low < 100:
            score += 15
            reasons.append(f"双低优选({double_low:.1f})")
        elif double_low < 120:
            score += 10
            reasons.append(f"双低尚可({double_low:.1f})")
        elif double_low < 140:
            score += 5
            reasons.append(f"双低一般({double_low:.1f})")
        
        bond_rt = cb.get('bond_rt', 50)
        if bond_rt < 15:
            score += 10
            reasons.append("债底保护强")
        elif bond_rt < 25:
            score += 6
            reasons.append("债底保护可")
        
        scale = cb.get('scale', 0)
        if 2e8 <= scale <= 10e8:
            score += 8
            reasons.append("规模适中")
        elif scale > 10e8:
            score += 4
            reasons.append("规模较大")
        
        if cb['cb_price'] < 105:
            score += 8
            reasons.append("接近债底")
        
        stock_pct = cb.get('stock_pct', 0)
        if stock_pct > 3 and cb['cb_price'] < 130:
            score += 10
            reasons.append("正股强势映射")
        
        m_regime = context.get('market_regime', 'NEUTRAL')
        if m_regime == 'BEAR':
            score = int(score * 1.2)
            reasons.append("熊市防御增强")
        elif m_regime == 'BULL':
            score = int(score * 0.9)
            reasons.append("牛市股性偏弱")
        
        return min(score, 100), " | ".join(reasons)
    
    def _get_level(self, score: int) -> str:
        """获取等级标签"""
        if score >= 80:
            return "⭐⭐⭐⭐⭐ [S级双低优选]"
        elif score >= 70:
            return "⭐⭐⭐⭐ [A级防守优选]"
        elif score >= 60:
            return "⭐⭐⭐ [B+级可配置]"
        else:
            return "⭐⭐ [观望级]"
    
    def _generate_market_msg(self, scored_cbs: List[CBSignal]) -> str:
        """生成市场消息"""
        if not scored_cbs:
            return "今日暂无符合条件的可转债信号"
        
        top = scored_cbs[0]
        qualified = sum(1 for c in scored_cbs if c.score >= 60)
        
        return (f"今日双低转债: {top.name}(价格{top.cb_price:.2f}元 "
                f"溢价{top.premium_rt:.1f}%) 评分{top.score}分 | {qualified}只达到关注标准")

