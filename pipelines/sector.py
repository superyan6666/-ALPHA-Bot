"""
行业轮动策略流水线
"""
from typing import List, Dict, Any
from dataclasses import dataclass

from core.config import StrategyType, SectorConfig
from pipelines.base import BasePipeline, PipelineResult


@dataclass
class SectorSignal:
    """行业信号"""
    name: str
    pct_chg: float
    ret_20d: float
    score: int
    level: str
    reasons: str
    signal_type: str


class SectorPipeline(BasePipeline):
    """行业轮动策略流水线"""
    
    def __init__(self, data_lake, config: SectorConfig = None):
        self._data_lake = data_lake
        self._config = config or SectorConfig()
    
    @property
    def name(self) -> str:
        return "行业轮动策略"
    
    def is_enabled(self) -> bool:
        return self._config.enabled
    
    def run(self, shared_context: dict) -> PipelineResult:
        """
        执行行业轮动策略
        
        Args:
            shared_context: 共享上下文
            
        Returns:
            行业轮动结果
        """
        signals = []
        watchlist = []
        market_msg = ""
        
        try:
            sector_rotation = self._data_lake.fetch_sector_rotation()
            
            if not sector_rotation:
                return PipelineResult(
                    signals=signals,
                    watchlist=watchlist,
                    market_msg="暂无行业轮动信号",
                    meta_info={}
                )
            
            scored_sectors = self._score_sectors(sector_rotation, shared_context)
            
            scored_sectors.sort(key=lambda x: x.score, reverse=True)
            
            for sector in scored_sectors[:self._config.max_sectors]:
                if sector.score >= 65:
                    signals.append(sector)
                elif sector.score >= 55:
                    watchlist.append((sector.name, sector.signal_type, sector.score))
            
            market_msg = self._generate_market_msg(scored_sectors)
            
        except Exception as e:
            market_msg = f"行业轮动策略执行失败: {str(e)}"
        
        return PipelineResult(
            signals=signals,
            watchlist=watchlist,
            market_msg=market_msg,
            meta_info={"strategy": StrategyType.SECTOR, "count": len(signals)}
        )
    
    def _score_sectors(self, sectors: List[Dict], context: dict) -> List[SectorSignal]:
        """对行业打分"""
        scored = []
        
        for sector in sectors:
            try:
                score, reasons = self._calculate_score(sector, context)
                level = self._get_level(score)
                
                scored.append(SectorSignal(
                    name=sector['name'],
                    pct_chg=sector['pct'],
                    ret_20d=sector['ret_20d'],
                    score=score,
                    level=level,
                    reasons=reasons,
                    signal_type=sector['signal']
                ))
            except Exception:
                continue
        
        return scored
    
    def _calculate_score(self, sector: Dict, context: dict) -> tuple:
        """计算行业分数"""
        score = 45
        reasons = []
        
        signal_type = sector.get('signal', '')
        ret_20d = sector.get('ret_20d', 0)
        pct_chg = sector.get('pct', 0)
        
        if signal_type == "主升":
            score += 20
            reasons.append("主升动量信号")
            if ret_20d > 10:
                score += 10
                reasons.append("强势动量")
            elif ret_20d > 5:
                score += 5
                reasons.append("稳健动量")
        
        elif signal_type == "反弹":
            score += 15
            reasons.append("超跌反弹信号")
            if pct_chg > 3:
                score += 8
                reasons.append("反弹确认")
            elif pct_chg > 1:
                score += 4
                reasons.append("反弹迹象")
        
        m_regime = context.get('market_regime', 'NEUTRAL')
        if m_regime == 'BULL' and signal_type == "主升":
            score = int(score * 1.2)
            reasons.append("牛市顺势增强")
        elif m_regime == 'BEAR' and signal_type == "反弹":
            score = int(score * 1.3)
            reasons.append("熊市反弹价值")
        
        return min(score, 100), " | ".join(reasons)
    
    def _get_level(self, score: int) -> str:
        """获取等级标签"""
        if score >= 80:
            return "🔥 [S级主线]"
        elif score >= 70:
            return "📈 [A级强势]"
        elif score >= 60:
            return "📊 [B+级机会]"
        else:
            return "📉 [观望]"
    
    def _generate_market_msg(self, scored_sectors: List[SectorSignal]) -> str:
        """生成市场消息"""
        if not scored_sectors:
            return "今日暂无强势行业信号"
        
        main_signals = [s for s in scored_sectors if s.signal_type == "主升"]
        reversal_signals = [s for s in scored_sectors if s.signal_type == "反弹"]
        
        msg_parts = []
        if main_signals:
            top = main_signals[0]
            msg_parts.append(f"主线: {top.name}({top.ret_20d:+.1f}%)")
        if reversal_signals:
            top = reversal_signals[0]
            msg_parts.append(f"反弹: {top.name}({top.pct_chg:+.1f}%)")
        
        return " | ".join(msg_parts) if msg_parts else "行业轮动信号清淡"

