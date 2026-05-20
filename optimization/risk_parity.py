"""
风险平价配置器模块

实现风险平价策略：
- 行业敞口约束
- 风险贡献均等化
- 波动率目标配置
- 组合优化器
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class AssetAllocation:
    """资产配置结果"""
    code: str
    weight: float
    risk_contribution: float
    sector: str
    expected_return: float = 0.0
    volatility: float = 0.0


@dataclass
class PortfolioConstraints:
    """组合约束"""
    max_sector_weight: float = 0.30
    min_sector_weight: float = 0.02
    max_single_weight: float = 0.15
    min_single_weight: float = 0.01
    max_sectors: int = 5
    leverage_limit: float = 1.0
    turnover_limit: float = 0.30


@dataclass
class RiskParityConfig:
    """风险平价配置"""
    target_risk: float = 0.15  # 目标波动率15%
    risk_budget: float = 0.03  # 单个资产最大风险贡献3%
    rebal_freq_days: int = 20
    min_trade_size: float = 0.005  # 最小调仓比例


class RiskParityOptimizer:
    """
    风险平价优化器
    
    核心功能：
    - 计算各资产风险贡献
    - 优化权重使风险贡献均等
    - 应用行业敞口约束
    - 考虑交易成本
    """
    
    def __init__(self, config: Optional[RiskParityConfig] = None,
                 constraints: Optional[PortfolioConstraints] = None):
        self.config = config or RiskParityConfig()
        self.constraints = constraints or PortfolioConstraints()
    
    def optimize(self, signals: List, sector_map: Dict[str, str],
                volatilities: Dict[str, float],
                current_weights: Optional[Dict[str, float]] = None) -> List[AssetAllocation]:
        """
        执行风险平价优化
        
        Args:
            signals: 信号列表（含code, score, sector等属性）
            sector_map: 代码到行业映射
            volatilities: 代码到波动率映射
            current_weights: 当前持仓权重
            
        Returns:
            优化后的资产配置列表
        """
        if not signals:
            return []
        
        if current_weights is None:
            current_weights = {}
        
        allocations = self._initial_allocation(signals, sector_map, volatilities)
        
        allocations = self._apply_sector_constraints(allocations)
        
        allocations = self._apply_risk_parity(allocations, volatilities)
        
        allocations = self._apply_single_constraints(allocations)
        
        allocations = self._apply_turnover_constraint(allocations, current_weights)
        
        allocations = self._normalize_weights(allocations)
        
        return allocations
    
    def _initial_allocation(self, signals: List, sector_map: Dict[str, str],
                           volatilities: Dict[str, float]) -> List[AssetAllocation]:
        """初始配置（基于信号评分）"""
        allocations = []
        total_score = sum(getattr(s, 'score', 0) for s in signals)
        
        for signal in signals:
            code = getattr(signal, 'code', str(signal))
            score = getattr(signal, 'score', 0)
            sector = sector_map.get(code, getattr(signal, 'sector', '其他'))
            vol = volatilities.get(code, 0.2)
            
            if total_score > 0:
                weight = score / total_score
            else:
                weight = 1.0 / len(signals)
            
            allocations.append(AssetAllocation(
                code=code,
                weight=weight,
                risk_contribution=0.0,
                sector=sector,
                volatility=vol
            ))
        
        return allocations
    
    def _apply_sector_constraints(self, allocations: List[AssetAllocation]) -> List[AssetAllocation]:
        """应用行业敞口约束"""
        sector_weights = defaultdict(float)
        for alloc in allocations:
            sector_weights[alloc.sector] += alloc.weight
        
        for alloc in allocations:
            sector_weight = sector_weights[alloc.sector]
            
            if sector_weight > self.constraints.max_sector_weight:
                scale = self.constraints.max_sector_weight / sector_weight
                alloc.weight *= scale
        
        return allocations
    
    def _apply_risk_parity(self, allocations: List[AssetAllocation],
                          volatilities: Dict[str, float]) -> List[AssetAllocation]:
        """应用风险平价调整"""
        total_risk = 0.0
        for alloc in allocations:
            alloc.risk_contribution = alloc.weight * alloc.volatility
            total_risk += alloc.risk_contribution
        
        target_contribution = total_risk / len(allocations) if allocations else 0.0
        
        for alloc in allocations:
            if alloc.volatility > 0:
                alloc.weight = target_contribution / alloc.volatility
        
        return allocations
    
    def _apply_single_constraints(self, allocations: List[AssetAllocation]) -> List[AssetAllocation]:
        """应用单资产约束"""
        for alloc in allocations:
            alloc.weight = max(self.constraints.min_single_weight,
                            min(self.constraints.max_single_weight, alloc.weight))
        return allocations
    
    def _apply_turnover_constraint(self, allocations: List[AssetAllocation],
                                  current_weights: Dict[str, float]) -> List[AssetAllocation]:
        """应用换手率约束"""
        target_weights = {a.code: a.weight for a in allocations}
        
        total_turnover = 0.0
        for code, target in target_weights.items():
            current = current_weights.get(code, 0)
            total_turnover += abs(target - current)
        
        if total_turnover > self.constraints.turnover_limit:
            scale = self.constraints.turnover_limit / total_turnover
            for alloc in allocations:
                current = current_weights.get(alloc.code, 0)
                new_weight = current + (alloc.weight - current) * scale
                alloc.weight = new_weight
        
        return allocations
    
    def _normalize_weights(self, allocations: List[AssetAllocation]) -> List[AssetAllocation]:
        """归一化权重"""
        total = sum(a.weight for a in allocations)
        if total > 0:
            for alloc in allocations:
                alloc.weight /= total
        return allocations
    
    def calculate_risk_contribution(self, allocations: List[AssetAllocation],
                                   cov_matrix: Optional[pd.DataFrame] = None) -> float:
        """计算组合风险贡献"""
        weights = np.array([a.weight for a in allocations])
        vols = np.array([a.volatility for a in allocations])
        
        if cov_matrix is not None:
            portfolio_var = weights @ cov_matrix @ weights.T
            portfolio_vol = np.sqrt(portfolio_var)
        else:
            portfolio_vol = np.sqrt(np.sum((weights * vols)**2))
        
        contributions = []
        for i, alloc in enumerate(allocations):
            if portfolio_vol > 0:
                contribution = (alloc.weight * alloc.volatility**2) / portfolio_vol
            else:
                contribution = 0.0
            alloc.risk_contribution = contribution
            contributions.append(contribution)
        
        return portfolio_vol
    
    def format_allocation_report(self, allocations: List[AssetAllocation]) -> str:
        """格式化配置报告"""
        lines = ["**风险平价配置报告**", ""]
        
        sector_totals = defaultdict(float)
        for alloc in allocations:
            sector_totals[alloc.sector] += alloc.weight
        
        lines.append("**行业敞口：**")
        for sector, weight in sorted(sector_totals.items(), key=lambda x: x[1], reverse=True):
            pct = weight * 100
            bar = "█" * int(pct / 2)
            lines.append(f"{sector}: {pct:5.1f}% {bar}")
        
        lines.append("")
        lines.append("**个股配置：**")
        for alloc in sorted(allocations, key=lambda x: x.weight, reverse=True):
            risk_pct = alloc.risk_contribution * 100
            lines.append(f"{alloc.code}: {alloc.weight*100:5.1f}% (风险贡献:{risk_pct:.1f}%)")
        
        total_vol = self.calculate_risk_contribution(allocations)
        lines.append("")
        lines.append(f"**组合波动率**: {total_vol*100:.1f}%")
        
        return "\n".join(lines)


class SectorExposureManager:
    """
    行业敞口管理器
    
    功能：
    - 监控行业敞口
    - 动态调整约束
    - 行业轮动适配
    """
    
    def __init__(self):
        self.sector_limits = {}
        self.historical_exposures = []
    
    def set_sector_limit(self, sector: str, max_weight: float):
        """设置行业上限"""
        self.sector_limits[sector] = max_weight
    
    def get_sector_exposure(self, allocations: List[AssetAllocation]) -> Dict[str, float]:
        """获取当前行业敞口"""
        exposures = defaultdict(float)
        for alloc in allocations:
            exposures[alloc.sector] += alloc.weight
        return exposures
    
    def is_within_limits(self, allocations: List[AssetAllocation]) -> bool:
        """检查是否在限制内"""
        exposures = self.get_sector_exposure(allocations)
        for sector, exposure in exposures.items():
            limit = self.sector_limits.get(sector, 0.3)
            if exposure > limit:
                return False
        return True
    
    def adjust_for_rotation(self, allocations: List[AssetAllocation],
                           hot_sectors: List[str], cold_sectors: List[str]) -> List[AssetAllocation]:
        """根据行业轮动调整配置"""
        for alloc in allocations:
            if alloc.sector in hot_sectors:
                alloc.weight *= 1.1
            elif alloc.sector in cold_sectors:
                alloc.weight *= 0.9
        
        total = sum(a.weight for a in allocations)
        if total > 0:
            for alloc in allocations:
                alloc.weight /= total
        
        return allocations


class VolatilityTargeting:
    """
    波动率目标管理
    
    功能：
    - 动态杠杆调整
    - 风险预算分配
    - 尾部风险控制
    """
    
    def __init__(self, target_vol: float = 0.15):
        self.target_vol = target_vol
        self.current_leverage = 1.0
    
    def calculate_leverage(self, current_vol: float) -> float:
        """计算所需杠杆"""
        if current_vol == 0:
            return 1.0
        
        leverage = self.target_vol / current_vol
        
        max_leverage = 2.0
        min_leverage = 0.5
        
        return max(min(leverage, max_leverage), min_leverage)
    
    def apply_leverage(self, allocations: List[AssetAllocation],
                      current_vol: float) -> List[AssetAllocation]:
        """应用杠杆调整"""
        leverage = self.calculate_leverage(current_vol)
        self.current_leverage = leverage
        
        for alloc in allocations:
            alloc.weight *= leverage
        
        return allocations
    
    def format_leverage_report(self, current_vol: float) -> str:
        """格式化杠杆报告"""
        leverage = self.calculate_leverage(current_vol)
        status = "🔍" if 0.8 <= leverage <= 1.2 else "📈" if leverage > 1.2 else "📉"
        
        return f"{status} 目标波动率:{self.target_vol*100:.1f}% | 当前波动率:{current_vol*100:.1f}% | 杠杆:{leverage:.2f}"

