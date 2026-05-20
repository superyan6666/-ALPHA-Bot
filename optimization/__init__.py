"""
策略优化模块

提供量化策略优化工具：
- 因子有效性分析
- 动态止损机制
- 信号过滤与去重
- 市场状态识别
- 风险平价配置
- 增量更新机制
"""
from .factor_effectiveness import (
    FactorEffectivenessAnalyzer,
    FactorCorrelationAnalyzer,
    FactorMetrics,
    FactorConfig,
)
from .stop_loss import (
    DynamicStopLossCalculator,
    StopLossOptimizer,
    StopType,
    StopLevel,
    Position,
)
from .signal_filter import (
    SignalFilter,
    SectorAllocator,
    PositionSizer,
    Signal,
    FilterResult,
)
from .market_state import (
    MarketStateDetector,
    MarketState,
    MarketRegime,
    RegimeProbabilities,
)
from .risk_parity import (
    RiskParityOptimizer,
    SectorExposureManager,
    VolatilityTargeting,
    AssetAllocation,
    PortfolioConstraints,
    RiskParityConfig,
)
from .incremental import (
    EventManager,
    IncrementalUpdater,
    DataChangeDetector,
    PipelineEventAdapter,
    MarketEvent,
    Subscription,
)

__all__ = [
    # 因子有效性
    'FactorEffectivenessAnalyzer',
    'FactorCorrelationAnalyzer',
    'FactorMetrics',
    'FactorConfig',
    # 止损
    'DynamicStopLossCalculator',
    'StopLossOptimizer',
    'StopType',
    'StopLevel',
    'Position',
    # 信号过滤
    'SignalFilter',
    'SectorAllocator',
    'PositionSizer',
    'Signal',
    'FilterResult',
    # 市场状态
    'MarketStateDetector',
    'MarketState',
    'MarketRegime',
    'RegimeProbabilities',
    # 风险平价
    'RiskParityOptimizer',
    'SectorExposureManager',
    'VolatilityTargeting',
    'AssetAllocation',
    'PortfolioConstraints',
    'RiskParityConfig',
    # 增量更新
    'EventManager',
    'IncrementalUpdater',
    'DataChangeDetector',
    'PipelineEventAdapter',
    'MarketEvent',
    'Subscription',
]

