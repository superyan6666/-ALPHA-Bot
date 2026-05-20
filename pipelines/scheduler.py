"""
流水线调度器
"""
from typing import List, Dict, Any
import logging

from core.config import strategy_config
from pipelines.base import BasePipeline, PipelineResult
from pipelines.etf import ETFPipeline
from pipelines.cb import CBPipeline
from pipelines.sector import SectorPipeline

log = logging.getLogger(__name__)


class PipelineScheduler:
    """流水线调度器"""
    
    def __init__(self, data_lake):
        self._data_lake = data_lake
        self._pipelines: Dict[str, BasePipeline] = {}
        self._init_pipelines()
    
    def _init_pipelines(self):
        """初始化所有流水线"""
        if strategy_config.is_enabled('etf'):
            self._pipelines['etf'] = ETFPipeline(
                self._data_lake, 
                strategy_config.etf
            )
            log.info("✅ ETF轮动流水线已注册")
        
        if strategy_config.is_enabled('cb'):
            self._pipelines['cb'] = CBPipeline(
                self._data_lake,
                strategy_config.cb
            )
            log.info("✅ 可转债流水线已注册")
        
        if strategy_config.is_enabled('sector'):
            self._pipelines['sector'] = SectorPipeline(
                self._data_lake,
                strategy_config.sector
            )
            log.info("✅ 行业轮动流水线已注册")
        
        log.info(f"📋 流水线调度器初始化完成，共注册 {len(self._pipelines)} 个流水线")
    
    def run_all(self, shared_context: dict = None) -> Dict[str, PipelineResult]:
        """
        运行所有启用的流水线
        
        Args:
            shared_context: 共享上下文
            
        Returns:
            各流水线执行结果字典
        """
        if shared_context is None:
            shared_context = {}
        
        results = {}
        
        for name, pipeline in self._pipelines.items():
            try:
                log.info(f"🚀 启动流水线: {pipeline.name}")
                result = pipeline.run(shared_context)
                results[name] = result
                log.info(f"✅ {pipeline.name} 完成，产出 {len(result.signals)} 个信号")
            except Exception as e:
                log.error(f"❌ {pipeline.name} 执行失败: {e}")
                results[name] = PipelineResult(
                    signals=[],
                    watchlist=[],
                    market_msg=f"执行失败: {str(e)}",
                    meta_info={"error": str(e)}
                )
        
        return results
    
    def run_strategy(self, strategy_type: str, shared_context: dict = None) -> PipelineResult:
        """
        运行指定策略流水线
        
        Args:
            strategy_type: 策略类型
            shared_context: 共享上下文
            
        Returns:
            执行结果
        """
        if strategy_type not in self._pipelines:
            return PipelineResult(
                signals=[],
                watchlist=[],
                market_msg=f"未找到策略: {strategy_type}",
                meta_info={}
            )
        
        if shared_context is None:
            shared_context = {}
        
        return self._pipelines[strategy_type].run(shared_context)
    
    def list_pipelines(self) -> List[Dict[str, str]]:
        """列出所有流水线"""
        return [
            {"name": p.name, "type": name, "enabled": True}
            for name, p in self._pipelines.items()
        ]
    
    def get_all_signals(self, results: Dict[str, PipelineResult]) -> List[Any]:
        """
        合并所有流水线的信号
        
        Args:
            results: 各流水线执行结果
            
        Returns:
            合并后的信号列表
        """
        all_signals = []
        
        for result in results.values():
            all_signals.extend(result.signals)
        
        all_signals.sort(key=lambda x: x.score if hasattr(x, 'score') else 0, reverse=True)
        
        return all_signals

