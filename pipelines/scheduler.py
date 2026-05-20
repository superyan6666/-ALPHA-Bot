"""
流水线调度器
"""
from typing import List, Dict, Any
import logging

from core.config import strategy_config
from pipelines.base import BasePipeline, PipelineResult

log = logging.getLogger(__name__)


class PipelineScheduler:
    """流水线调度器"""
    
    def __init__(self, data_lake):
        self._data_lake = data_lake
        self._pipelines: Dict[str, BasePipeline] = {}
        self._init_pipelines()
    
    def _init_pipelines(self):
        """初始化所有流水线"""
        try:
            from .stock import StockPipeline
            self._pipelines['stock'] = StockPipeline(self._data_lake)
            log.info("✅ 股票策略流水线已注册")
        except Exception as e:
            log.warning(f"股票策略流水线注册失败: {e}")
        
        if strategy_config.is_enabled('etf'):
            try:
                from .etf import ETFPipeline
                self._pipelines['etf'] = ETFPipeline(
                    self._data_lake, 
                    strategy_config.etf
                )
                log.info("✅ ETF轮动流水线已注册")
            except Exception as e:
                log.warning(f"ETF轮动流水线注册失败: {e}")
        
        if strategy_config.is_enabled('sector'):
            try:
                from .sector import SectorPipeline
                self._pipelines['sector'] = SectorPipeline(
                    self._data_lake,
                    strategy_config.sector
                )
                log.info("✅ 行业轮动流水线已注册")
            except Exception as e:
                log.warning(f"行业轮动流水线注册失败: {e}")
        
        if strategy_config.is_enabled('cb'):
            try:
                from .cb import CBPipeline
                self._pipelines['cb'] = CBPipeline(
                    self._data_lake,
                    strategy_config.cb
                )
                log.info("✅ 可转债流水线已注册")
            except Exception as e:
                log.warning(f"可转债流水线注册失败: {e}")
        
        log.info(f"📋 流水线调度器初始化完成，共注册 {len(self._pipelines)} 个流水线")
        log.info(f"   已注册: {list(self._pipelines.keys())}")
    
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
            strategy_type: 策略类型 (stock/etf/sector/cb)
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
        
        pipeline = self._pipelines[strategy_type]
        try:
            return pipeline.run(shared_context)
        except Exception as e:
            log.error(f"❌ {pipeline.name} 执行失败: {e}")
            return PipelineResult(
                signals=[],
                watchlist=[],
                market_msg=f"执行失败: {str(e)}",
                meta_info={"error": str(e)}
            )
    
    def run_stock(self, shared_context: dict = None) -> PipelineResult:
        """运行股票策略流水线"""
        return self.run_strategy('stock', shared_context)
    
    def run_etf(self, shared_context: dict = None) -> PipelineResult:
        """运行ETF轮动流水线"""
        return self.run_strategy('etf', shared_context)
    
    def run_sector(self, shared_context: dict = None) -> PipelineResult:
        """运行行业轮动流水线"""
        return self.run_strategy('sector', shared_context)
    
    def run_cb(self, shared_context: dict = None) -> PipelineResult:
        """运行可转债流水线"""
        return self.run_strategy('cb', shared_context)
    
    def list_pipelines(self) -> List[Dict[str, str]]:
        """列出所有流水线"""
        return [
            {"name": p.name, "type": name, "enabled": "True"}
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
        
        all_signals.sort(
            key=lambda x: getattr(x, 'score', 0) if hasattr(x, 'score') else 0, 
            reverse=True
        )
        
        return all_signals
    
    def get_signal_summary(self, results: Dict[str, PipelineResult]) -> Dict[str, int]:
        """获取信号汇总"""
        summary = {}
        for strategy, result in results.items():
            summary[strategy] = {
                "signals": len(result.signals),
                "watchlist": len(result.watchlist),
                "message": result.market_msg[:50] + "..." if len(result.market_msg) > 50 else result.market_msg
            }
        return summary

