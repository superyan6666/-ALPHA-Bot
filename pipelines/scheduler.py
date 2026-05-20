"""
流水线调度器

集成所有优化模块：
- 因子有效性分析
- 动态止损
- 信号过滤
- 市场状态识别
- 风险平价配置
- 增量更新机制
- 回测框架
- 超参数优化
"""
from typing import List, Dict, Any, Optional
import logging

from core.config import strategy_config
from pipelines.base import BasePipeline, PipelineResult

log = logging.getLogger(__name__)


class PipelineScheduler:
    """流水线调度器"""
    
    def __init__(self, data_lake, enable_optimization: bool = True):
        """
        Args:
            data_lake: 数据湖实例
            enable_optimization: 是否启用优化模块
        """
        self._data_lake = data_lake
        self._pipelines: Dict[str, BasePipeline] = {}
        self._enable_optimization = enable_optimization
        self._optimizers = {}
        self._event_manager = None
        self._incremental_updater = None
        self._risk_parity_optimizer = None
        
        self._init_pipelines()
        
        if enable_optimization:
            self._init_optimizers()
            self._init_event_system()
    
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
    
    def _init_optimizers(self):
        """初始化优化器"""
        try:
            from optimization import (
                FactorEffectivenessAnalyzer,
                DynamicStopLossCalculator,
                SignalFilter,
                MarketStateDetector,
                RiskParityOptimizer,
                Backtester,
                WalkForwardTester,
                StrategyParameterOptimizer,
            )
            
            self._optimizers['factor_analyzer'] = FactorEffectivenessAnalyzer()
            self._optimizers['stop_loss'] = DynamicStopLossCalculator()
            self._optimizers['signal_filter'] = SignalFilter()
            self._optimizers['market_detector'] = MarketStateDetector()
            self._optimizers['risk_parity'] = RiskParityOptimizer()
            self._optimizers['backtester'] = Backtester()
            self._optimizers['walk_forward'] = WalkForwardTester()
            self._optimizers['param_optimizer'] = StrategyParameterOptimizer()
            
            self._risk_parity_optimizer = RiskParityOptimizer()
            
            log.info("✅ 优化器模块初始化完成")
        except Exception as e:
            log.warning(f"优化器初始化失败: {e}")
    
    def _init_event_system(self):
        """初始化事件系统"""
        try:
            from optimization import (
                EventManager,
                IncrementalUpdater,
                PipelineEventAdapter,
            )
            
            self._event_manager = EventManager()
            self._incremental_updater = IncrementalUpdater(self._data_lake)
            self._event_adapter = PipelineEventAdapter(self._event_manager)
            
            log.info("✅ 事件系统初始化完成")
        except Exception as e:
            log.warning(f"事件系统初始化失败: {e}")
    
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
        
        if self._enable_optimization and 'market_detector' in self._optimizers:
            shared_context = self._enrich_context_with_market_state(shared_context)
        
        results = {}
        
        for name, pipeline in self._pipelines.items():
            try:
                log.info(f"🚀 启动流水线: {pipeline.name}")
                result = pipeline.run(shared_context)
                
                if self._enable_optimization:
                    result = self._apply_optimizations(result, shared_context)
                
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
        
        if self._event_adapter:
            self._publish_results(results)
        
        return results
    
    def _enrich_context_with_market_state(self, context: dict) -> dict:
        """使用市场状态检测器丰富上下文"""
        detector = self._optimizers['market_detector']
        
        market_data = context.get('market_data', {})
        if market_data:
            try:
                state = detector.detect(market_data)
                context['market_regime'] = state.regime.value
                context['market_state'] = state
                context['factor_weights'] = state.get_factor_weights()
                
                log.info(f"📊 市场状态: {detector.get_regime_string(state)}")
            except Exception as e:
                log.debug(f"市场状态检测失败: {e}")
        
        return context
    
    def _apply_optimizations(self, result: PipelineResult, context: dict) -> PipelineResult:
        """应用所有优化"""
        if 'signal_filter' in self._optimizers:
            result = self._apply_signal_filter(result, context)
        
        if 'risk_parity' in self._optimizers and result.signals:
            result = self._apply_risk_parity(result, context)
        
        return result
    
    def _apply_signal_filter(self, result: PipelineResult, context: dict) -> PipelineResult:
        """应用信号过滤器"""
        filter_engine = self._optimizers['signal_filter']
        
        sector_map = context.get('sector_map', {})
        liquidity_map = context.get('liquidity_map', {})
        current_positions = context.get('current_positions', set())
        
        try:
            filtered = filter_engine.filter(
                result.signals,
                sector_map=sector_map,
                liquidity_map=liquidity_map,
                current_positions=current_positions
            )
            
            result.signals = filtered.passed_signals
            
            if filtered.filter_stats:
                log.info(f"🔍 信号过滤: 通过{len(filtered.passed_signals)}只, "
                        f"过滤{len(filtered.filtered_signals)}只")
            
        except Exception as e:
            log.debug(f"信号过滤失败: {e}")
        
        return result
    
    def _apply_risk_parity(self, result: PipelineResult, context: dict) -> PipelineResult:
        """应用风险平价配置"""
        optimizer = self._optimizers['risk_parity']
        
        sector_map = context.get('sector_map', {})
        volatilities = context.get('volatilities', {})
        current_weights = context.get('current_weights', {})
        
        try:
            allocations = optimizer.optimize(
                result.signals,
                sector_map=sector_map,
                volatilities=volatilities,
                current_weights=current_weights
            )
            
            result.meta_info['allocations'] = allocations
            
            log.info(f"⚖️ 风险平价配置完成，{len(allocations)}个资产")
            
        except Exception as e:
            log.debug(f"风险平价配置失败: {e}")
        
        return result
    
    def _publish_results(self, results: Dict[str, PipelineResult]):
        """发布结果事件"""
        for strategy, result in results.items():
            if self._event_adapter and result.signals:
                try:
                    self._event_adapter.publish_signals(result.signals)
                except Exception as e:
                    log.debug(f"事件发布失败: {e}")
    
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
        
        if self._enable_optimization and 'market_detector' in self._optimizers:
            shared_context = self._enrich_context_with_market_state(shared_context)
        
        pipeline = self._pipelines[strategy_type]
        try:
            result = pipeline.run(shared_context)
            
            if self._enable_optimization:
                result = self._apply_optimizations(result, shared_context)
            
            return result
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
    
    def run_backtest(self, signals: List[Dict], historical_data: Dict) -> dict:
        """
        执行回测
        
        Args:
            signals: 信号列表
            historical_data: 历史数据
            
        Returns:
            回测结果
        """
        if 'backtester' not in self._optimizers:
            return {"error": "回测模块未启用"}
        
        try:
            backtester = self._optimizers['backtester']
            result = backtester.run_backtest(signals, historical_data)
            
            return {
                'success': True,
                'result': result.__dict__,
                'report': backtester.format_result(result)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def run_walk_forward(self, signals: List[Dict], historical_data: Dict) -> dict:
        """
        执行滚动窗口测试
        
        Args:
            signals: 信号列表
            historical_data: 历史数据
            
        Returns:
            滚动窗口测试结果
        """
        if 'walk_forward' not in self._optimizers:
            return {"error": "滚动窗口测试模块未启用"}
        
        try:
            wf_tester = self._optimizers['walk_forward']
            results = wf_tester.run(signals, historical_data)
            aggregate = wf_tester.aggregate_results(results)
            
            return {
                'success': True,
                'results': [r.__dict__ for r in results],
                'aggregate': aggregate
            }
        except Exception as e:
            return {"error": str(e)}
    
    def optimize_parameters(self, objective_func: Callable, n_trials: int = 50) -> dict:
        """
        执行超参数优化
        
        Args:
            objective_func: 目标函数
            n_trials: 试验次数
            
        Returns:
            优化结果
        """
        if 'param_optimizer' not in self._optimizers:
            return {"error": "超参数优化模块未启用"}
        
        try:
            optimizer = self._optimizers['param_optimizer']
            optimizer.setup_default_parameters()
            result = optimizer.optimize(objective_func, n_trials=n_trials)
            
            return {
                'success': True,
                'best_params': result.best_params,
                'best_value': result.best_value,
                'report': optimizer.format_report(result)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def list_pipelines(self) -> List[Dict[str, str]]:
        """列出所有流水线"""
        return [
            {"name": p.name, "type": name, "enabled": "True"}
            for name, p in self._pipelines.items()
        ]
    
    def get_all_signals(self, results: Dict[str, PipelineResult]) -> List[Any]:
        """合并所有流水线的信号"""
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
    
    def get_optimizer(self, name: str):
        """获取指定优化器"""
        return self._optimizers.get(name)
    
    def get_factor_report(self) -> str:
        """获取因子有效性报告"""
        if 'factor_analyzer' in self._optimizers:
            return self._optimizers['factor_analyzer'].get_report()
        return "因子分析器未启用"
    
    def get_market_state_report(self) -> str:
        """获取市场状态报告"""
        if 'market_detector' in self._optimizers:
            state = self._optimizers['market_detector'].state_history[-1] if self._optimizers['market_detector'].state_history else None
            if state:
                return self._optimizers['market_detector'].format_state_report(state)
        return "市场状态检测器未启用"
    
    def get_risk_parity_report(self, signals: List, sector_map: Dict) -> str:
        """获取风险平价配置报告"""
        if 'risk_parity' in self._optimizers and signals:
            allocations = self._optimizers['risk_parity'].optimize(
                signals, sector_map, {}
            )
            return self._optimizers['risk_parity'].format_allocation_report(allocations)
        return "风险平价优化器未启用或无信号"

