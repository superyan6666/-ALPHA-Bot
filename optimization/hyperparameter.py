"""
超参数优化器模块

实现自动化超参数调优：
- Optuna集成
- 贝叶斯优化
- 参数空间定义
- 多目标优化
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import json
import os

try:
    import optuna
    from optuna.trial import TrialState
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

log = logging.getLogger(__name__)


@dataclass
class ParamConfig:
    """参数配置"""
    name: str
    param_type: str  # int, float, categorical
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List] = None
    step: Optional[float] = None


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_value: float = 0.0
    trials_count: int = 0
    best_trial_id: int = 0
    optimization_time: float = 0.0


class HyperparameterOptimizer:
    """
    超参数优化器
    
    功能：
    - 贝叶斯优化
    - 参数空间搜索
    - 多目标优化
    """
    
    def __init__(self, study_name: str = "strategy_optimization",
                 storage_path: Optional[str] = None):
        self.study_name = study_name
        self.storage_path = storage_path or f"sqlite:///optuna_{study_name}.db"
        self.study = None
        self.best_params = {}
        self._param_configs: List[ParamConfig] = []
    
    def add_param(self, name: str, param_type: str, **kwargs):
        """
        添加参数定义
        
        Args:
            name: 参数名
            param_type: 参数类型 (int, float, categorical)
            **kwargs: 额外参数
        """
        config = ParamConfig(name=name, param_type=param_type, **kwargs)
        self._param_configs.append(config)
        log.info(f"✅ 添加参数: {name} ({param_type})")
    
    def suggest_params(self, trial) -> Dict[str, Any]:
        """根据参数配置生成建议参数"""
        params = {}
        
        for config in self._param_configs:
            if config.param_type == 'float':
                params[config.name] = trial.suggest_float(
                    config.name, 
                    config.low, 
                    config.high,
                    step=config.step
                )
            elif config.param_type == 'int':
                params[config.name] = trial.suggest_int(
                    config.name,
                    int(config.low),
                    int(config.high),
                    step=config.step
                )
            elif config.param_type == 'categorical':
                params[config.name] = trial.suggest_categorical(
                    config.name,
                    config.choices
                )
        
        return params
    
    def optimize(self, objective: Callable, n_trials: int = 50,
                direction: str = "maximize") -> OptimizationResult:
        """
        执行优化
        
        Args:
            objective: 目标函数
            n_trials: 试验次数
            direction: 优化方向 (maximize/minimize)
            
        Returns:
            优化结果
        """
        if not OPTUNA_AVAILABLE:
            log.warning("❌ Optuna 未安装，跳过超参数优化")
            return OptimizationResult()
        
        if not self._param_configs:
            log.warning("❌ 未定义参数空间")
            return OptimizationResult()
        
        start_time = datetime.now()
        
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_path,
            direction=direction,
            load_if_exists=True
        )
        
        self.study.optimize(objective, n_trials=n_trials)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        result = OptimizationResult(
            best_params=self.study.best_params,
            best_value=self.study.best_value,
            trials_count=len(self.study.trials),
            best_trial_id=self.study.best_trial.number,
            optimization_time=elapsed
        )
        
        self.best_params = self.study.best_params
        
        log.info(f"✅ 优化完成: 最佳值={result.best_value:.4f}, 参数={result.best_params}")
        
        return result
    
    def get_best_params(self) -> Dict[str, Any]:
        """获取最佳参数"""
        return self.best_params
    
    def save_results(self, filepath: str):
        """保存优化结果"""
        if not self.study:
            return
        
        results = {
            'best_params': self.study.best_params,
            'best_value': float(self.study.best_value),
            'trials': [
                {
                    'id': t.number,
                    'params': t.params,
                    'value': float(t.value) if t.value else None,
                    'state': t.state.name
                }
                for t in self.study.trials
            ],
            'optimization_time': self._last_optimization_time
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def format_report(self, result: OptimizationResult) -> str:
        """格式化优化报告"""
        lines = ["**⚙️ 超参数优化报告**", ""]
        lines.append("-" * 50)
        lines.append(f"🎯 最佳值: {result.best_value:.4f}")
        lines.append(f"📊 试验次数: {result.trials_count}")
        lines.append(f"⏱️ 优化耗时: {result.optimization_time:.2f}秒")
        lines.append(f"🏆 最优试验ID: {result.best_trial_id}")
        lines.append("")
        lines.append("**最佳参数配置:**")
        for name, value in result.best_params.items():
            lines.append(f"- {name}: {value}")
        lines.append("-" * 50)
        
        return "\n".join(lines)


class FactorWeightOptimizer:
    """
    因子权重优化器
    
    功能：
    - 基于回测优化因子权重
    - 约束权重和为1
    - 考虑因子相关性
    """
    
    def __init__(self, factor_names: List[str]):
        self.factor_names = factor_names
        self.optimizer = HyperparameterOptimizer("factor_weight_opt")
        
        for name in factor_names:
            self.optimizer.add_param(name, 'float', low=0.01, high=1.0)
    
    def optimize_weights(self, backtest_func: Callable, n_trials: int = 30) -> Dict[str, float]:
        """
        优化因子权重
        
        Args:
            backtest_func: 回测函数，接受权重字典返回收益
            n_trials: 试验次数
            
        Returns:
            优化后的权重字典
        """
        def objective(trial):
            raw_weights = {}
            total = 0.0
            
            for name in self.factor_names:
                w = trial.suggest_float(name, 0.01, 1.0)
                raw_weights[name] = w
                total += w
            
            normalized_weights = {k: v / total for k, v in raw_weights.items()}
            
            return backtest_func(normalized_weights)
        
        result = self.optimizer.optimize(objective, n_trials=n_trials)
        
        total = sum(result.best_params.values())
        return {k: v / total for k, v in result.best_params.items()}


class ThresholdOptimizer:
    """
    阈值优化器
    
    功能：
    - 优化信号阈值
    - 优化止损/止盈参数
    - 优化过滤条件
    """
    
    def __init__(self):
        self.optimizer = HyperparameterOptimizer("threshold_opt")
    
    def add_threshold(self, name: str, low: float, high: float, step: float = 0.01):
        """添加阈值参数"""
        self.optimizer.add_param(name, 'float', low=low, high=high, step=step)
    
    def add_int_threshold(self, name: str, low: int, high: int, step: int = 1):
        """添加整数阈值参数"""
        self.optimizer.add_param(name, 'int', low=low, high=high, step=step)
    
    def optimize(self, objective: Callable, n_trials: int = 50) -> OptimizationResult:
        """执行优化"""
        return self.optimizer.optimize(objective, n_trials=n_trials)


class StrategyParameterOptimizer:
    """
    策略参数优化器（综合）
    
    功能：
    - 一站式策略参数优化
    - 预设常见参数空间
    """
    
    def __init__(self):
        self.optimizer = HyperparameterOptimizer("strategy_opt")
    
    def setup_default_parameters(self):
        """设置默认参数空间"""
        self.optimizer.add_param('rsi_overbought', 'int', low=60, high=80, step=5)
        self.optimizer.add_param('rsi_oversold', 'int', low=20, high=40, step=5)
        self.optimizer.add_param('atr_multiplier', 'float', low=1.5, high=3.0, step=0.25)
        self.optimizer.add_param('stop_loss_pct', 'float', low=0.05, high=0.15, step=0.01)
        self.optimizer.add_param('min_score', 'int', low=50, high=80, step=5)
        self.optimizer.add_param('max_positions', 'int', low=5, high=15, step=1)
        self.optimizer.add_param('position_size', 'float', low=0.05, high=0.15, step=0.01)
        self.optimizer.add_param('signal_validity_days', 'int', low=1, high=7, step=1)
    
    def optimize(self, objective: Callable, n_trials: int = 50) -> OptimizationResult:
        """执行优化"""
        return self.optimizer.optimize(objective, n_trials=n_trials)
    
    def get_default_params(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'atr_multiplier': 2.0,
            'stop_loss_pct': 0.08,
            'min_score': 60,
            'max_positions': 10,
            'position_size': 0.10,
            'signal_validity_days': 3,
        }

