"""
因子有效性监控模块

提供因子IC/IR分析、动态权重调整功能
"""
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class FactorMetrics:
    """因子指标数据"""
    name: str
    ic: float = 0.0
    ir: float = 0.0
    ic_std: float = 0.0
    hit_rate: float = 0.0
    decay_rate: float = 0.0
    last_updated: str = ""


@dataclass
class FactorConfig:
    """因子配置"""
    name: str
    base_weight: float
    min_weight: float = 0.1
    max_weight: float = 2.0
    ic_threshold: float = 0.02
    ir_threshold: float = 0.3


class FactorEffectivenessAnalyzer:
    """
    因子有效性分析器
    
    功能：
    - 计算因子IC（信息系数）
    - 计算因子IR（信息比率）
    - 追踪因子衰减
    - 动态调整因子权重
    """
    
    def __init__(self, history_window: int = 60):
        """
        Args:
            history_window: 历史IC窗口（交易日）
        """
        self.history_window = history_window
        self.ic_history: Dict[str, deque] = {}
        self.factor_configs: Dict[str, FactorConfig] = {}
        self.metrics_file = self._get_metrics_file()
        self._load_metrics()
    
    def _get_metrics_file(self) -> str:
        """获取指标文件路径"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, "factor_metrics.json")
    
    def _load_metrics(self):
        """加载历史指标"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    for name, history in data.get('ic_history', {}).items():
                        self.ic_history[name] = deque(history[-self.history_window:], 
                                                      maxlen=self.history_window)
            except Exception as e:
                log.warning(f"加载因子指标失败: {e}")
    
    def _save_metrics(self):
        """保存历史指标"""
        try:
            data = {
                'ic_history': {k: list(v) for k, v in self.ic_history.items()},
                'last_updated': datetime.now().isoformat()
            }
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log.warning(f"保存因子指标失败: {e}")
    
    def register_factor(self, name: str, base_weight: float = 1.0, 
                       min_weight: float = 0.1, max_weight: float = 2.0):
        """注册因子配置"""
        self.factor_configs[name] = FactorConfig(
            name=name,
            base_weight=base_weight,
            min_weight=min_weight,
            max_weight=max_weight
        )
        if name not in self.ic_history:
            self.ic_history[name] = deque(maxlen=self.history_window)
    
    def update_ic(self, factor_name: str, ic_value: float):
        """更新因子IC值"""
        if factor_name not in self.ic_history:
            self.register_factor(factor_name)
        
        self.ic_history[factor_name].append(ic_value)
        self._save_metrics()
    
    def calculate_ic(self, factor_values: pd.Series, forward_returns: pd.Series) -> float:
        """
        计算信息系数（IC）
        
        Args:
            factor_values: 因子值序列
            forward_returns: 未来收益序列
            
        Returns:
            IC值（-1到1之间，越接近1越好）
        """
        valid_idx = factor_values.notna() & forward_returns.notna()
        if valid_idx.sum() < 10:
            return 0.0
        
        return factor_values[valid_idx].corr(forward_returns[valid_idx])
    
    def calculate_rank_ic(self, factor_values: pd.Series, forward_returns: pd.Series) -> float:
        """
        计算秩相关系数（Rank IC / Spearman IC）
        
        更稳健的IC计算方式，对极端值不敏感
        """
        valid_idx = factor_values.notna() & forward_returns.notna()
        if valid_idx.sum() < 10:
            return 0.0
        
        return factor_values[valid_idx].corr(forward_returns[valid_idx], method='spearman')
    
    def calculate_metrics(self, factor_name: str) -> FactorMetrics:
        """
        计算因子完整指标
        
        Returns:
            FactorMetrics对象
        """
        ic_series = pd.Series(list(self.ic_history.get(factor_name, [])))
        
        if len(ic_series) < 5:
            return FactorMetrics(name=factor_name)
        
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ir = ic_mean / ic_std if ic_std > 0 else 0.0
        
        hit_rate = (ic_series > 0).mean()
        
        decay_rate = self._calculate_decay_rate(ic_series)
        
        return FactorMetrics(
            name=factor_name,
            ic=ic_mean,
            ir=ir,
            ic_std=ic_std,
            hit_rate=hit_rate,
            decay_rate=decay_rate,
            last_updated=datetime.now().isoformat()
        )
    
    def _calculate_decay_rate(self, ic_series: pd.Series) -> float:
        """计算因子IC衰减率"""
        if len(ic_series) < 20:
            return 0.0
        
        recent = ic_series.iloc[-10:].mean()
        earlier = ic_series.iloc[-20:-10].mean()
        
        if abs(earlier) < 0.001:
            return 0.0
        
        return (recent - earlier) / abs(earlier)
    
    def get_all_metrics(self) -> Dict[str, FactorMetrics]:
        """获取所有因子指标"""
        return {
            name: self.calculate_metrics(name) 
            for name in self.ic_history.keys()
        }
    
    def adjust_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        根据IC动态调整因子权重
        
        调整规则：
        - IC > 0.05: 权重上调
        - IC > 0.02: 权重不变
        - 0 < IC < 0.02: 权重下调
        - IC < 0: 权重大幅下调
        
        Args:
            base_weights: 基础权重字典
            
        Returns:
            调整后的权重字典
        """
        adjusted = {}
        
        for factor, base_w in base_weights.items():
            metrics = self.calculate_metrics(factor)
            config = self.factor_configs.get(factor)
            
            if config is None:
                adjusted[factor] = base_w
                continue
            
            ic = metrics.ic
            ir = metrics.ir
            
            if len(self.ic_history.get(factor, [])) < 10:
                adjusted[factor] = base_w
                continue
            
            multiplier = 1.0
            
            if ir > 0.5:
                multiplier = 1.5
            elif ir > 0.3:
                multiplier = 1.2
            elif ir > 0.1:
                multiplier = 1.0
            elif ir > 0:
                multiplier = 0.8
            else:
                multiplier = 0.5
            
            if metrics.decay_rate < -0.3:
                multiplier *= 0.7
            elif metrics.decay_rate > 0.2:
                multiplier *= 1.2
            
            new_weight = base_w * multiplier
            new_weight = max(config.min_weight * base_w, 
                            min(config.max_weight * base_w, new_weight))
            
            adjusted[factor] = new_weight
        
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted
    
    def is_factor_valid(self, factor_name: str) -> bool:
        """检查因子是否仍然有效"""
        metrics = self.calculate_metrics(factor_name)
        
        if len(self.ic_history.get(factor_name, [])) < 10:
            return True
        
        if metrics.ir < 0.1 and metrics.hit_rate < 0.5:
            return False
        
        return True
    
    def get_report(self) -> str:
        """生成因子有效性报告"""
        metrics = self.get_all_metrics()
        
        lines = ["📊 **因子有效性报告**", ""]
        lines.append("-" * 50)
        
        for name, m in sorted(metrics.items(), key=lambda x: x[1].ic, reverse=True):
            status = "✅" if self.is_factor_valid(name) else "❌"
            lines.append(f"{status} **{name}**")
            lines.append(f"   IC: {m.ic:.4f} | IR: {m.ir:.4f} | 命中率: {m.hit_rate:.1%}")
            lines.append(f"   衰减率: {m.decay_rate:+.1%}")
            lines.append("")
        
        return "\n".join(lines)


class FactorCorrelationAnalyzer:
    """因子相关性分析器"""
    
    def __init__(self, high_threshold: float = 0.7):
        """
        Args:
            high_threshold: 高相关性阈值（超过此值认为因子冗余）
        """
        self.high_threshold = high_threshold
        self.factor_values: Dict[str, pd.Series] = {}
    
    def add_factor(self, name: str, values: pd.Series):
        """添加因子数据"""
        self.factor_values[name] = values.dropna()
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """计算相关性矩阵"""
        if len(self.factor_values) < 2:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.factor_values)
        return df.corr()
    
    def find_redundant_factors(self) -> List[Tuple[str, str, float]]:
        """
        找出冗余因子对
        
        Returns:
            [(因子1, 因子2, 相关系数), ...]
        """
        corr_matrix = self.calculate_correlation_matrix()
        if corr_matrix.empty:
            return []
        
        redundant = []
        for i, f1 in enumerate(corr_matrix.columns):
            for f2 in corr_matrix.columns[i+1:]:
                corr = corr_matrix.loc[f1, f2]
                if abs(corr) >= self.high_threshold:
                    redundant.append((f1, f2, corr))
        
        return sorted(redundant, key=lambda x: abs(x[2]), reverse=True)
    
    def get_groups(self) -> List[List[str]]:
        """
        将高相关因子分组（同组内因子高度相关）
        
        Returns:
            [[因子列表], ...]
        """
        redundant = self.find_redundant_factors()
        if not redundant:
            return [[f] for f in self.factor_values.keys()]
        
        groups = []
        assigned = set()
        
        for f1, f2, _ in redundant:
            if f1 in assigned and f2 in assigned:
                continue
            
            found_group = None
            for g in groups:
                if f1 in g or f2 in g:
                    found_group = g
                    break
            
            if found_group:
                if f1 not in assigned:
                    found_group.append(f1)
                    assigned.add(f1)
                if f2 not in assigned:
                    found_group.append(f2)
                    assigned.add(f2)
            else:
                groups.append([f1, f2])
                assigned.add(f1)
                assigned.add(f2)
        
        for f in self.factor_values:
            if f not in assigned:
                groups.append([f])
                assigned.add(f)
        
        return groups

