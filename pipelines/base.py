
"""
流水线基类模块
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Any, Dict
import pandas as pd

from core.config import StrategyType


@dataclass
class PipelineResult:
    """流水线执行结果"""
    signals: List[Any]
    watchlist: List[tuple]
    market_msg: str
    meta_info: Dict[str, Any]


class BasePipeline(ABC):
    """流水线基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass
    
    @abstractmethod
    def is_enabled(self) -> bool:
        """是否启用"""
        pass
    
    @abstractmethod
    def run(self, shared_context: dict) -> PipelineResult:
        """
        运行流水线
        
        Args:
            shared_context: 共享上下文，包含数据、市场状态等
            
        Returns:
            PipelineResult 执行结果
        """
        pass

