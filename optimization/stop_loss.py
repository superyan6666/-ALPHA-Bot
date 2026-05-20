"""
动态止损模块

提供多层次止损机制：
- 硬止损（固定百分比）
- ATR动态止损
- 时间止损
- 跟踪止损（移动止损）
- 波动率自适应止损
"""
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np

log = logging.getLogger(__name__)


class StopType(Enum):
    """止损类型"""
    HARD = "硬止损"           # 固定百分比止损
    ATR = "ATR止损"          # ATR倍数止损
    TIME = "时间止损"        # 持股时间止损
    TRAILING = "跟踪止损"     # 移动止损
    VOLATILITY = "波动止损"   # 波动率自适应止损


@dataclass
class StopLevel:
    """止损级别"""
    stop_type: StopType
    stop_price: float
    stop_pct: float
    reason: str
    priority: int = 0


@dataclass
class Position:
    """持仓信息"""
    code: str
    entry_price: float
    entry_date: str
    quantity: int = 100
    peak_price: float = 0.0
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.peak_price - self.entry_price) * self.quantity
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.peak_price - self.entry_price) / self.entry_price * 100


class DynamicStopLossCalculator:
    """
    动态止损计算器
    
    功能：
    - 多层次止损计算
    - 波动率自适应
    - 跟踪止损保护
    - 止损条件判断
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 止损配置
        """
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'hard_stop_pct': 0.08,      # 硬止损8%
            'atr_multiplier': 2.0,     # ATR倍数
            'trailing_pct': 0.07,       # 跟踪止损7%
            'time_stop_days': 10,       # 时间止损10天
            'profit_protect_pct': 0.05, # 利润保护线5%
            'max_stop_pct': 0.15,       # 最大止损15%
            'volatility_window': 20,    # 波动率窗口
        }
    
    def calculate_all_stops(self, position: Position, atr: float, 
                           current_price: float, holding_days: int) -> Dict[StopType, StopLevel]:
        """
        计算所有止损级别
        
        Args:
            position: 持仓信息
            atr: ATR值
            current_price: 当前价格
            holding_days: 持股天数
            
        Returns:
            各类型止损级别字典
        """
        stops = {}
        
        hard_stop = self._calculate_hard_stop(position.entry_price)
        stops[StopType.HARD] = StopLevel(
            stop_type=StopType.HARD,
            stop_price=hard_stop,
            stop_pct=self.config['hard_stop_pct'] * 100,
            reason="固定百分比止损",
            priority=1
        )
        
        atr_stop = self._calculate_atr_stop(position.entry_price, atr)
        stops[StopType.ATR] = StopLevel(
            stop_type=StopType.ATR,
            stop_price=atr_stop,
            stop_pct=abs(atr_stop - position.entry_price) / position.entry_price * 100,
            reason=f"ATR{self.config['atr_multiplier']}倍止损",
            priority=2
        )
        
        trailing_stop = self._calculate_trailing_stop(position.peak_price)
        if trailing_stop > position.entry_price:
            trailing_pct = abs(trailing_stop - position.peak_price) / position.peak_price * 100
            stops[StopType.TRAILING] = StopLevel(
                stop_type=StopType.TRAILING,
                stop_price=trailing_stop,
                stop_pct=trailing_pct,
                reason="跟踪止损（保护利润）",
                priority=3
            )
        
        if holding_days >= self.config['time_stop_days']:
            time_stop = self._calculate_time_stop(position.entry_price, current_price)
            stops[StopType.TIME] = StopLevel(
                stop_type=StopType.TIME,
                stop_price=time_stop,
                stop_pct=abs(time_stop - position.entry_price) / position.entry_price * 100,
                reason=f"持股{holding_days}天超时止损",
                priority=4
            )
        
        vol_stop = self._calculate_volatility_stop(
            position.entry_price, atr, current_price, 
            position.unrealized_pnl_pct
        )
        if vol_stop > 0:
            stops[StopType.VOLATILITY] = StopLevel(
                stop_type=StopType.VOLATILITY,
                stop_price=vol_stop,
                stop_pct=abs(vol_stop - position.entry_price) / position.entry_price * 100,
                reason="波动率自适应止损",
                priority=5
            )
        
        return stops
    
    def _calculate_hard_stop(self, entry_price: float) -> float:
        """计算硬止损价"""
        return entry_price * (1 - self.config['hard_stop_pct'])
    
    def _calculate_atr_stop(self, entry_price: float, atr: float) -> float:
        """计算ATR止损价"""
        return entry_price - atr * self.config['atr_multiplier']
    
    def _calculate_trailing_stop(self, peak_price: float) -> float:
        """计算跟踪止损价"""
        return peak_price * (1 - self.config['trailing_pct'])
    
    def _calculate_time_stop(self, entry_price: float, current_price: float) -> float:
        """计算时间止损价"""
        time_penalty = self.config['time_stop_days'] * 0.002
        return entry_price * (1 - time_penalty)
    
    def _calculate_volatility_stop(self, entry_price: float, atr: float, 
                                  current_price: float, unrealized_pct: float) -> float:
        """
        计算波动率自适应止损
        
        规则：
        - 盈利时：允许更大波动
        - 亏损时：收紧止损
        """
        vol_adjusted_multiplier = self.config['atr_multiplier']
        
        if unrealized_pct > 10:
            vol_adjusted_multiplier *= 1.2
        elif unrealized_pct > 5:
            vol_adjusted_multiplier *= 1.0
        elif unrealized_pct > 0:
            vol_adjusted_multiplier *= 0.8
        else:
            vol_adjusted_multiplier *= 0.6
        
        return entry_price - atr * vol_adjusted_multiplier
    
    def get_active_stop(self, stops: Dict[StopType, StopLevel], 
                       current_price: float) -> Optional[StopLevel]:
        """
        获取当前触发的止损级别
        
        Args:
            stops: 所有止损级别
            current_price: 当前价格
            
        Returns:
            触发的止损级别，如果没有则返回None
        """
        active_stops = []
        
        for stop_type, level in stops.items():
            if current_price <= level.stop_price:
                active_stops.append(level)
        
        if not active_stops:
            return None
        
        active_stops.sort(key=lambda x: x.priority)
        return active_stops[0]
    
    def get_recommended_stop(self, stops: Dict[StopType, StopLevel]) -> float:
        """
        获取推荐止损价（所有止损中最严格的一个）
        
        Args:
            stops: 所有止损级别
            
        Returns:
            推荐止损价
        """
        if not stops:
            return 0.0
        
        max_stop_price = max(level.stop_price for level in stops.values())
        return max_stop_price
    
    def format_stop_report(self, stops: Dict[StopType, StopLevel], 
                          current_price: float, position: Position) -> str:
        """
        格式化止损报告
        
        Returns:
            止损报告字符串
        """
        lines = ["**止损体系报告**", ""]
        
        active = self.get_active_stop(stops, current_price)
        
        for stop_type, level in sorted(stops.items(), key=lambda x: x[1].priority):
            status = "🔴 **触发**" if active and active.stop_type == stop_type else "🟢"
            lines.append(f"{status} {level.stop_type.value}：`¥{level.stop_price:.2f}` ({level.stop_pct:.1f}%)")
            lines.append(f"   └ {level.reason}")
        
        lines.append("")
        recommended = self.get_recommended_stop(stops)
        lines.append(f"**推荐止损价**：`¥{recommended:.2f}`")
        
        if active:
            lines.append(f"🔴 **已触发**：`{active.stop_type.value}` - {active.reason}")
        
        unrealized = position.unrealized_pnl_pct
        if unrealized > 0:
            lines.append(f"**浮动盈利**：`{unrealized:+.1f}%`")
        else:
            lines.append(f"**浮动亏损**：`{unrealized:.1f}%`")
        
        return "\n".join(lines)


class StopLossOptimizer:
    """止损参数优化器"""
    
    def __init__(self):
        self.trade_history = []
    
    def add_trade(self, trade: Dict):
        """添加交易记录"""
        self.trade_history.append(trade)
    
    def optimize(self) -> Dict:
        """
        基于历史交易优化止损参数
        
        Returns:
            最优止损配置
        """
        if len(self.trade_history) < 20:
            return self._default_config()
        
        wins = [t for t in self.trade_history if t.get('pnl', 0) > 0]
        losses = [t for t in self.trade_history if t.get('pnl', 0) <= 0]
        
        if not wins or not losses:
            return self._default_config()
        
        avg_win = np.mean([t['pnl'] for t in wins])
        avg_loss = np.mean([t['pnl'] for t in losses])
        
        win_rate = len(wins) / len(self.trade_history)
        
        optimal_r = avg_win / abs(avg_loss) if avg_loss != 0 else 1.5
        
        hard_stop = 0.08
        if optimal_r < 1.0:
            hard_stop = 0.05
        elif optimal_r < 1.5:
            hard_stop = 0.06
        elif optimal_r > 2.5:
            hard_stop = 0.10
        
        return {
            'hard_stop_pct': hard_stop,
            'atr_multiplier': 2.0 if win_rate > 0.5 else 1.8,
            'trailing_pct': 0.05 + win_rate * 0.05,
            'time_stop_days': 8 if win_rate < 0.4 else 12,
        }
    
    def _default_config(self) -> Dict:
        return {
            'hard_stop_pct': 0.08,
            'atr_multiplier': 2.0,
            'trailing_pct': 0.07,
            'time_stop_days': 10,
        }

