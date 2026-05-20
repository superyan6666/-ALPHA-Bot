"""
回测框架模块

实现历史策略验证：
- 历史数据回测
- 策略指标计算
- 风险收益分析
- 多周期验证
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Trade:
    """交易记录"""
    code: str
    entry_date: str
    entry_price: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    quantity: int = 100
    side: str = "LONG"
    status: str = "OPEN"  # OPEN/CLOSED/WIN/LOSS
    pnl: float = 0.0
    pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    holding_days: int = 0


@dataclass
class BacktestResult:
    """回测结果"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)


@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: str = ""
    end_date: str = ""
    initial_capital: float = 100000.0
    transaction_cost_pct: float = 0.0015
    slippage_pct: float = 0.001
    max_positions: int = 10
    position_size_pct: float = 0.10


class Backtester:
    """
    回测引擎
    
    功能：
    - 历史数据回测
    - 交易执行模拟
    - 绩效指标计算
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.current_capital = config.initial_capital if config else 100000.0
        self.current_positions: Dict[str, Trade] = {}
        self.daily_pnl = []
    
    def run_backtest(self, signals: List[Dict], historical_data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """
        执行回测
        
        Args:
            signals: 信号列表（含date, code, score等）
            historical_data: 历史数据字典 {code: DataFrame}
            
        Returns:
            回测结果
        """
        signals.sort(key=lambda x: x.get('date', ''))
        
        dates = sorted(set(s.get('date', '') for s in signals))
        
        for date in dates:
            day_signals = [s for s in signals if s.get('date') == date]
            self._process_day(date, day_signals, historical_data)
        
        return self._calculate_results()
    
    def _process_day(self, date: str, signals: List[Dict], historical_data: Dict[str, pd.DataFrame]):
        """处理单日"""
        self._close_trades(date, historical_data)
        
        self._open_new_positions(date, signals, historical_data)
        
        self._update_equity(date, historical_data)
    
    def _close_trades(self, date: str, historical_data: Dict[str, pd.DataFrame]):
        """关闭到期或止损的交易"""
        to_close = []
        
        for code, trade in self.current_positions.items():
            if code not in historical_data:
                continue
            
            df = historical_data[code]
            idx = df[df['date'] == date]
            if idx.empty:
                continue
            
            current_price = idx.iloc[0]['close']
            
            days_held = (datetime.fromisoformat(date) - 
                        datetime.fromisoformat(trade.entry_date)).days
            
            if days_held >= 20:
                to_close.append(code)
                continue
            
            atr = idx.iloc[0].get('atr', 0)
            if atr > 0:
                atr_stop = trade.entry_price - atr * 2.0
                if current_price <= atr_stop:
                    to_close.append(code)
                    continue
            
            if current_price < trade.entry_price * 0.92:
                to_close.append(code)
        
        for code in to_close:
            self._close_trade(code, date, historical_data)
    
    def _open_new_positions(self, date: str, signals: List[Dict], historical_data: Dict[str, pd.DataFrame]):
        """开新仓"""
        signals.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        max_new = self.config.max_positions - len(self.current_positions)
        
        for signal in signals[:max_new]:
            code = signal.get('code')
            if code in self.current_positions or code not in historical_data:
                continue
            
            df = historical_data[code]
            idx = df[df['date'] == date]
            if idx.empty:
                continue
            
            entry_price = idx.iloc[0]['close'] * (1 + self.config.slippage_pct)
            
            position_value = self.current_capital * self.config.position_size_pct
            quantity = int(position_value / entry_price / 100) * 100
            
            if quantity <= 0:
                continue
            
            commission = entry_price * quantity * self.config.transaction_cost_pct
            self.current_capital -= (entry_price * quantity + commission)
            
            trade = Trade(
                code=code,
                entry_date=date,
                entry_price=entry_price,
                quantity=quantity,
                status="OPEN"
            )
            
            self.current_positions[code] = trade
            self.trades.append(trade)
    
    def _close_trade(self, code: str, date: str, historical_data: Dict[str, pd.DataFrame]):
        """关闭单个交易"""
        trade = self.current_positions[code]
        
        df = historical_data[code]
        idx = df[df['date'] == date]
        
        if idx.empty:
            return
        
        exit_price = idx.iloc[0]['close'] * (1 - self.config.slippage_pct)
        commission = exit_price * trade.quantity * self.config.transaction_cost_pct
        
        trade.exit_date = date
        trade.exit_price = exit_price
        trade.holding_days = (datetime.fromisoformat(date) - 
                            datetime.fromisoformat(trade.entry_date)).days
        
        pnl = (exit_price - trade.entry_price) * trade.quantity - commission
        trade.pnl = pnl
        trade.pnl_pct = (exit_price - trade.entry_price) / trade.entry_price * 100
        
        self.current_capital += exit_price * trade.quantity - commission
        
        if trade.pnl > 0:
            trade.status = "WIN"
        else:
            trade.status = "LOSS"
        
        del self.current_positions[code]
    
    def _update_equity(self, date: str, historical_data: Dict[str, pd.DataFrame]):
        """更新权益曲线"""
        positions_value = 0.0
        
        for code, trade in self.current_positions.items():
            if code in historical_data:
                df = historical_data[code]
                idx = df[df['date'] == date]
                if not idx.empty:
                    positions_value += idx.iloc[0]['close'] * trade.quantity
        
        total_equity = self.current_capital + positions_value
        self.equity_curve.append(total_equity)
        self.daily_pnl.append(total_equity - (self.equity_curve[-2] if len(self.equity_curve) > 1 else self.config.initial_capital))
    
    def _calculate_results(self) -> BacktestResult:
        """计算回测结果"""
        closed_trades = [t for t in self.trades if t.status in ("WIN", "LOSS")]
        
        if not closed_trades:
            return BacktestResult()
        
        wins = [t for t in closed_trades if t.status == "WIN"]
        losses = [t for t in closed_trades if t.status == "LOSS"]
        
        total_pnl = sum(t.pnl for t in closed_trades)
        total_return = (self.equity_curve[-1] / self.config.initial_capital) - 1
        
        profit_factor = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses else float('inf')
        
        max_dd = self._calculate_max_drawdown()
        
        daily_returns = pd.Series([0.0] + [(self.equity_curve[i] / self.equity_curve[i-1]) - 1 
                                            for i in range(1, len(self.equity_curve))])
        
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        
        downside_returns = daily_returns[daily_returns < 0]
        sortino_ratio = np.sqrt(252) * daily_returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        return BacktestResult(
            total_trades=len(closed_trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=len(wins) / len(closed_trades),
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            total_return=total_return,
            annualized_return=self._calculate_annualized_return(total_return),
            avg_win=sum(t.pnl for t in wins) / len(wins) if wins else 0,
            avg_loss=sum(t.pnl for t in losses) / len(losses) if losses else 0,
            trades=self.trades,
            equity_curve=self.equity_curve,
            dates=self.dates
        )
    
    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if not self.equity_curve:
            return 0.0
        
        running_max = []
        current_max = self.equity_curve[0]
        
        for eq in self.equity_curve:
            current_max = max(current_max, eq)
            running_max.append(current_max)
        
        drawdowns = [(running_max[i] - self.equity_curve[i]) / running_max[i] 
                    for i in range(len(self.equity_curve))]
        
        return max(drawdowns) if drawdowns else 0.0
    
    def _calculate_annualized_return(self, total_return: float) -> float:
        """计算年化收益率"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        days = len(self.equity_curve)
        years = days / 252
        
        if years <= 0:
            return 0.0
        
        return (1 + total_return) ** (1 / years) - 1
    
    def format_result(self, result: BacktestResult) -> str:
        """格式化回测结果报告"""
        lines = ["**📊 回测结果报告**", ""]
        lines.append("-" * 50)
        lines.append(f"📈 总收益: {result.total_return*100:.1f}%")
        lines.append(f"📅 年化收益: {result.annualized_return*100:.1f}%")
        lines.append(f"🎯 胜率: {result.win_rate*100:.1f}% ({result.winning_trades}/{result.total_trades})")
        lines.append(f"💰 盈亏比: {result.avg_win/abs(result.avg_loss):.2f}")
        lines.append(f"📉 最大回撤: {result.max_drawdown*100:.1f}%")
        lines.append(f"⚖️ 夏普比率: {result.sharpe_ratio:.2f}")
        lines.append(f"🔻 Sortino: {result.sortino_ratio:.2f}")
        lines.append("-" * 50)
        
        return "\n".join(lines)


class WalkForwardTester:
    """
    滚动窗口测试器
    
    功能：
    - 多周期验证
    - 参数稳定性检验
    - 样本外测试
    """
    
    def __init__(self, window_size_days: int = 120, step_size_days: int = 30):
        self.window_size = window_size_days
        self.step_size = step_size_days
    
    def run(self, signals: List[Dict], historical_data: Dict[str, pd.DataFrame]) -> List[BacktestResult]:
        """执行滚动窗口测试"""
        signals.sort(key=lambda x: x.get('date', ''))
        
        dates = sorted(set(s.get('date', '') for s in signals))
        
        results = []
        
        for i in range(self.window_size, len(dates), self.step_size):
            train_end = i - 1
            test_start = i
            test_end = min(i + self.step_size - 1, len(dates) - 1)
            
            train_signals = [s for s in signals if s.get('date') <= dates[train_end]]
            test_signals = [s for s in signals 
                          if dates[test_start] <= s.get('date') <= dates[test_end]]
            
            backtester = Backtester()
            result = backtester.run_backtest(test_signals, historical_data)
            result.dates = [dates[test_start], dates[test_end]]
            results.append(result)
        
        return results
    
    def aggregate_results(self, results: List[BacktestResult]) -> Dict:
        """聚合滚动窗口结果"""
        if not results:
            return {}
        
        metrics = {
            'total_return': [r.total_return for r in results],
            'win_rate': [r.win_rate for r in results],
            'max_drawdown': [r.max_drawdown for r in results],
            'sharpe_ratio': [r.sharpe_ratio for r in results],
        }
        
        return {
            'mean_total_return': np.mean(metrics['total_return']),
            'std_total_return': np.std(metrics['total_return']),
            'mean_win_rate': np.mean(metrics['win_rate']),
            'std_win_rate': np.std(metrics['win_rate']),
            'max_max_drawdown': max(metrics['max_drawdown']),
            'mean_sharpe': np.mean(metrics['sharpe_ratio']),
            'pass_rate': sum(1 for r in results if r.win_rate > 0.5) / len(results),
        }

