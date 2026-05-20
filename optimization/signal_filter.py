"""
信号过滤器模块

提供多层次信号过滤：
- 行业持仓限制
- 相似度去重
- 流动性过滤
- 时间窗口管理
- 信号有效期控制
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Signal:
    """信号基类"""
    code: str
    name: str
    score: float
    sector: str = ""
    industry: str = ""
    price: float = 0.0
    timestamp: str = ""
    
    def __hash__(self):
        return hash(self.code)


@dataclass
class FilterResult:
    """过滤结果"""
    passed_signals: List[Signal]
    filtered_signals: List[Tuple[Signal, str]]
    filter_stats: Dict[str, int]


class SignalFilter:
    """
    信号过滤器
    
    功能：
    - 行业持仓分散化
    - 标的相似度去重
    - 流动性过滤
    - 信号有效期管理
    - 动态阈值调整
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 过滤器配置
        """
        self.config = config or self._default_config()
        self.sector_counts: Dict[str, int] = defaultdict(int)
        self.position_sectors: Dict[str, str] = {}
        self.recent_signals: List[Tuple[str, datetime]] = []
        self.signal_prices: Dict[str, float] = {}
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'max_per_sector': 2,
            'max_total_signals': 10,
            'min_score': 60,
            'min_liquidity': 5000 * 10000,
            'max_similarity': 0.8,
            'signal_validity_days': 3,
            'min_price_change_filter': 0.02,
            'enable_sector_limit': True,
            'enable_deduplication': True,
            'enable_liquidity_filter': True,
        }
    
    def filter(self, signals: List, sector_map: Optional[Dict[str, str]] = None,
               liquidity_map: Optional[Dict[str, float]] = None,
               current_positions: Optional[Set[str]] = None) -> FilterResult:
        """
        执行信号过滤
        
        Args:
            signals: 待过滤信号列表
            sector_map: 股票代码到行业的映射
            liquidity_map: 股票代码到成交额的映射
            current_positions: 当前持仓代码集合
            
        Returns:
            FilterResult
        """
        if sector_map is None:
            sector_map = {}
        if liquidity_map is None:
            liquidity_map = {}
        if current_positions is None:
            current_positions = set()
        
        passed = []
        filtered = []
        filter_stats = defaultdict(int)
        
        sorted_signals = sorted(signals, key=lambda x: getattr(x, 'score', 0), reverse=True)
        
        sector_counts = defaultdict(int)
        
        for signal in sorted_signals:
            code = getattr(signal, 'code', str(signal))
            name = getattr(signal, 'name', code)
            score = getattr(signal, 'score', 0)
            sector = sector_map.get(code, getattr(signal, 'sector', ''))
            
            reason = self._check_filters(
                signal, score, sector, sector_counts,
                liquidity_map, current_positions
            )
            
            if reason:
                filtered.append((signal, reason))
                filter_stats[reason] += 1
            else:
                passed.append(signal)
                sector_counts[sector] += 1
                
                if code not in self.position_sectors:
                    self.position_sectors[code] = sector
        
        if len(passed) > self.config['max_total_signals']:
            overflow = passed[self.config['max_total_signals']:]
            passed = passed[:self.config['max_total_signals']]
            for sig in overflow:
                filtered.append((sig, f"超过总数限制({self.config['max_total_signals']})"))
                filter_stats[f"超过总数限制({self.config['max_total_signals']})"] += 1
        
        self._update_recent_signals(passed)
        
        return FilterResult(
            passed_signals=passed,
            filtered_signals=filtered,
            filter_stats=dict(filter_stats)
        )
    
    def _check_filters(self, signal, score: float, sector: str,
                      sector_counts: Dict[str, int],
                      liquidity_map: Dict[str, float],
                      current_positions: Set[str]) -> Optional[str]:
        """检查各项过滤条件"""
        code = getattr(signal, 'code', str(signal))
        
        if score < self.config['min_score']:
            return f"分数低于阈值({score:.0f}<{self.config['min_score']})"
        
        if self.config['enable_sector_limit']:
            if sector_counts.get(sector, 0) >= self.config['max_per_sector']:
                return f"行业{sector}已达上限({self.config['max_per_sector']})"
        
        if self.config['enable_liquidity_filter']:
            liquidity = liquidity_map.get(code, 0)
            if liquidity > 0 and liquidity < self.config['min_liquidity']:
                return f"流动性不足(¥{liquidity/1e8:.1f}亿<{self.config['min_liquidity']/1e8:.0f}亿)"
        
        if code in current_positions:
            return "已在持仓中"
        
        if self._is_recently_pushed(code):
            return f"近期已推送({self.config['signal_validity_days']}天内)"
        
        if self.config['enable_deduplication']:
            dup_reason = self._check_similarity(signal, passed_signals=[])
            if dup_reason:
                return dup_reason
        
        return None
    
    def _is_recently_pushed(self, code: str) -> bool:
        """检查是否近期推送过"""
        cutoff = datetime.now() - timedelta(days=self.config['signal_validity_days'])
        for c, dt in self.recent_signals:
            if c == code and dt > cutoff:
                return True
        return False
    
    def _check_similarity(self, signal, passed_signals: List) -> Optional[str]:
        """检查信号相似度"""
        if not passed_signals:
            return None
        
        code = getattr(signal, 'code', str(signal))
        
        for passed in passed_signals[-5:]:
            p_code = getattr(passed, 'code', str(passed))
            
            if code == p_code:
                return f"与{p_code}重复"
            
            p_sector = getattr(passed, 'sector', '')
            c_sector = getattr(signal, 'sector', '')
            
            if p_sector and c_sector and p_sector == c_sector:
                score_diff = abs(getattr(signal, 'score', 0) - getattr(passed, 'score', 0))
                if score_diff < 5:
                    return f"与{p_code}(同行业)高度相似"
        
        return None
    
    def _update_recent_signals(self, passed_signals: List):
        """更新近期信号记录"""
        now = datetime.now()
        self.recent_signals = [
            (c, dt) for c, dt in self.recent_signals
            if now - dt < timedelta(days=self.config['signal_validity_days'] * 2)
        ]
        
        for signal in passed_signals:
            code = getattr(signal, 'code', str(signal))
            self.recent_signals.append((code, now))
    
    def format_filter_report(self, result: FilterResult) -> str:
        """格式化过滤报告"""
        lines = ["**信号过滤报告**", ""]
        lines.append(f"通过: {len(result.passed_signals)} | 过滤: {len(result.filtered_signals)}")
        
        if result.filter_stats:
            lines.append("")
            lines.append("**过滤原因分布：**")
            for reason, count in sorted(result.filter_stats.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"- {reason}: {count}只")
        
        if result.passed_signals:
            lines.append("")
            lines.append("**通过信号：**")
            for i, sig in enumerate(result.passed_signals[:5], 1):
                code = getattr(sig, 'code', '?')
                name = getattr(sig, 'name', '?')
                score = getattr(sig, 'score', 0)
                sector = getattr(sig, 'sector', '')
                lines.append(f"{i}. {code} {name} | 评分:{score:.0f} | 行业:{sector}")
        
        return "\n".join(lines)


class SectorAllocator:
    """
    行业配置器
    
    功能：
    - 行业敞口限制
    - 行业权重动态分配
    - 行业轮动适配
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'max_sector_weight': 0.30,
            'min_sector_weight': 0.05,
            'max_sectors': 5,
        }
    
    def allocate(self, signals: List, sector_map: Dict[str, str],
                target_total_weight: float = 1.0) -> Dict[str, float]:
        """
        分配行业权重
        
        Args:
            signals: 信号列表
            sector_map: 代码到行业映射
            target_total_weight: 目标总权重
            
        Returns:
            行业权重字典
        """
        sector_signals = defaultdict(list)
        
        for signal in signals:
            code = getattr(signal, 'code', str(signal))
            sector = sector_map.get(code, getattr(signal, 'sector', '其他'))
            score = getattr(signal, 'score', 0)
            sector_signals[sector].append(score)
        
        sector_weights = {}
        
        for sector, scores in sector_signals.items():
            avg_score = np.mean(scores)
            weight = avg_score / 100.0
            weight = max(self.config['min_sector_weight'], 
                        min(self.config['max_sector_weight'], weight))
            sector_weights[sector] = weight
        
        total = sum(sector_weights.values())
        if total > 0:
            sector_weights = {k: v / total * target_total_weight 
                            for k, v in sector_weights.items()}
        
        sorted_sectors = sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_sectors) > self.config['max_sectors']:
            top_sectors = dict(sorted_sectors[:self.config['max_sectors']])
            other_weight = sum(v for k, v in sorted_sectors[self.config['max_sectors']:])
            if '其他' in top_sectors:
                top_sectors['其他'] += other_weight
            else:
                top_sectors['其他'] = other_weight
            sector_weights = top_sectors
        
        return sector_weights
    
    def format_allocation_report(self, allocations: Dict[str, float]) -> str:
        """格式化分配报告"""
        lines = ["**行业配置报告**", ""]
        
        for sector, weight in sorted(allocations.items(), key=lambda x: x[1], reverse=True):
            pct = weight * 100
            bar = "█" * int(pct / 2)
            lines.append(f"{sector}: {pct:5.1f}% {bar}")
        
        return "\n".join(lines)


class PositionSizer:
    """
    仓位管理器
    
    功能：
    - 凯利公式仓位计算
    - 风险平价仓位
    - 波动率调整仓位
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'max_position_pct': 0.20,
            'min_position_pct': 0.05,
            'kelly_fraction': 0.5,
            'risk_per_trade_pct': 0.02,
        }
    
    def calculate_size(self, signal, account_value: float,
                      win_rate: float, avg_win: float, avg_loss: float,
                      volatility: float = 0.0) -> float:
        """
        计算仓位
        
        Args:
            signal: 信号
            account_value: 账户总值
            win_rate: 胜率
            avg_win: 平均盈利
            avg_loss: 平均亏损
            volatility: 波动率
            
        Returns:
            建议仓位金额
        """
        base_size = account_value * self.config['min_position_pct']
        max_size = account_value * self.config['max_position_pct']
        
        kelly = self._kelly_formula(win_rate, avg_win / avg_loss if avg_loss != 0 else 1)
        kelly_size = account_value * kelly * self.config['kelly_fraction']
        
        vol_size = base_size
        if volatility > 0:
            vol_multiplier = 0.15 / volatility if volatility > 0.05 else 1.0
            vol_size = base_size * vol_multiplier
        
        risk_size = account_value * self.config['risk_per_trade_pct']
        
        size = min(kelly_size, max_size, vol_size)
        size = max(size, base_size)
        
        return size
    
    def _kelly_formula(self, win_rate: float, win_loss_ratio: float) -> float:
        """凯利公式"""
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        return max(0, kelly)
    
    def format_size_report(self, positions: List[Tuple[str, float, float]]) -> str:
        """格式化仓位报告"""
        lines = ["**仓位分配报告**", ""]
        
        total_value = sum(p[1] for p in positions)
        
        for code, value, pct in positions:
            bar = "█" * int(pct * 20)
            lines.append(f"{code}: ¥{value:,.0f} ({pct*100:.1f}%) {bar}")
        
        lines.append("")
        lines.append(f"**总仓位**: ¥{total_value:,.0f} ({total_value/sum(p[1] for p in positions)*100:.1f}%)")
        
        return "\n".join(lines)

