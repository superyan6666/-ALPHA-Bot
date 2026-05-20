"""
市场状态识别模块

提供多维度市场状态分析：
- 多状态分类（BULL/BEAR/PANIC/NEUTRAL）
- 模糊状态（概率分布）
- 趋势强度评估
- 流动性评估
- 状态持续性预测
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class MarketRegime(Enum):
    """市场状态枚举"""
    BULL = "BULL"        # 牛市/上升趋势
    NEUTRAL = "NEUTRAL"  # 中性/震荡
    BEAR = "BEAR"        # 熊市/下降趋势
    PANIC = "PANIC"      # 恐慌/极端下跌


@dataclass
class RegimeProbabilities:
    """状态概率分布"""
    bull: float = 0.25
    neutral: float = 0.25
    bear: float = 0.25
    panic: float = 0.25
    
    def dominant(self) -> MarketRegime:
        """获取最可能的状态"""
        probs = {
            MarketRegime.BULL: self.bull,
            MarketRegime.NEUTRAL: self.neutral,
            MarketRegime.BEAR: self.bear,
            MarketRegime.PANIC: self.panic,
        }
        return max(probs, key=probs.get)
    
    def confidence(self) -> float:
        """获取判断置信度"""
        return max(self.bull, self.neutral, self.bear, self.panic) - 0.25
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'BULL': self.bull,
            'NEUTRAL': self.neutral,
            'BEAR': self.bear,
            'PANIC': self.panic,
        }


@dataclass
class MarketState:
    """市场状态"""
    regime: MarketRegime = MarketRegime.NEUTRAL
    regime_prob: RegimeProbabilities = field(default_factory=RegimeProbabilities)
    regime_confidence: float = 0.0
    
    volatility: float = 0.0       # 波动率（ATR/价格）
    volatility_level: str = "NORMAL"  # HIGH/NORMAL/LOW
    
    trend_strength: float = 0.0   # 趋势强度（ADX）
    trend_direction: str = "SIDEWAYS"  # UP/DOWN/SIDEWAYS
    
    liquidity: float = 1.0        # 流动性指标
    liquidity_level: str = "NORMAL"  # HIGH/NORMAL/LOW
    
    momentum: float = 0.0         # 动量（RSI类指标）
    momentum_level: str = "NEUTRAL"  # STRONG/WEAK/NEUTRAL
    
    breadcrumb: List[str] = field(default_factory=list)
    last_updated: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'regime': self.regime.value,
            'regime_confidence': self.regime_confidence,
            'volatility': self.volatility,
            'volatility_level': self.volatility_level,
            'trend_strength': self.trend_strength,
            'trend_direction': self.trend_direction,
            'liquidity': self.liquidity,
            'liquidity_level': self.liquidity_level,
            'momentum': self.momentum,
            'momentum_level': self.momentum_level,
            'breadcrumb': self.breadcrumb,
        }
    
    def is_bullish(self) -> bool:
        return self.regime in (MarketRegime.BULL,) and self.regime_confidence > 0.4
    
    def is_bearish(self) -> bool:
        return self.regime in (MarketRegime.BEAR, MarketRegime.PANIC) and self.regime_confidence > 0.4
    
    def is_stable(self) -> bool:
        return self.volatility_level == "LOW" and self.trend_direction == "SIDEWAYS"
    
    def get_factor_weights(self) -> Dict[str, float]:
        """获取推荐因子权重"""
        weights = {
            'f_val': 1.0,
            'f_mom': 1.0,
            'f_rev': 1.0,
            'f_risk': 1.0,
        }
        
        if self.regime == MarketRegime.BULL:
            weights.update({'f_mom': 1.3, 'f_val': 0.8, 'f_risk': 0.8})
        elif self.regime == MarketRegime.BEAR:
            weights.update({'f_val': 1.3, 'f_rev': 1.2, 'f_mom': 0.6, 'f_risk': 1.5})
        elif self.regime == MarketRegime.PANIC:
            weights.update({'f_rev': 1.5, 'f_val': 1.2, 'f_mom': 0.3, 'f_risk': 1.5})
        else:
            weights.update({'f_val': 1.0, 'f_mom': 1.0, 'f_rev': 1.0, 'f_risk': 1.0})
        
        if self.volatility_level == "HIGH":
            weights['f_rev'] *= 1.2
            weights['f_mom'] *= 0.8
        
        return weights


class MarketStateDetector:
    """
    市场状态检测器
    
    功能：
    - 多指标综合判断
    - 模糊状态概率
    - 趋势强度评估
    - 状态历史追踪
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.state_history: List[MarketState] = []
    
    def _default_config(self) -> Dict:
        return {
            'bull_threshold': 0.60,
            'bear_threshold': 0.60,
            'panic_threshold': 0.70,
            'high_vol_threshold': 0.03,
            'low_vol_threshold': 0.01,
            'strong_trend_threshold': 25,
            'weak_trend_threshold': 15,
            'momentum_strong': 65,
            'momentum_weak': 40,
        }
    
    def detect(self, market_data: Dict) -> MarketState:
        """
        检测市场状态
        
        Args:
            market_data: 市场数据字典，包含：
                - index_close: 指数收盘价
                - index_high: 指数最高价
                - index_low: 指数最低价
                - volume: 成交量
                - atr: ATR值
                - adx: ADX值
                - rsi: RSI值
                - vix: VIX值（可选）
                
        Returns:
            MarketState
        """
        from datetime import datetime
        
        close = market_data.get('index_close', 0)
        high = market_data.get('index_high', close)
        low = market_data.get('index_low', close)
        volume = market_data.get('volume', 0)
        atr = market_data.get('atr', 0)
        adx = market_data.get('adx', 20)
        rsi = market_data.get('rsi', 50)
        vix = market_data.get('vix', 20)
        
        volatility = atr / close if close > 0 else 0.0
        vol_level = self._classify_volatility(volatility)
        
        trend_strength = float(adx)
        trend_dir = self._classify_trend(trend_strength, market_data)
        
        momentum = float(rsi)
        mom_level = self._classify_momentum(momentum)
        
        regime_prob = self._calculate_regime_prob(
            close, high, low, volume, volatility, 
            trend_strength, momentum, vix, market_data
        )
        
        regime = regime_prob.dominant()
        confidence = regime_prob.confidence()
        
        liquidity = self._calculate_liquidity(volume, market_data)
        liq_level = self._classify_liquidity(liquidity)
        
        state = MarketState(
            regime=regime,
            regime_prob=regime_prob,
            regime_confidence=confidence,
            volatility=volatility,
            volatility_level=vol_level,
            trend_strength=trend_strength,
            trend_direction=trend_dir,
            liquidity=liquidity,
            liquidity_level=liq_level,
            momentum=momentum,
            momentum_level=mom_level,
            last_updated=datetime.now().isoformat()
        )
        
        self.state_history.append(state)
        if len(self.state_history) > 60:
            self.state_history = self.state_history[-60:]
        
        return state
    
    def _calculate_regime_prob(self, close: float, high: float, low: float,
                             volume: float, volatility: float,
                             trend_strength: float, momentum: float,
                             vix: float, market_data: Dict) -> RegimeProbabilities:
        """计算状态概率"""
        scores = {
            MarketRegime.BULL: 0.0,
            MarketRegime.NEUTRAL: 0.0,
            MarketRegime.BEAR: 0.0,
            MarketRegime.PANIC: 0.0,
        }
        
        ma5 = market_data.get('ma5', close)
        ma20 = market_data.get('ma20', close)
        ma60 = market_data.get('ma60', close)
        
        if ma5 > ma20 > ma60:
            scores[MarketRegime.BULL] += 0.2
        elif ma5 < ma20 < ma60:
            scores[MarketRegime.BEAR] += 0.2
        
        if trend_strength > self.config['strong_trend_threshold']:
            if momentum > 50:
                scores[MarketRegime.BULL] += 0.3
            else:
                scores[MarketRegime.BEAR] += 0.3
        elif trend_strength < self.config['weak_trend_threshold']:
            scores[MarketRegime.NEUTRAL] += 0.2
        
        if momentum > self.config['momentum_strong']:
            scores[MarketRegime.BULL] += 0.2
        elif momentum < self.config['momentum_weak']:
            scores[MarketRegime.BEAR] += 0.15
            if volatility > self.config['high_vol_threshold']:
                scores[MarketRegime.PANIC] += 0.15
        
        if vix > 25:
            scores[MarketRegime.BEAR] += 0.15
            scores[MarketRegime.PANIC] += 0.1
        elif vix < 15:
            scores[MarketRegime.BULL] += 0.1
        
        if volatility > self.config['high_vol_threshold']:
            scores[MarketRegime.PANIC] += 0.2
            scores[MarketRegime.BEAR] += 0.1
        elif volatility < self.config['low_vol_threshold']:
            scores[MarketRegime.NEUTRAL] += 0.15
        
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        base = 0.25
        scores = {k: base + 0.75 * (v / max(sum(scores.values()), 0.01)) for k, v in scores.items()}
        
        total = sum(scores.values())
        return RegimeProbabilities(
            bull=scores[MarketRegime.BULL] / total,
            neutral=scores[MarketRegime.NEUTRAL] / total,
            bear=scores[MarketRegime.BEAR] / total,
            panic=scores[MarketRegime.PANIC] / total,
        )
    
    def _classify_volatility(self, volatility: float) -> str:
        if volatility > self.config['high_vol_threshold']:
            return "HIGH"
        elif volatility < self.config['low_vol_threshold']:
            return "LOW"
        return "NORMAL"
    
    def _classify_trend(self, trend_strength: float, market_data: Dict) -> str:
        if trend_strength > self.config['strong_trend_threshold']:
            ma5 = market_data.get('ma5', 0)
            ma20 = market_data.get('ma20', 0)
            if ma5 > ma20:
                return "UP"
            elif ma5 < ma20:
                return "DOWN"
        return "SIDEWAYS"
    
    def _classify_momentum(self, momentum: float) -> str:
        if momentum > self.config['momentum_strong']:
            return "STRONG"
        elif momentum < self.config['momentum_weak']:
            return "WEAK"
        return "NEUTRAL"
    
    def _calculate_liquidity(self, volume: float, market_data: Dict) -> float:
        vol_ma20 = market_data.get('volume_ma20', volume)
        if vol_ma20 > 0:
            return volume / vol_ma20
        return 1.0
    
    def _classify_liquidity(self, liquidity: float) -> str:
        if liquidity > 1.5:
            return "HIGH"
        elif liquidity < 0.7:
            return "LOW"
        return "NORMAL"
    
    def get_regime_string(self, state: MarketState) -> str:
        """获取状态描述字符串"""
        regime_names = {
            MarketRegime.BULL: "上涨趋势",
            MarketRegime.NEUTRAL: "震荡整理",
            MarketRegime.BEAR: "下跌趋势",
            MarketRegime.PANIC: "恐慌抛售",
        }
        
        conf_level = "高" if state.regime_confidence > 0.5 else "中" if state.regime_confidence > 0.3 else "低"
        
        parts = [
            f"{regime_names[state.regime]}(置信{conf_level})",
            f"波动{state.volatility_level}",
            f"趋势{state.trend_direction}",
            f"动量{state.momentum_level}",
        ]
        
        return " | ".join(parts)
    
    def predict_next_state(self, horizon: int = 1) -> Optional[MarketState]:
        """
        预测下一状态（简单马尔可夫链）
        
        Args:
            horizon: 预测步数
            
        Returns:
            预测的状态
        """
        if len(self.state_history) < 5:
            return None
        
        transitions = {}
        for i in range(len(self.state_history) - 1):
            curr = self.state_history[i].regime
            next_s = self.state_history[i + 1].regime
            key = (curr, next_s)
            transitions[key] = transitions.get(key, 0) + 1
        
        current = self.state_history[-1].regime
        
        next_counts = {r: 0 for r in MarketRegime}
        for (c, n), count in transitions.items():
            if c == current:
                next_counts[n] += 1
        
        total = sum(next_counts.values())
        if total == 0:
            return None
        
        probs = RegimeProbabilities(
            bull=next_counts[MarketRegime.BULL] / total,
            neutral=next_counts[MarketRegime.NEUTRAL] / total,
            bear=next_counts[MarketRegime.BEAR] / total,
            panic=next_counts[MarketRegime.PANIC] / total,
        )
        
        last_state = self.state_history[-1]
        return MarketState(
            regime=probs.dominant(),
            regime_prob=probs,
            regime_confidence=probs.confidence(),
            volatility=last_state.volatility,
            volatility_level=last_state.volatility_level,
            trend_strength=last_state.trend_strength,
            trend_direction=last_state.trend_direction,
            liquidity=last_state.liquidity,
            liquidity_level=last_state.liquidity_level,
            momentum=last_state.momentum,
            momentum_level=last_state.momentum_level,
            breadcrumb=["预测"],
        )
    
    def format_state_report(self, state: MarketState) -> str:
        """格式化状态报告"""
        lines = ["**市场状态报告**", ""]
        
        prob_str = " | ".join([
            f"{k.value}:{v:.0%}" 
            for k, v in state.regime_prob.to_dict().items()
        ])
        lines.append(f"**状态概率**: {prob_str}")
        lines.append(f"**主状态**: {state.regime.value} (置信{state.regime_confidence:.0%})")
        lines.append("")
        lines.append(f"- 波动率: {state.volatility*100:.2f}% ({state.volatility_level})")
        lines.append(f"- 趋势强度: {state.trend_strength:.1f} ({state.trend_direction})")
        lines.append(f"- 流动性: {state.liquidity:.2f}x ({state.liquidity_level})")
        lines.append(f"- 动量: {state.momentum:.1f} ({state.momentum_level})")
        
        weights = state.get_factor_weights()
        lines.append("")
        lines.append("**推荐因子权重**:")
        for name, weight in weights.items():
            lines.append(f"- {name}: {weight:.1f}")
        
        return "\n".join(lines)

