"""
数据挖掘增强模块

实现深度数据价值挖掘：
- 多维度关联分析
- 智能数据融合
- 预测性指标生成
- 隐藏模式发现
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class DataInsight:
    """数据洞察"""
    insight_type: str      # trend, anomaly, correlation, prediction
    title: str
    description: str
    confidence: float      # 置信度 0-1
    data_points: int       # 数据点数量
    timestamp: str = ""


@dataclass
class FeatureCorrelation:
    """特征相关性"""
    feature1: str
    feature2: str
    correlation: float
    p_value: float = 0.0


@dataclass
class AnomalyDetection:
    """异常检测结果"""
    code: str
    feature: str
    value: float
    expected_range: Tuple[float, float]
    anomaly_score: float
    direction: str  # high/low


class DataMiningEngine:
    """
    数据挖掘引擎
    
    核心功能：
    - 特征相关性分析
    - 异常值检测
    - 趋势预测
    - 模式发现
    """
    
    def __init__(self):
        self._correlation_cache = {}
        self._anomaly_thresholds = {
            'price_change': (0.01, 0.15),
            'volume_change': (0.1, 0.5),
            'rsi': (20, 80),
            'atr': (0.01, 0.1),
        }
    
    def analyze_correlations(self, features: pd.DataFrame, 
                            target_feature: str) -> List[FeatureCorrelation]:
        """
        分析特征相关性
        
        Args:
            features: 特征DataFrame
            target_feature: 目标特征
            
        Returns:
            相关性列表（按相关性绝对值排序）
        """
        if target_feature not in features.columns:
            return []
        
        correlations = []
        target = features[target_feature]
        
        for col in features.columns:
            if col == target_feature:
                continue
            
            valid_mask = features[col].notna() & target.notna()
            if valid_mask.sum() < 10:
                continue
            
            corr = features[col][valid_mask].corr(target[valid_mask])
            
            if abs(corr) > 0.3:
                correlations.append(FeatureCorrelation(
                    feature1=col,
                    feature2=target_feature,
                    correlation=corr
                ))
        
        correlations.sort(key=lambda x: abs(x.correlation), reverse=True)
        
        return correlations
    
    def detect_anomalies(self, code: str, features: Dict[str, float]) -> List[AnomalyDetection]:
        """
        检测异常值
        
        Args:
            code: 股票代码
            features: 特征字典
            
        Returns:
            异常检测结果列表
        """
        anomalies = []
        
        for feature, value in features.items():
            if feature not in self._anomaly_thresholds:
                continue
            
            lower, upper = self._anomaly_thresholds[feature]
            
            if value < lower:
                score = (lower - value) / lower
                anomalies.append(AnomalyDetection(
                    code=code,
                    feature=feature,
                    value=value,
                    expected_range=(lower, upper),
                    anomaly_score=score,
                    direction='low'
                ))
            elif value > upper:
                score = (value - upper) / upper
                anomalies.append(AnomalyDetection(
                    code=code,
                    feature=feature,
                    value=value,
                    expected_range=(lower, upper),
                    anomaly_score=score,
                    direction='high'
                ))
        
        anomalies.sort(key=lambda x: x.anomaly_score, reverse=True)
        
        return anomalies
    
    def identify_patterns(self, historical_data: pd.DataFrame) -> List[DataInsight]:
        """
        识别数据模式
        
        Args:
            historical_data: 历史数据
            
        Returns:
            数据洞察列表
        """
        insights = []
        
        if len(historical_data) < 20:
            return insights
        
        insights.extend(self._detect_trend_patterns(historical_data))
        insights.extend(self._detect_volume_patterns(historical_data))
        insights.extend(self._detect_momentum_patterns(historical_data))
        
        return insights
    
    def _detect_trend_patterns(self, data: pd.DataFrame) -> List[DataInsight]:
        """检测趋势模式"""
        insights = []
        
        ma5 = data['close'].rolling(5).mean()
        ma20 = data['close'].rolling(20).mean()
        ma60 = data['close'].rolling(60).mean()
        
        latest_ma5 = ma5.iloc[-1]
        latest_ma20 = ma20.iloc[-1]
        latest_ma60 = ma60.iloc[-1]
        
        if latest_ma5 > latest_ma20 > latest_ma60:
            insights.append(DataInsight(
                insight_type='trend',
                title='多头排列',
                description='MA5 > MA20 > MA60，形成多头排列',
                confidence=0.85,
                data_points=60
            ))
        
        if latest_ma5 < latest_ma20 < latest_ma60:
            insights.append(DataInsight(
                insight_type='trend',
                title='空头排列',
                description='MA5 < MA20 < MA60，形成空头排列',
                confidence=0.85,
                data_points=60
            ))
        
        return insights
    
    def _detect_volume_patterns(self, data: pd.DataFrame) -> List[DataInsight]:
        """检测成交量模式"""
        insights = []
        
        volume_ma20 = data['volume'].rolling(20).mean()
        latest_volume = data['volume'].iloc[-1]
        avg_volume = volume_ma20.iloc[-1]
        
        volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 0
        
        if volume_ratio > 2.0:
            insights.append(DataInsight(
                insight_type='anomaly',
                title='放量异动',
                description=f'成交量较20日均量放大{volume_ratio:.1f}倍',
                confidence=0.9,
                data_points=20
            ))
        
        if volume_ratio < 0.3:
            insights.append(DataInsight(
                insight_type='anomaly',
                title='极致缩量',
                description=f'成交量萎缩至20日均量的{volume_ratio*100:.0f}%',
                confidence=0.85,
                data_points=20
            ))
        
        return insights
    
    def _detect_momentum_patterns(self, data: pd.DataFrame) -> List[DataInsight]:
        """检测动量模式"""
        insights = []
        
        if len(data) < 14:
            return insights
        
        delta = data['close'].diff(1)
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        latest_rsi = rsi.iloc[-1]
        
        if latest_rsi > 70:
            insights.append(DataInsight(
                insight_type='momentum',
                title='超买信号',
                description=f'RSI={latest_rsi:.1f}，进入超买区域',
                confidence=0.8,
                data_points=14
            ))
        
        if latest_rsi < 30:
            insights.append(DataInsight(
                insight_type='momentum',
                title='超卖信号',
                description=f'RSI={latest_rsi:.1f}，进入超卖区域',
                confidence=0.8,
                data_points=14
            ))
        
        return insights
    
    def generate_predictive_features(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        生成预测性特征
        
        Args:
            historical_data: 历史数据
            
        Returns:
            预测性特征字典
        """
        features = {}
        
        if len(historical_data) < 20:
            return features
        
        prices = historical_data['close']
        volumes = historical_data['volume']
        
        features['volatility_20d'] = prices.pct_change().rolling(20).std().iloc[-1]
        features['volume_trend'] = volumes.rolling(5).mean().iloc[-1] / volumes.rolling(20).mean().iloc[-1]
        features['price_momentum'] = (prices.iloc[-1] - prices.iloc[-20]) / prices.iloc[-20]
        features['avg_true_range'] = self._calculate_atr(historical_data)
        features['bollinger_width'] = self._calculate_bollinger_width(historical_data)
        
        return features
    
    def _calculate_atr(self, data: pd.DataFrame) -> float:
        """计算ATR"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        return atr / close.iloc[-1]
    
    def _calculate_bollinger_width(self, data: pd.DataFrame) -> float:
        """计算布林带宽度"""
        ma20 = data['close'].rolling(20).mean()
        std20 = data['close'].rolling(20).std()
        
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        
        width = (upper.iloc[-1] - lower.iloc[-1]) / ma20.iloc[-1]
        
        return width
    
    def format_insights_report(self, insights: List[DataInsight]) -> str:
        """格式化洞察报告"""
        if not insights:
            return "暂无数据洞察"
        
        lines = ["**💡 数据洞察报告**", ""]
        
        insights.sort(key=lambda x: x.confidence, reverse=True)
        
        for insight in insights:
            confidence_bar = "█" * int(insight.confidence * 20)
            lines.append(f"{'🔥' if insight.confidence > 0.8 else '💡'} **{insight.title}**")
            lines.append(f"   └ {insight.description}")
            lines.append(f"     置信度: {insight.confidence*100:.0f}% {confidence_bar}")
            lines.append("")
        
        return "\n".join(lines)


class SmartDataFusion:
    """
    智能数据融合器
    
    功能：
    - 多源数据融合
    - 数据质量加权
    - 冲突解决
    - 数据清洗与标准化
    """
    
    def __init__(self):
        self._source_weights = {
            'akshare': 1.0,
            'tushare': 0.9,
            'yfinance': 0.95,
            'sina': 0.7,
            'baostock': 0.85,
        }
    
    def fuse_data(self, responses: List[Dict]) -> Dict:
        """
        融合多个数据源的数据
        
        Args:
            responses: 多个数据源响应
            
        Returns:
            融合后的统一数据
        """
        if not responses:
            return {}
        
        fused = {}
        
        for response in responses:
            source = response.get('source', 'unknown')
            data = response.get('data', {})
            weight = self._source_weights.get(source, 0.5)
            
            for key, value in data.items():
                if key not in fused:
                    fused[key] = []
                
                fused[key].append((value, weight))
        
        for key, weighted_values in fused.items():
            if all(isinstance(v[0], (int, float)) for v in weighted_values):
                total_weight = sum(w for _, w in weighted_values)
                if total_weight > 0:
                    fused[key] = sum(v * w for v, w in weighted_values) / total_weight
            else:
                fused[key] = weighted_values[0][0]
        
        return fused
    
    def resolve_conflicts(self, data1: Dict, data2: Dict, threshold: float = 0.1) -> Dict:
        """
        解决数据冲突
        
        Args:
            data1: 数据源1
            data2: 数据源2
            threshold: 差异阈值
            
        Returns:
            解决冲突后的数据
        """
        resolved = {}
        
        all_keys = set(data1.keys()) | set(data2.keys())
        
        for key in all_keys:
            v1 = data1.get(key)
            v2 = data2.get(key)
            
            if v1 is None:
                resolved[key] = v2
            elif v2 is None:
                resolved[key] = v1
            elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                if v1 == 0:
                    resolved[key] = v2
                elif abs(v1 - v2) / max(abs(v1), abs(v2)) > threshold:
                    resolved[key] = (v1 + v2) / 2
                else:
                    resolved[key] = v1
            else:
                resolved[key] = v1
        
        return resolved
    
    def standardize_data(self, data: Dict) -> Dict:
        """标准化数据格式"""
        standard_mapping = {
            'trade_date': 'date',
            'stock_code': 'code',
            'stock_name': 'name',
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'close_price': 'close',
            'trade_volume': 'volume',
            'turnover': 'amount',
        }
        
        standardized = {}
        for old_key, new_key in standard_mapping.items():
            if old_key in data:
                standardized[new_key] = data[old_key]
        
        standardized.update({k: v for k, v in data.items() if k not in standard_mapping})
        
        return standardized

