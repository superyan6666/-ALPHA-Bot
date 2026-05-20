"""
增量更新机制模块

实现事件驱动的增量更新：
- 监听市场事件
- 仅更新变化的数据
- 支持订阅/发布模式
- 增量特征计算
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Set
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib

log = logging.getLogger(__name__)


@dataclass
class MarketEvent:
    """市场事件"""
    event_type: str  # PRICE_CHANGE, VOLUME_SURGE, NEWS, EARNINGS, etc.
    code: str
    timestamp: str
    data: Dict = field(default_factory=dict)
    priority: int = 5


@dataclass
class Subscription:
    """订阅配置"""
    event_types: List[str]
    codes: Optional[Set[str]] = None
    callback: Callable = None
    last_triggered: str = ""


class EventManager:
    """
    事件管理器
    
    功能：
    - 事件发布/订阅
    - 事件过滤
    - 优先级调度
    """
    
    def __init__(self):
        self.subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self.event_queue: List[MarketEvent] = []
        self.processed_events: Set[str] = set()
    
    def subscribe(self, event_type: str, callback: Callable,
                 codes: Optional[Set[str]] = None):
        """
        订阅事件类型
        
        Args:
            event_type: 事件类型
            callback: 回调函数
            codes: 关注的股票代码集合（None表示全部）
        """
        subscription = Subscription(
            event_types=[event_type],
            codes=codes,
            callback=callback
        )
        self.subscriptions[event_type].append(subscription)
        log.info(f"✅ 订阅事件: {event_type} (代码数: {len(codes) if codes else '全部'})")
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """取消订阅"""
        if event_type in self.subscriptions:
            self.subscriptions[event_type] = [
                s for s in self.subscriptions[event_type]
                if s.callback != callback
            ]
    
    def publish(self, event: MarketEvent):
        """发布事件"""
        event_hash = self._hash_event(event)
        if event_hash in self.processed_events:
            return
        
        self.processed_events.add(event_hash)
        self.event_queue.append(event)
        
        if len(self.processed_events) > 10000:
            self.processed_events = set(list(self.processed_events)[-5000:])
    
    def _hash_event(self, event: MarketEvent) -> str:
        """生成事件哈希用于去重"""
        event_str = f"{event.event_type}_{event.code}_{event.timestamp}"
        return hashlib.md5(event_str.encode()).hexdigest()
    
    def process_events(self, max_events: int = 100):
        """
        处理事件队列
        
        Args:
            max_events: 单次最大处理事件数
        """
        self.event_queue.sort(key=lambda x: x.priority)
        
        processed = 0
        while self.event_queue and processed < max_events:
            event = self.event_queue.pop(0)
            
            if event.event_type in self.subscriptions:
                for subscription in self.subscriptions[event.event_type]:
                    if subscription.codes is None or event.code in subscription.codes:
                        try:
                            subscription.callback(event)
                            subscription.last_triggered = event.timestamp
                        except Exception as e:
                            log.error(f"事件处理失败 {event.event_type}: {e}")
            
            processed += 1
    
    def clear_queue(self):
        """清空事件队列"""
        self.event_queue.clear()
    
    def get_event_count(self) -> int:
        """获取队列中事件数"""
        return len(self.event_queue)


class IncrementalUpdater:
    """
    增量更新器
    
    功能：
    - 检测数据变化
    - 增量特征计算
    - 缓存管理
    """
    
    def __init__(self, data_lake):
        self.data_lake = data_lake
        self.last_update = {}
        self.change_thresholds = {
            'price': 0.01,      # 价格变化1%触发更新
            'volume': 0.3,      # 成交量变化30%触发
            'score': 5,         # 评分变化5分触发
        }
    
    def check_changes(self, code: str, current_data: Dict) -> Optional[Dict]:
        """
        检查数据变化
        
        Args:
            code: 股票代码
            current_data: 当前数据
            
        Returns:
            变化字段字典，如果无变化返回None
        """
        if code not in self.last_update:
            self.last_update[code] = current_data
            return current_data
        
        changes = {}
        last_data = self.last_update[code]
        
        for key, threshold in self.change_thresholds.items():
            current = current_data.get(key)
            last = last_data.get(key)
            
            if current is None or last is None:
                continue
            
            if isinstance(current, (int, float)) and isinstance(last, (int, float)):
                if last != 0:
                    diff = abs(current - last) / abs(last)
                    if diff >= threshold:
                        changes[key] = {'old': last, 'new': current, 'change_pct': diff}
                else:
                    if current != 0:
                        changes[key] = {'old': last, 'new': current, 'change_pct': float('inf')}
        
        if changes:
            self.last_update[code] = current_data
        
        return changes if changes else None
    
    def update_features_incrementally(self, code: str, changes: Dict,
                                     feature_cache: Dict) -> Dict:
        """
        增量更新特征
        
        Args:
            code: 股票代码
            changes: 变化字段
            feature_cache: 特征缓存
            
        Returns:
            更新后的特征字典
        """
        features = feature_cache.get(code, {})
        
        if 'price' in changes:
            features = self._update_price_features(features, changes['price'])
        
        if 'volume' in changes:
            features = self._update_volume_features(features, changes['volume'])
        
        feature_cache[code] = features
        return features
    
    def _update_price_features(self, features: Dict, price_change: Dict) -> Dict:
        """更新价格相关特征"""
        features['last_price_change'] = price_change['change_pct']
        features['price_change_direction'] = 'UP' if price_change['new'] > price_change['old'] else 'DOWN'
        features['last_price_update'] = datetime.now().isoformat()
        return features
    
    def _update_volume_features(self, features: Dict, volume_change: Dict) -> Dict:
        """更新成交量相关特征"""
        features['volume_surge'] = volume_change['change_pct'] > 0.5
        features['last_volume_update'] = datetime.now().isoformat()
        return features
    
    def cleanup_old_updates(self, days: int = 7):
        """清理过期的更新记录"""
        cutoff = datetime.now() - timedelta(days=days)
        self.last_update = {
            code: data for code, data in self.last_update.items()
            if datetime.fromisoformat(data.get('timestamp', '2000-01-01T00:00:00')) > cutoff
        }


class DataChangeDetector:
    """
    数据变化检测器
    
    功能：
    - 监控数据源变化
    - 触发增量更新
    - 管理更新频率
    """
    
    def __init__(self, update_interval_seconds: int = 60):
        self.update_interval = update_interval_seconds
        self.last_check = {}
        self.change_listeners = []
    
    def add_listener(self, listener: Callable):
        """添加变化监听器"""
        self.change_listeners.append(listener)
    
    def remove_listener(self, listener: Callable):
        """移除变化监听器"""
        self.change_listeners.remove(listener)
    
    def check_for_changes(self, data_source: str, current_hash: str) -> bool:
        """
        检查数据源是否变化
        
        Args:
            data_source: 数据源名称
            current_hash: 当前数据哈希
            
        Returns:
            是否发生变化
        """
        now = datetime.now()
        
        if data_source in self.last_check:
            last_time, last_hash = self.last_check[data_source]
            
            time_diff = (now - last_time).total_seconds()
            if time_diff < self.update_interval:
                return False
            
            if last_hash == current_hash:
                self.last_check[data_source] = (now, current_hash)
                return False
        
        self.last_check[data_source] = (now, current_hash)
        
        for listener in self.change_listeners:
            try:
                listener(data_source, current_hash)
            except Exception as e:
                log.error(f"变化通知失败: {e}")
        
        return True
    
    def calculate_data_hash(self, data: Dict) -> str:
        """计算数据哈希"""
        data_str = str(sorted(data.items()))
        return hashlib.md5(data_str.encode()).hexdigest()


class PipelineEventAdapter:
    """
    流水线事件适配器
    
    功能：
    - 将流水线输出转换为事件
    - 事件去重
    - 事件优先级排序
    """
    
    def __init__(self, event_manager: EventManager):
        self.event_manager = event_manager
    
    def signals_to_events(self, signals: List) -> List[MarketEvent]:
        """
        将信号转换为事件
        
        Args:
            signals: 信号列表
            
        Returns:
            事件列表
        """
        events = []
        
        for signal in signals:
            code = getattr(signal, 'code', str(signal))
            score = getattr(signal, 'score', 0)
            pct_chg = getattr(signal, 'pct_chg', 0)
            
            priority = 1 if score >= 80 else 3 if score >= 70 else 5
            
            event = MarketEvent(
                event_type='SIGNAL_GENERATED',
                code=code,
                timestamp=datetime.now().isoformat(),
                data={
                    'score': score,
                    'pct_chg': pct_chg,
                    'signal_type': 'BUY' if score >= 70 else 'WATCH'
                },
                priority=priority
            )
            
            events.append(event)
        
        return events
    
    def publish_signals(self, signals: List):
        """发布信号事件"""
        events = self.signals_to_events(signals)
        for event in events:
            self.event_manager.publish(event)

