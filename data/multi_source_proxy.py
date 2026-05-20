"""
多源数据代理模块

实现数据源故障转移和降级机制：
- 主备数据源切换
- 自动故障检测
- 数据质量评分
- 熔断机制
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import time

log = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """数据源配置"""
    name: str
    priority: int = 100      # 优先级，数字越小越优先
    weight: float = 1.0      # 权重
    timeout: int = 30        # 超时时间(秒)
    max_retries: int = 3     # 最大重试次数
    enabled: bool = True     # 是否启用
    health_check_url: str = ""


@dataclass
class DataSourceStatus:
    """数据源状态"""
    name: str
    healthy: bool = True
    last_success: str = ""
    last_failure: str = ""
    failure_count: int = 0
    latency_ms: float = 0.0
    availability: float = 1.0


@dataclass
class DataRequest:
    """数据请求"""
    data_type: str           # kline, quote, fund_flow, etc.
    code: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    params: Dict = field(default_factory=dict)


@dataclass
class DataResponse:
    """数据响应"""
    success: bool
    data: Optional[Dict] = None
    source: str = ""
    latency_ms: float = 0.0
    error: str = ""


class MultiSourceDataProxy:
    """
    多源数据代理
    
    核心功能：
    - 自动选择最优数据源
    - 故障自动降级
    - 熔断保护
    - 数据质量监控
    """
    
    def __init__(self):
        self._sources: Dict[str, DataSourceConfig] = {}
        self._source_status: Dict[str, DataSourceStatus] = {}
        self._source_implementations: Dict[str, Callable] = {}
        self._circuit_breakers: Dict[str, float] = {}  # 熔断时间戳
        self._request_history: List[Tuple[str, str, bool, float]] = []
        
        self._register_default_sources()
    
    def _register_default_sources(self):
        """注册默认数据源"""
        self.register_source(
            name="akshare",
            priority=10,
            weight=1.0,
            timeout=30
        )
        self.register_source(
            name="tushare",
            priority=20,
            weight=0.8,
            timeout=30
        )
        self.register_source(
            name="yfinance",
            priority=15,
            weight=0.9,
            timeout=45
        )
        self.register_source(
            name="sina",
            priority=30,
            weight=0.7,
            timeout=15
        )
        self.register_source(
            name="baostock",
            priority=25,
            weight=0.8,
            timeout=20
        )
        self.register_source(
            name="efinance",
            priority=35,
            weight=0.7,
            timeout=20
        )
        
        log.info("✅ 默认数据源注册完成")
    
    def register_source(self, name: str, **kwargs):
        """注册数据源"""
        config = DataSourceConfig(name=name, **kwargs)
        self._sources[name] = config
        self._source_status[name] = DataSourceStatus(name=name)
        self._circuit_breakers[name] = 0.0
        log.info(f"📥 注册数据源: {name} (优先级: {config.priority})")
    
    def register_implementation(self, name: str, func: Callable):
        """注册数据源实现"""
        self._source_implementations[name] = func
        log.info(f"🔧 注册数据源实现: {name}")
    
    def _is_source_available(self, name: str) -> bool:
        """检查数据源是否可用"""
        config = self._sources.get(name)
        if not config or not config.enabled:
            return False
        
        if self._circuit_breakers.get(name, 0) > time.time():
            return False
        
        return self._source_status[name].healthy
    
    def _select_best_source(self, data_type: str) -> List[str]:
        """选择最优数据源列表"""
        available = []
        
        for name, config in self._sources.items():
            if self._is_source_available(name):
                available.append((name, config.priority, config.weight))
        
        available.sort(key=lambda x: (x[1], -x[2]))
        
        return [name for name, _, _ in available]
    
    def _execute_request(self, source_name: str, request: DataRequest) -> DataResponse:
        """执行单个数据源请求"""
        start_time = time.time()
        
        try:
            impl = self._source_implementations.get(source_name)
            if not impl:
                return DataResponse(
                    success=False,
                    source=source_name,
                    error=f"未实现数据源: {source_name}"
                )
            
            data = impl(request)
            
            latency_ms = (time.time() - start_time) * 1000
            
            self._update_source_status(source_name, success=True, latency_ms=latency_ms)
            
            return DataResponse(
                success=True,
                data=data,
                source=source_name,
                latency_ms=latency_ms
            )
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            self._update_source_status(source_name, success=False, latency_ms=latency_ms)
            
            return DataResponse(
                success=False,
                source=source_name,
                latency_ms=latency_ms,
                error=str(e)
            )
    
    def _update_source_status(self, name: str, success: bool, latency_ms: float):
        """更新数据源状态"""
        status = self._source_status[name]
        now = datetime.now().isoformat()
        
        if success:
            status.healthy = True
            status.last_success = now
            status.failure_count = 0
            status.latency_ms = latency_ms
            status.availability = min(1.0, status.availability + 0.1)
            self._circuit_breakers[name] = 0.0
        else:
            status.last_failure = now
            status.failure_count += 1
            status.availability = max(0.0, status.availability - 0.05)
            
            if status.failure_count >= 5:
                status.healthy = False
                self._circuit_breakers[name] = time.time() + 300
                log.warning(f"🔴 数据源 {name} 熔断，5分钟后恢复")
    
    def get_data(self, request: DataRequest) -> DataResponse:
        """
        获取数据（自动选择最优数据源并降级）
        
        Args:
            request: 数据请求
            
        Returns:
            数据响应
        """
        sources = self._select_best_source(request.data_type)
        
        if not sources:
            return DataResponse(
                success=False,
                error="所有数据源不可用"
            )
        
        for i, source_name in enumerate(sources):
            response = self._execute_request(source_name, request)
            
            if response.success:
                if i > 0:
                    log.info(f"🔄 降级到备用数据源: {source_name}")
                return response
            
            log.warning(f"⚠️ 数据源 {source_name} 失败: {response.error}")
            
            if i < len(sources) - 1:
                log.info(f"🔄 尝试下一个数据源: {sources[i+1]}")
        
        return DataResponse(
            success=False,
            error=f"所有数据源均失败，尝试了: {sources}"
        )
    
    def batch_get_data(self, requests: List[DataRequest]) -> List[DataResponse]:
        """批量获取数据"""
        responses = []
        for req in requests:
            responses.append(self.get_data(req))
        return responses
    
    def get_source_status(self) -> Dict[str, DataSourceStatus]:
        """获取所有数据源状态"""
        return self._source_status
    
    def format_status_report(self) -> str:
        """格式化状态报告"""
        lines = ["**📡 数据源状态报告**", ""]
        
        sources = sorted(self._sources.values(), key=lambda x: x.priority)
        
        for source in sources:
            status = self._source_status[source.name]
            health = "✅" if status.healthy else "❌"
            cb_status = "🔴熔断中" if self._circuit_breakers.get(source.name, 0) > time.time() else ""
            
            lines.append(f"{health} **{source.name}**")
            lines.append(f"   └ 优先级: {source.priority} | 可用性: {status.availability*100:.1f}% | 延迟: {status.latency_ms:.0f}ms {cb_status}")
        
        return "\n".join(lines)


class DataQualityChecker:
    """
    数据质量检查器
    
    功能：
    - 数据完整性检查
    - 数据一致性验证
    - 异常值检测
    """
    
    def __init__(self):
        self._rules = {
            'kline': self._check_kline_quality,
            'quote': self._check_quote_quality,
            'fund_flow': self._check_fund_flow_quality,
        }
    
    def _check_kline_quality(self, data: Dict) -> Tuple[bool, str]:
        """检查K线数据质量"""
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        
        for field in required_fields:
            if field not in data:
                return False, f"缺少字段: {field}"
        
        if data.get('high', 0) < data.get('low', 1):
            return False, "high < low 无效"
        
        if data.get('volume', 0) < 0:
            return False, "成交量为负"
        
        return True, "OK"
    
    def _check_quote_quality(self, data: Dict) -> Tuple[bool, str]:
        """检查报价数据质量"""
        required_fields = ['code', 'name', 'price']
        
        for field in required_fields:
            if field not in data:
                return False, f"缺少字段: {field}"
        
        if data.get('price', 0) <= 0:
            return False, "价格无效"
        
        return True, "OK"
    
    def _check_fund_flow_quality(self, data: Dict) -> Tuple[bool, str]:
        """检查资金流数据质量"""
        if 'net_flow' not in data:
            return False, "缺少净流字段"
        
        if abs(data.get('net_flow', 0)) > 1e12:
            return False, "资金流数据异常"
        
        return True, "OK"
    
    def check(self, data_type: str, data: Dict) -> Tuple[bool, str]:
        """执行质量检查"""
        checker = self._rules.get(data_type)
        if checker:
            return checker(data)
        return True, "无检查规则"


class DataCacheWithFallback:
    """
    带降级的缓存机制
    
    功能：
    - 多级缓存
    - 缓存失效时使用历史数据
    - 数据新鲜度管理
    """
    
    def __init__(self, max_cache_age_hours: int = 24):
        self._cache = {}
        self._cache_metadata = {}
        self._max_cache_age = max_cache_age_hours * 3600
    
    def get(self, key: str, allow_stale: bool = True) -> Optional[Dict]:
        """获取缓存数据"""
        if key not in self._cache:
            return None
        
        metadata = self._cache_metadata.get(key, {})
        created_at = metadata.get('created_at', 0)
        
        if time.time() - created_at <= self._max_cache_age:
            return self._cache[key]
        
        if allow_stale:
            log.warning(f"⚠️ 使用过期缓存: {key}")
            return self._cache[key]
        
        return None
    
    def set(self, key: str, data: Dict):
        """设置缓存数据"""
        self._cache[key] = data
        self._cache_metadata[key] = {
            'created_at': time.time(),
            'data_size': len(str(data))
        }
    
    def invalidate(self, key: str):
        """失效缓存"""
        if key in self._cache:
            del self._cache[key]
            del self._cache_metadata[key]
    
    def cleanup(self):
        """清理过期缓存"""
        now = time.time()
        expired_keys = [
            key for key, metadata in self._cache_metadata.items()
            if now - metadata.get('created_at', 0) > self._max_cache_age * 2
        ]
        
        for key in expired_keys:
            self.invalidate(key)
        
        log.info(f"🧹 清理过期缓存: {len(expired_keys)} 条")

