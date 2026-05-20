"""
数据模块

提供数据获取、缓存、挖掘等功能：
- 多源数据代理（故障转移）
- 本地数据缓存
- 数据挖掘引擎
"""
from .proxy import DataProxy
from .cache import (
    LocalDataLake,
    load_pushed_state,
    save_pushed_state,
    is_recently_pushed,
    load_and_update_paper_trades,
    save_paper_trades,
    get_score_bucket
)
from .multi_source_proxy import (
    MultiSourceDataProxy,
    DataSourceConfig,
    DataSourceStatus,
    DataRequest,
    DataResponse,
    DataQualityChecker,
    DataCacheWithFallback,
)
from .data_mining import (
    DataMiningEngine,
    SmartDataFusion,
    DataInsight,
    FeatureCorrelation,
    AnomalyDetection,
)

__all__ = [
    # 原有模块
    'DataProxy',
    'LocalDataLake',
    'load_pushed_state',
    'save_pushed_state',
    'is_recently_pushed',
    'load_and_update_paper_trades',
    'save_paper_trades',
    'get_score_bucket',
    # 多源代理
    'MultiSourceDataProxy',
    'DataSourceConfig',
    'DataSourceStatus',
    'DataRequest',
    'DataResponse',
    'DataQualityChecker',
    'DataCacheWithFallback',
    # 数据挖掘
    'DataMiningEngine',
    'SmartDataFusion',
    'DataInsight',
    'FeatureCorrelation',
    'AnomalyDetection',
]

