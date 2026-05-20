
# 量化引擎架构文档

## 项目结构

```
/workspace/
├── main.py                        # 保留的原完整文件（兼容版本）
├── factors_config.py             # 因子配置文件
├── daily_briefing.py           # 日报生成文件
├── core/
│   ├── __init__.py
│   └── config.py              # 核心配置类
├── data/
│   ├── __init__.py
│   ├── proxy.py             # 数据代理类 DataProxy
│   └── cache.py             # 本地数据缓存 LocalDataLake
└── pipelines/
│   ├── __init__.py
│   ├── base.py             # 流水线基类 BasePipeline
│   └── stock.py            # (待完善) 股票策略流水线
├── paper_trades.json
├── pushed_state.json
└── ARCHITECTURE.md
```

## 模块说明

### 1. core/config.py

核心配置模块，包含：
- `Cols`: 列名常量定义
- `Config`: 全局配置类
- `StrategyConfig`: 策略配置类
- `StrategyType`: 策略类型枚举

### 2. data/proxy.py

数据代理模块，提供统一的数据获取接口：
- `DataProxy`: 多源数据获取（Akshare / Tushare / Baostock / Efinance）
- 支持数据回退机制
- 重试和异常处理

### 3. data/cache.py

本地数据缓存模块：
- `LocalDataLake`: 本地数据缓存类
- 缓存文件管理
- 模拟交易和推送状态管理

### 4. pipelines/base.py

流水线基类模块：
- `PipelineResult`: 流水线执行结果数据类
- `BasePipeline`: 策略流水线基类（抽象）

## 使用指南

### 向后兼容

原有的 `main.py` 保持完整，可以继续使用。

### 新架构使用方式

```python
# 导入新模块
from core.config import Config
from data import DataProxy, LocalDataLake
from pipelines import pipelines.base import BasePipeline

# 初始化数据代理
proxy = DataProxy()
dl = LocalDataLake(proxy)
```

## 策略类型扩展

已支持的策略类型：
- `StrategyType.STOCK`: 股票策略
- `StrategyType.ETF`: ETF轮动策略
- `StrategyType.CONVERTIBLE_BOND`: 可转债策略
- `StrategyType.SECTOR`: 行业轮动
- `StrategyType.HK`: 港股通
- `StrategyType.US`: 美股映射

