# 量化引擎架构文档

## 项目结构

```
/workspace/
├── main.py                    # 原有完整文件（向后兼容）
├── factors_config.py          # 因子配置文件
├── daily_briefing.py          # 日报生成文件
├── ARCHITECTURE.md            # 本文档
│
├── core/                      # 核心配置模块
│   ├── __init__.py
│   └── config.py              # Cols / Config / StrategyConfig / StrategyType
│
├── data/                      # 数据获取模块
│   ├── __init__.py
│   ├── proxy.py               # DataProxy 多源数据代理
│   └── cache.py              # LocalDataLake 本地缓存
│
└── pipelines/                 # 策略流水线模块
    ├── __init__.py
    ├── base.py                # BasePipeline / PipelineResult
    ├── etf.py                 # ETFPipeline ETF轮动策略
    ├── cb.py                  # CBPipeline 可转债策略
    ├── sector.py              # SectorPipeline 行业轮动策略
    └── scheduler.py           # PipelineScheduler 流水线调度器
```

## 模块说明

### 1. core/config.py

核心配置模块，包含：
- `Cols`: 列名常量定义（股票/历史/指数列名）
- `Config`: 全局配置类（从环境变量加载）
- `StrategyConfig`: 策略配置管理（ETF/可转债/行业/港股/美股）
- `StrategyType`: 策略类型枚举

### 2. data/proxy.py

数据代理模块，提供统一的数据获取接口：
- `WAFBypassSession`: 带 WAF 旁路的请求会话（线程安全）
- `retry()`: 重试装饰器
- `DataProxy`: 多源数据获取类
  - 历史数据: Tushare → Baostock → Akshare
  - 实时行情: QMT → Efinance → Akshare → Tushare
  - 指数数据: 腾讯 → 东方财富 → Baostock
  - 核心池: Akshare → Tushare → 静态兜底
  - ETF/可转债/港股/南向资金

### 3. data/cache.py

本地数据缓存模块：
- `LocalDataLake`: 本地数据缓存类
  - 内存缓存 + 文件缓存双层
  - TTL 时效控制
  - 自动清理过期缓存
- 模拟交易和推送状态管理

### 4. pipelines/base.py

流水线基类模块：
- `PipelineResult`: 流水线执行结果数据类
- `BasePipeline`: 策略流水线抽象基类

### 5. pipelines/etf.py

ETF轮动策略流水线：
- `ETFPipeline`: ETF 轮动策略
- `ETFSignal`: ETF 信号数据类
- 功能：
  - ETF 候选筛选（成交量/价格/涨跌幅过滤）
  - 20日涨幅排名计算
  - 均线多头/量能放大检测
  - 牛市顺势增强/熊市反弹机会识别

### 6. pipelines/cb.py

可转债策略流水线：
- `CBPipeline`: 可转债双低策略
- `CBSignal`: 可转债信号数据类
- 功能：
  - 双低筛选（价格 < 150，溢价率 < 30%）
  - 债底保护评估（纯债溢价率）
  - 规模/评级过滤
  - 正股强势映射

### 7. pipelines/sector.py

行业轮动策略流水线：
- `SectorPipeline`: 行业轮动策略
- `SectorSignal`: 行业信号数据类
- 功能：
  - 主升信号识别（20日涨幅 > 5% + 量能放大）
  - 反弹信号识别（超跌 + 今日反弹）
  - 行业强弱排名

### 8. pipelines/scheduler.py

流水线调度器：
- `PipelineScheduler`: 统一调度多个流水线
- 功能：
  - 自动注册启用的流水线
  - 共享上下文传递
  - 统一结果收集和排序

## 使用示例

```python
from core.config import Config
from data import DataProxy, LocalDataLake
from pipelines import PipelineScheduler

# 初始化
proxy = DataProxy()
dl = LocalDataLake(proxy)

# 创建调度器（自动注册所有启用的流水线）
scheduler = PipelineScheduler(dl)

# 共享上下文
context = {
    'market_regime': 'BULL',
    'market_ok': True,
    'hot_sectors': ['半导体', '人工智能']
}

# 运行所有流水线
results = scheduler.run_all(context)

# 获取所有信号并排序
all_signals = scheduler.get_all_signals(results)

# 或者运行指定策略
etf_result = scheduler.run_strategy('etf', context)
```

## 环境变量配置

```bash
# 策略启用控制
ACTIVE_STRATEGIES=stock,etf,sector,cb,hk,us

# 数据源 Token
TUSHARE_TOKEN=your_token_here
DINGTALK_WEBHOOK=your_webhook_here

# 运行模式
RUN_MODE=all
CORE_POOL_ONLY=false
TEST_MODE=false
```

## 策略优先级

| 策略 | 优先级 | 说明 |
|------|--------|------|
| stock | ⭐⭐⭐⭐⭐ | 股票策略（默认启用） |
| etf | ⭐⭐⭐⭐ | ETF轮动（默认启用） |
| sector | ⭐⭐⭐ | 行业轮动（默认启用） |
| cb | ⭐⭐ | 可转债（默认禁用） |
| hk | ⭐⭐ | 港股通（默认禁用） |
| us | ⭐ | 美股映射（默认禁用） |

