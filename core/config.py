
"""
核心配置模块
"""
import os
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum

from dotenv import load_dotenv

load_dotenv()


class Cols:
    """数据列名常量"""
    S_PRICE: str = '最新价'
    S_HIGH: str = '最高'
    S_LOW: str = '最低'
    S_OPEN: str = '今开'
    S_PCT: str = '涨跌幅'
    S_TURN: str = '换手率'
    S_AMT: str = '成交额'
    S_VOL: str = '成交量'
    S_CODE: str = '代码'
    S_NAME: str = '名称'
    S_MCAP: str = '流通市值'
    S_PE: str = '市盈率-动态'
    S_PB: str = '市净率'
    S_VR: str = '量比'
    H_DATE: str = '日期'
    H_OPEN: str = '开盘'
    H_CLOSE: str = '收盘'
    H_HIGH: str = '最高'
    H_LOW: str = '最低'
    I_CLOSE: str = 'close'
    B_NAME: str = '板块名称'
    B_PCT: str = '涨跌幅'


class StrategyType:
    """策略类型枚举"""
    STOCK = "stock"
    ETF = "etf"
    CONVERTIBLE_BOND = "cb"
    SECTOR = "sector"
    HK = "hk"
    US = "us"


@dataclass(frozen=True)
class Config:
    """全局配置类"""
    DINGTALK_WEBHOOK: Optional[str] = os.getenv("DINGTALK_WEBHOOK")
    CORE_POOL_ONLY: bool = os.getenv("CORE_POOL_ONLY", "false").lower() == "true"
    IS_MANUAL: bool = os.getenv("IS_MANUAL", "false").lower() == "true"
    PUSH_EMPTY: bool = os.getenv("PUSH_EMPTY", "false").lower() == "true"
    RUN_MODE: str = os.getenv("RUN_MODE", "all").lower()
    IS_TEST_MODE: bool = os.getenv("TEST_MODE", "false").lower() == "true"

    # 股票池筛选参数
    MIN_PE: float = float(os.getenv("MIN_PE", "0.5"))
    MAX_PE: float = float(os.getenv("MAX_PE", "150.0"))
    MIN_CAP: float = float(os.getenv("MIN_CAP", "10.0")) * 100000000.0
    MAX_CAP: float = float(os.getenv("MAX_CAP", "800.0")) * 100000000.0
    MIN_PCT_CHG: float = float(os.getenv("MIN_PCT_CHG", "-10.0"))
    MAX_PRICE: float = float(os.getenv("MAX_PRICE", "120.0"))
    MIN_TURNOVER: float = float(os.getenv("MIN_TURNOVER", "0.5"))
    MAX_TURNOVER: float = float(os.getenv("MAX_TURNOVER", "25.0"))
    MIN_VOL_RATIO: float = float(os.getenv("MIN_VOL_RATIO", "0.5"))
    MAX_VOL_RATIO: float = float(os.getenv("MAX_VOL_RATIO", "20.0"))

    REQUIRED_COLS: Tuple[str, ...] = (Cols.S_PRICE, Cols.S_OPEN, Cols.S_HIGH, Cols.S_LOW, Cols.S_VOL, Cols.S_AMT,
                                      Cols.S_PCT, Cols.S_TURN, Cols.S_CODE, Cols.S_NAME, Cols.S_MCAP, Cols.S_PE, Cols.S_PB)
    OPTIONAL_COLS: Tuple[str, ...] = (Cols.S_VR,)
    HIST_COLS: Tuple[str, ...] = (Cols.H_DATE, Cols.H_OPEN, Cols.H_CLOSE, Cols.H_HIGH, Cols.H_LOW, Cols.S_VOL)

    @staticmethod
    def print_summary(logger):
        """打印配置摘要"""
        logger.info("=" * 60)
        logger.info("🤖 配置摘要")
        logger.info(f"   - 钉钉推送: {'✅' if Config.DINGTALK_WEBHOOK else '❌'}")
        logger.info(f"   - 运行模式: {Config.RUN_MODE}")
        logger.info(f"   - 核心池: {'✅' if Config.CORE_POOL_ONLY else '❌'}")
        logger.info(f"   - 测试模式: {'✅' if Config.IS_TEST_MODE else '❌'}")
        logger.info("=" * 60)


@dataclass(frozen=True)
class ETFConfig:
    """ETF策略配置"""
    enabled: bool = True
    max_positions: int = 3
    min_volume: float = 5000 * 10000
    atr_multiplier: float = 2.5
    rank_window: int = 20
    min_rank_pct: float = 0.25


@dataclass(frozen=True)
class CBConfig:
    """可转债策略配置"""
    enabled: bool = False
    max_positions: int = 2
    max_price: float = 130.0
    max_premium: float = 30.0
    min_rating: str = "AA"
    min_scale: float = 2 * 100000000


@dataclass(frozen=True)
class SectorConfig:
    """行业轮动策略配置"""
    enabled: bool = True
    max_sectors: int = 3
    momentum_threshold: float = 5.0
    reversal_threshold: float = -8.0


@dataclass(frozen=True)
class HKConfig:
    """港股通策略配置"""
    enabled: bool = False
    max_positions: int = 2
    atr_multiplier: float = 2.5
    min_dividend: float = 3.0


@dataclass(frozen=True)
class USConfig:
    """美股策略配置"""
    enabled: bool = False
    max_positions: int = 3
    atr_multiplier: float = 2.0


class StrategyConfig:
    """策略全局配置"""

    def __init__(self):
        self.etf = ETFConfig()
        self.cb = CBConfig()
        self.sector = SectorConfig()
        self.hk = HKConfig()
        self.us = USConfig()
        self._load_from_env()

    def _load_from_env(self):
        """从环境变量加载配置"""
        active = os.getenv('ACTIVE_STRATEGIES', 'stock,etf,sector').split(',')
        self.etf = ETFConfig(enabled='etf' in active)
        self.cb = CBConfig(enabled='cb' in active)
        self.sector = SectorConfig(enabled='sector' in active)
        self.hk = HKConfig(enabled='hk' in active)
        self.us = USConfig(enabled='us' in active)

    def is_enabled(self, strategy_type: str) -> bool:
        """检查策略是否启用"""
        if strategy_type == StrategyType.STOCK:
            return True
        return getattr(self, strategy_type, None) and getattr(self, strategy_type).enabled


# 全局配置实例
config = Config()
strategy_config = StrategyConfig()

