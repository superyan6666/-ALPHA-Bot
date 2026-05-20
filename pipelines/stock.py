"""
股票策略流水线
"""
import time
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from core.config import StrategyType
from pipelines.base import BasePipeline, PipelineResult

log = logging.getLogger(__name__)


@dataclass
class StockSignal:
    """股票信号"""
    code: str
    name: str
    price: float
    pct_chg: float
    score: int
    level: str
    reasons: str
    stop_loss: float
    target1: float
    strategy_type: str = "stock"
    money_risk_msg: str = ""
    tranche_plan_msg: str = ""
    plan_b_msg: str = ""
    hold_period_msg: str = ""


class MathUtils:
    """数学工具类"""
    
    @staticmethod
    def calc_vcp_quality(df: pd.DataFrame) -> tuple:
        """VCP质量计算"""
        if len(df) < 31:
            return 0.5, False
            
        segments = []
        for i in [(-31, -21), (-21, -11), (-11, -1)]:
            seg = df.iloc[i[0]:i[1]]
            low = seg['low'].min()
            if low > 0:
                amp = (seg['high'].max() - low) / low
                segments.append(amp)
                
        if len(segments) < 3:
            return segments[-1] if segments else 0.5, False
            
        is_vcp = segments[0] > segments[1] > segments[2]
        return segments[-1], is_vcp

    @staticmethod
    def calc_atr_adx(hist: pd.DataFrame, period: int = 14) -> tuple:
        """ATR和ADX计算"""
        high, low, close = hist['high'], hist['low'], hist['close']
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        up, dn = high.diff(), -low.diff()
        plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
        minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
        plus_di = 100 * pd.Series(plus_dm, index=hist.index).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=hist.index).rolling(period).mean() / atr
        denom = (plus_di + minus_di).replace(0, np.nan)
        dx = ((plus_di - minus_di).abs() / denom * 100)
        return atr, dx.rolling(period).mean()


def is_earnings_danger_zone(now: datetime) -> tuple:
    """检查是否在财报危险期"""
    month = now.month
    DANGER_WINDOWS = [
        (3, 25, 4, 30, "年报/一季报披露末期"),
        (8, 15, 8, 31, "半年报披露末期"),
        (10, 15, 10, 31, "三季报披露末期"),
    ]
    for s_m, s_d, e_m, e_d, label in DANGER_WINDOWS:
        start_dt = now.replace(month=s_m, day=s_d, hour=0, minute=0)
        end_dt = now.replace(month=e_m, day=e_d, hour=23, minute=59)
        if start_dt <= now <= end_dt:
            return True, label
    return False, ""


class StockTechnicals:
    """股票技术特征提取"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        close = self.df['close']
        high, low, vol = self.df['high'], self.df['low'], self.df['volume']
        
        for span in (5, 10, 20, 60): 
            self.df[f'MA{span}'] = close.rolling(span).mean()
        self.df['MA5_V'] = vol.rolling(5).mean()
        self.df['MA20_V'] = vol.rolling(20).mean()

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        self.df['DIF'] = ema12 - ema26
        self.df['DEA'] = self.df['DIF'].ewm(span=9, adjust=False).mean()
        self.df['MACD'] = (self.df['DIF'] - self.df['DEA']) * 2

        self.df['ATR'], self.df['ADX'] = MathUtils.calc_atr_adx(self.df)
        
        delta = close.diff()
        gain = delta.clip(lower=0.0).rolling(14).mean()
        loss = (-delta.clip(upper=0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        self.df['RSI14'] = 100 - (100 / (1 + rs))

        self.df['REF_C'] = close.shift()
        self.df['PCT_CHG'] = close.pct_change() * 100
        self.df['OBV'] = np.where(close > self.df['REF_C'], vol, np.where(close < self.df['REF_C'], -vol, 0)).cumsum()
        
        self.today = self.df.iloc[-1]
        self.yest = self.df.iloc[-2]

    def get_features(self) -> Optional[dict]:
        df, today, yest = self.df, self.today, self.yest
        if pd.isna(today['ATR']) or today['ATR'] <= 1e-5:
            return None

        min_1y, max_1y = df['low'].min(), df['high'].max()
        rng = max_1y - min_1y
        if rng <= 0:
            return None
        
        price_pct = (today['close'] - min_1y) / rng
        
        if today['close'] < today['MA20'] * 0.85:
            return None

        rsi = float(today.get('RSI14', 50))
        if pd.isna(rsi) or rsi > 85:
            return None

        consecutive_down = 0
        for i in range(2, 8):
            if len(df) >= i and df['close'].iloc[-i] < df['open'].iloc[-i]:
                consecutive_down += 1
            else:
                break

        extreme_shrink_vol = yest['volume'] < today['MA20_V'] * 0.75

        rec120 = df.iloc[-121:-1]
        has_chip_break = False
        if len(rec120) > 20 and rec120['volume'].sum() > 0:
            counts, edges = np.histogram(rec120['close'].values, bins=20, weights=rec120['volume'].values)
            poc = (edges[counts.argmax()] + edges[counts.argmax() + 1]) / 2
            has_chip_break = bool((today['REF_C'] <= poc) and (today['close'] > poc))

        red_days = 0
        for i in range(1, 4):
            if df['close'].iloc[-i] > df['open'].iloc[-i]:
                red_days += 1
            else:
                break
            
        total_range = today['high'] - today['low']
        upper_shadow_pct = ((today['high'] - max(today['open'], today['close'])) / total_range * 100) if total_range > 1e-5 else 0.0

        last_hist_pct = float(df['PCT_CHG'].iloc[-2]) if len(df) >= 2 else 0.0
        has_pullback = bool(
            today['close'] >= today['MA20'] * 0.97 and 
            today['volume'] < today['MA5_V'] * 1.2 and
            -6.0 <= last_hist_pct <= 3.5
        )
        
        surge_5d = (today['close'] / df['close'].iloc[-6] - 1) * 100 if len(df) >= 6 else 0.0
        
        vcp_amp, is_true_vcp = MathUtils.calc_vcp_quality(df)

        macd_divergence = False
        if len(df) >= 40:
            w1_low = float(df['low'].iloc[-40:-20].min())
            w2_low = float(df['low'].iloc[-20:].min())
            w1_macd_min = float(df['MACD'].iloc[-40:-20].min())
            w2_macd_min = float(df['MACD'].iloc[-20:].min())
            if w2_low < w1_low and w2_macd_min > w1_macd_min and w2_macd_min < 0:
                if float(today.get('ADX', 50)) < 30 or extreme_shrink_vol or is_true_vcp:
                    macd_divergence = True

        recent_consec_zt = False
        if len(df) >= 5:
            recent_consec_zt = bool(((df['PCT_CHG'].iloc[-6:-1] >= 9.5).rolling(2).sum() >= 2).any())

        is_first_dip = False
        if recent_consec_zt:
            today_is_green = today['close'] < today['open'] or float(df['PCT_CHG'].iloc[-1]) < 0
            yest_is_red = yest['close'] >= yest['open']
            above_ma5 = today['close'] >= today['MA5']
            no_nuclear = float(df['PCT_CHG'].iloc[-1]) > -6.5
            no_trap = upper_shadow_pct < 20.0
            vol_shrink = today['volume'] <= yest['volume'] * 1.1
            
            if today_is_green and yest_is_red and above_ma5 and no_nuclear and no_trap and vol_shrink:
                is_first_dip = True

        is_pivot_point = False
        if len(df) >= 30:
            prev_10_high = float(df['high'].iloc[-11:-1].max())
            volume_surge = today['volume'] > today['MA20_V'] * 1.5
            price_breakout = today['close'] > prev_10_high
            ma_trend = today['MA10'] > today['MA20']
            if volume_surge and price_breakout and ma_trend:
                is_pivot_point = True

        is_ma_converging = False
        is_ma_diverging_up = False
        if len(df) >= 25:
            ma_spread = (df[['MA5', 'MA10', 'MA20']].std(axis=1) / df['MA20']).iloc[-25:]
            current_spread = ma_spread.iloc[-1]
            spread_percentile = (ma_spread < current_spread).sum() / len(ma_spread)
            is_ma_converging = bool(spread_percentile < 0.2 and current_spread < 0.02)
            if is_ma_converging or ma_spread.iloc[-2] < 0.02:
                if today['MA5'] > today['MA10'] > today['MA20'] and current_spread > ma_spread.iloc[-2] * 1.2:
                    is_ma_diverging_up = True
        
        is_nr7 = False
        if len(df) >= 10:
            amp_recent = df.iloc[-8:-1].apply(lambda x: x['high'] - x['low'], axis=1)
            today_amp = today['high'] - today['low']
            if len(amp_recent) >= 6:
                min_amp = amp_recent.min()
                is_nr7 = bool(today_amp <= min_amp * 1.05)

        return {
            'is_first_dip': is_first_dip,
            'macd_divergence': macd_divergence,
            'price_pct': price_pct,
            'max_1y': max_1y,
            'adx': float(today['ADX']),
            'bull_rank': (today['MA20'] > today['MA60']),
            'extreme_shrink_vol': extreme_shrink_vol,
            'has_zt': bool((df['PCT_CHG'].iloc[-61:-1] >= 9.5).any()),
            'has_consecutive_zt': bool(((df['PCT_CHG'].iloc[-61:-1] >= 9.5).rolling(2).sum() >= 2).any()),
            'vcp_amp': vcp_amp,
            'is_true_vcp': is_true_vcp,
            'upper_shadow_pct': upper_shadow_pct,
            'has_obv_break': bool(df['OBV'].iloc[-1] > df['OBV'].iloc[-21:-1].max()),
            'has_pullback': has_pullback,
            'has_chip_break': has_chip_break,
            'dist_ma20': (today['close'] / today['MA20'] - 1) * 100,
            'red_days': red_days,
            'rsi': rsi,
            'consecutive_down': consecutive_down,
            'surge_5d': surge_5d,
            'macd_dea': float(today['DEA']),
            'ma10_val': float(today['MA10']),
            'ma20_val': float(today['MA20']),
            'atr_val': float(today['ATR']),
            'close_val': float(today['close']),
            'low_val': float(today['low']),
            'recent_20_low': float(df['low'].iloc[-20:].min()),
            'is_pivot_point': is_pivot_point,
            'is_ma_converging': is_ma_converging,
            'is_ma_diverging_up': is_ma_diverging_up,
            'is_nr7': is_nr7,
            'roe': 0.0,
            'revenue_growth': 0.0,
            'profit_growth': 0.0,
            'dividend_yield': 0.0,
            'has_financial_red_flag': False,
        }


class StockPipeline(BasePipeline):
    """股票策略流水线"""
    
    def __init__(self, data_lake, config=None):
        self._data_lake = data_lake
        self._config = config or {}
    
    @property
    def name(self) -> str:
        return "股票策略"
    
    def is_enabled(self) -> bool:
        return True
    
    def run(self, shared_context: dict) -> PipelineResult:
        """
        执行股票策略
        
        Args:
            shared_context: 共享上下文
            
        Returns:
            股票策略结果
        """
        signals = []
        watchlist = []
        market_msg = ""
        
        try:
            now = datetime.now()
            df_spot = self._data_lake.fetch_spot()
            
            if df_spot.empty:
                return PipelineResult(
                    signals=signals,
                    watchlist=watchlist,
                    market_msg="无行情数据",
                    meta_info={}
                )
            
            market_regime = shared_context.get('market_regime', 'NEUTRAL')
            market_ok = shared_context.get('market_ok', True)
            market_overheated = shared_context.get('market_overheated', False)
            hot_sectors = shared_context.get('hot_sectors', {})
            vol_surge = shared_context.get('vol_surge', False)
            win_stats = shared_context.get('win_stats', {})
            
            core_pool = self._data_lake.fetch_core_pool()
            is_core_only = shared_context.get('core_only', True)
            
            filtered = self._filter_stocks(df_spot, core_pool, is_core_only)
            
            if filtered.empty:
                return PipelineResult(
                    signals=signals,
                    watchlist=watchlist,
                    market_msg="无符合条件的股票",
                    meta_info={}
                )
            
            scored = self._score_stocks(filtered, now, market_regime, vol_surge, 
                                       win_stats, hot_sectors, market_overheated)
            
            scored.sort(key=lambda x: x[1], reverse=True)
            
            for code, score, reasons, stop, target, price in scored:
                if score >= 80:
                    row = df_spot[df_spot['code'] == code].iloc[0]
                    name = str(row.get('名称', code))
                    pct = float(row.get('涨跌幅', 0))
                    level = self._get_level(score)
                    
                    money_risk = self._format_money_risk(price, stop, target)
                    tranche = self._generate_tranche(price, score, market_ok, market_overheated)
                    plan_b = self._generate_plan_b(price, stop, row.get('MA20', price * 0.98))
                    hold = self._generate_hold_period(
                        scored[0][0] if scored else None,
                        {'adx': 20, 'price_pct': 0.5, 'has_chip_break': False}
                    )
                    
                    signals.append(StockSignal(
                        code=code, name=name, price=price, pct_chg=pct,
                        score=score, level=level, reasons=reasons,
                        stop_loss=stop, target1=target,
                        money_risk_msg=money_risk, tranche_plan_msg=tranche,
                        plan_b_msg=plan_b, hold_period_msg=hold
                    ))
                elif score >= 60:
                    watchlist.append((code, df_spot[df_spot['code'] == code]['名称'].values[0], score))
            
            market_msg = f"扫描 {len(filtered)} 只股票，{len(signals)} 个信号，{len(watchlist)} 只观察"
            
        except Exception as e:
            log.error(f"股票策略执行失败: {e}")
            market_msg = f"执行失败: {str(e)}"
        
        return PipelineResult(
            signals=signals,
            watchlist=watchlist,
            market_msg=market_msg,
            meta_info={"strategy": StrategyType.STOCK, "count": len(signals)}
        )
    
    def _filter_stocks(self, df_spot, core_pool: set, is_core_only: bool) -> pd.DataFrame:
        """筛选股票"""
        df = df_spot.copy()
        
        for col in ['代码', '最新价', '涨跌幅', '换手率', '流通市值', '市盈率-动态']:
            if col not in df.columns:
                df[col] = np.nan
        
        df['code'] = df['代码'].astype(str).str.zfill(6)
        df['price'] = pd.to_numeric(df['最新价'], errors='coerce')
        df['pct'] = pd.to_numeric(df['涨跌幅'], errors='coerce')
        df['turnover'] = pd.to_numeric(df['换手率'], errors='coerce')
        df['mcap'] = pd.to_numeric(df['流通市值'], errors='coerce')
        df['pe'] = pd.to_numeric(df['市盈率-动态'], errors='coerce')
        
        mask = (
            df['price'].notna() & (df['price'] > 0) &
            df['price'].notna() & (df['price'] < 150) &
            df['pct'].notna() & (df['pct'] >= -10) & (df['pct'] <= 10) &
            df['turnover'].notna() & (df['turnover'] >= 0.5) & (df['turnover'] <= 30) &
            df['mcap'].notna() & (df['mcap'] >= 10e8) & (df['mcap'] <= 2000e8)
        )
        
        if is_core_only and core_pool:
            mask &= df['code'].isin(core_pool)
        
        return df[mask]
    
    def _score_stocks(self, stocks, now, market_regime, vol_surge, 
                      win_stats, hot_sectors, market_overheated) -> List:
        """对股票打分"""
        scored = []
        
        def process_stock(row):
            code = row['code']
            try:
                end = datetime.now().strftime('%Y%m%d')
                start = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
                hist = self._data_lake.fetch_hist(code, start, end)
                
                if hist is None or len(hist) < 60:
                    return None
                
                tech = StockTechnicals(hist)
                features = tech.get_features()
                
                if features is None:
                    return None
                
                score, reasons = self._apply_scoring(
                    features, now, market_regime, vol_surge,
                    hot_sectors, market_overheated
                )
                
                if score < 60:
                    return None
                
                stop = self._calc_stop_loss(features, row['price'])
                target = self._calc_target(row['price'], stop, features)
                
                return (code, score, reasons, stop, target, float(row['price']))
                
            except Exception as e:
                log.debug(f"处理 {code} 失败: {e}")
                return None
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_stock, row): idx 
                      for idx, row in stocks.iterrows()}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    scored.append(result)
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def _apply_scoring(self, data: dict, now: datetime, m_regime: str, vol_surge: bool,
                       hot_sectors: dict, market_overheated: bool) -> tuple:
        """应用打分逻辑"""
        adx = data['adx']
        tw, rw = (1.4, 0.7) if adx > 25 else (0.8, 1.4) if adx < 15 else (1.0, 1.0)
        
        f_val, f_mom, f_rev, f_risk = 1.0, 1.0, 1.0, 1.0
        
        if m_regime == 'BULL':
            f_mom, f_val, f_risk = 1.3, 0.8, 0.8
        elif m_regime == 'BEAR':
            f_val, f_mom, f_rev, f_risk = 1.3, 0.6, 1.2, 1.5
        elif m_regime == 'PANIC':
            f_rev, f_mom, f_val, f_risk = 1.5, 0.3, 1.2, 1.5
        
        if vol_surge:
            f_mom += 0.2
        
        in_danger, danger_label = is_earnings_danger_zone(now)
        
        score = 45
        reasons = []
        
        if data['bull_rank']:
            score += int(12 * tw)
            reasons.append("均线多头")
        
        if data['has_chip_break']:
            score += int(15 * f_mom)
            reasons.append("筹码突破")
        
        if data['is_first_dip'] and not market_overheated:
            score += int(18 * rw)
            reasons.append("龙头首阴")
        
        if data['macd_divergence']:
            score += int(12 * rw)
            reasons.append("MACD底背离")
        
        if data['is_true_vcp']:
            score += int(10 * f_mom)
            reasons.append("VCP收紧")
        
        if data['has_pullback']:
            score += int(8 * f_rev)
            reasons.append("缩量回踩")
        
        if data['extreme_shrink_vol']:
            score += int(5 * f_rev)
            reasons.append("极致缩量")
        
        if data['has_obv_break']:
            score += int(8 * f_mom)
            reasons.append("资金流入")
        
        if data['is_pivot_point']:
            score += int(15 * f_mom)
            reasons.append("口袋支点")
        
        if data['is_ma_diverging_up']:
            score += int(12 * f_mom)
            reasons.append("均线发散")
        
        if data['is_nr7']:
            score += int(8 * f_rev)
            reasons.append("NR7收缩")
        
        resonance_bonus = 0
        if data['is_pivot_point'] and data['has_obv_break']:
            resonance_bonus = 5
            reasons.append("共振加成")
        
        score += resonance_bonus
        
        score = min(score, 100)
        
        bucket = self._get_score_bucket(score)
        if bucket in win_stats and win_stats[bucket]['total'] >= 5:
            wr = win_stats[bucket]['win'] / win_stats[bucket]['total']
            score = int(score * (0.8 + 0.4 * wr))
            reasons.append(f"历史胜率{int(wr*100)}%")
        
        score = max(0, min(score, 100))
        
        return score, " | ".join(reasons)
    
    def _calc_stop_loss(self, data: dict, price: float) -> float:
        """计算止损价"""
        atr_stop = price - data['atr_val'] * 2.0
        return max(atr_stop, price * 0.92)
    
    def _calc_target(self, price: float, stop: float, data: dict) -> float:
        """计算目标价"""
        risk = price - stop
        if data.get('has_chip_break'):
            max_1y = data.get('max_1y', price * 1.20)
            chip_target = price + (max_1y - price) * 0.5
            min_target = price + risk * 1.5
            return round(max(chip_target, min_target), 2)
        return round(price + risk * 2.0, 2)
    
    def _get_level(self, score: int) -> str:
        """获取等级"""
        if score >= 90:
            return "⭐⭐⭐⭐⭐ 📊 **[S级·最强信号]**"
        elif score >= 80:
            return "⭐⭐⭐⭐ 📈 **[A级·强势信号]**"
        elif score >= 70:
            return "⭐⭐⭐ 📊 **[B+级·可关注]**"
        elif score >= 60:
            return "⭐⭐ 📉 **[观望级]**"
        return "⭐ [观察]"
    
    def _get_score_bucket(self, score: int) -> str:
        """获取分档"""
        if score >= 85:
            return '85-100'
        elif score >= 80:
            return '80-85'
        elif score >= 75:
            return '75-80'
        elif score >= 70:
            return '70-75'
        return '<70'
    
    def _format_money_risk(self, price: float, stop: float, target: float) -> str:
        """格式化资金风险信息"""
        one_hand = price * 100
        budget = 10000
        hands = max(1, int(budget / one_hand))
        cost = hands * one_hand
        loss = (price - stop) * hands * 100
        gain = (target - price) * hands * 100
        ratio = gain / max(loss, 1)
        
        eval_text = "🎯 **高容错**" if ratio >= 2.5 else "✅ **尚可**" if ratio >= 1.5 else "⚠️ **需谨慎**"
        
        return (f"- 💸 **仓位**：买 {hands} 手约 ¥{cost:.0f}\n"
                f"- 🔴 **止损**：-¥{loss:.0f}\n"
                f"- 🟢 **目标**：+¥{gain:.0f}\n"
                f"- 📐 **盈亏比**：1:{ratio:.1f} ➡️ {eval_text}")
    
    def _generate_tranche(self, price: float, score: int, market_ok: bool, overheated: bool) -> str:
        """生成仓位计划"""
        if overheated:
            return "🛑 **【系统熔断】市场极度过热，禁止建仓！**"
        
        base = 30 if score >= 85 else 20 if score >= 70 else 10
        if not market_ok:
            base = base // 2
        
        t1 = max(1, base // 3)
        t2 = max(1, base // 3)
        t3 = max(1, base - t1 - t2)
        
        lower = round(price * 0.985, 2)
        upper = round(price * 1.005, 2)
        add_p = round(price * 1.025, 2)
        stop_add = round(price * 1.05, 2)
        
        return (f"- **① 关注支撑**：`¥{lower} - ¥{upper}` 缩量企稳可分批 **{t1}%** 试错\n"
                f"- **② 稳健加仓**：站稳 `¥{add_p}` 可加仓 **{t2}%**\n"
                f"- **③ 追击确认**：突破 `¥{stop_add}` 追加 **{t3}%**")
    
    def _generate_plan_b(self, price: float, stop: float, ma20: float) -> str:
        """生成备用计划"""
        shake = round(price * 0.97, 2)
        shake = max(shake, stop + 0.01)
        
        return (f"- **📉 正常波动**：收盘未破 `¥{shake:.2f}` 属于洗盘\n"
                f"- **🔪 铁血防线**：有效跌破 `¥{stop:.2f}` 必须止损\n"
                f"- **💥 系统风险**：大盘非理性暴跌优先保本金")
    
    def _generate_hold_period(self, code: str, data: dict) -> str:
        """生成持股预期"""
        adx = data.get('adx', 20)
        price_pct = data.get('price_pct', 0.5)
        has_chip = data.get('has_chip_break', False)
        
        if price_pct < 0.35 and adx < 20:
            return "- **⏳ 持股预期**：🐢 **【底部潜伏型】(1~3个月)**"
        elif adx > 25 or has_chip:
            return "- **⏳ 持股预期**：🐎 **【右侧趋势型】(3~10天)**"
        else:
            return "- **⏳ 持股预期**：🐕 **【稳健震荡型】(2~4周)**"


from datetime import timedelta
