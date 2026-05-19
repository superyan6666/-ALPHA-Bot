from dataclasses import dataclass
from typing import Callable
from enum import Enum

class FactorGroup(str, Enum):
    VAL = "VALUE"             # 价值/基本面/低位防守类
    MOM = "MOMENTUM"          # 动量/强势突破/资金接力类
    POS = "POSITION"          # 位置/形态类 (绝对低位/多头趋势)
    MA20 = "MA20_SUPPORT"     # 均线支撑类 (贴地潜伏/强势发力)
    VOL = "VOLUME"            # 量能活跃类 (放量/异动/红盘)
    VCP = "VCP_PATTERN"       # 波动率收敛与洗盘结构类
    TREND = "TREND_RSI"       # 趋势指标类 (RSI/DEA/BullRank)
    SPECIAL = "SPECIAL_ALPHA" # 特殊独立加分项 (如MACD底背离/龙头首阴)
    FUNDAMENTAL = "FUNDAMENTAL" # 新增：基本面质量因子 (ROE/营收增长)
    RESONANCE = "RESONANCE"   # 新增：多因子共振加成项

@dataclass
class Factor:
    condition: Callable[[dict], bool]
    points: int
    weight: float = 1.0
    template: str = ""
    group: str = ""

def get_factors_config(f_val: float, f_mom: float, f_rev: float, f_risk: float, 
                       tw: float, rw: float, m_regime: str, 
                       in_danger: bool, danger_label: str, data: dict = None) -> list[Factor]:
    """
    数据驱动的因子引擎配置表。
    
    【动态权重参数说明】
    - f_val (Value): 价值因子权重乘数。在熊市(BEAR)中放大，优先考虑低市盈率/破净股的安全边际。
    - f_mom (Momentum): 动量因子权重乘数。在牛市(BULL)中放大，追随资金热度和强势突破。
    - f_rev (Reversal): 反转因子权重乘数。在冰点期(PANIC)中放大，捕捉恐慌盘杀跌后的超跌反弹。
    - f_risk (Risk): 风险惩罚乘数。在熊市和冰点期显著放大，严厉惩罚短期暴涨、均线破位等高危形态。
    - tw (Trend Weight): 趋势乘数。当 ADX > 25 (主升浪) 时被激活放大。
    - rw (Reversal Weight): 震荡反转乘数。当 ADX < 15 (震荡蓄势) 时被激活放大。
    """
    # 调整震荡市的 RS 因子权重
    adj_rs = 1.3 if m_regime == 'NEUTRAL' else 1.0
    # 调整热门板块的因子权重
    hot_sector_boost = 1.2 if (data and data.get('in_hot_sector', False)) else 1.0
    # 财报窗口时禁用连板因子
    zt_enabled = not in_danger
    
    factors = []
    
    # --- 【激活沉睡特征：强化版】 ---
    factors.append(Factor(lambda d: d.get('rs_rating', 0) > 10,  12, f_mom * adj_rs * hot_sector_boost, "- 🏆 **RS Line 强势**：近60日涨幅远超指数，机构持续运作的优质标的 (强加分)", FactorGroup.MOM))
    factors.append(Factor(lambda d: d.get('is_pivot_point', False), 15, f_mom * (1.4 if m_regime == 'BULL' else 1.0), "- 🚀 **口袋支点**：股价中期调整后放量突破前高，经典强势启动信号 (强加分)", FactorGroup.MOM))
    factors.append(Factor(lambda d: d.get('red_days', 0) >= 3 and d.get('surge_5d', 0) < 15, 5, f_mom, "- 🔴 **连阳吸筹**：连续3日阳线且未过度上涨，资金持续流入", FactorGroup.VOL))
    factors.append(Factor(lambda d: data and data.get('close_val', 0) < data.get('boll_lower', float('inf')) * 1.02 and d.get('vol_ratio', 0) < 0.8, 8, f_rev, "- 🟢 **布林下轨**：股价触及下轨且缩量，低吸信号", FactorGroup.POS))
    factors.append(Factor(lambda d: d.get('extreme_shrink_vol', False), 5, f_rev * rw, "- 🧊 **极致缩量**：较20日均量萎缩30%以上，供应枯竭", FactorGroup.VCP))
    
    # --- 【新增：NR7/ID波幅收缩因子】 ---
    factors.append(Factor(lambda d: d.get('is_nr7', False) and d.get('close_val', 0) > (data.get('ma10_val', 0) if data else 0), 5, f_rev, "- 🔬 **NR7窄幅**：今日振幅是近7日最小，变盘蓄力信号", FactorGroup.VCP))
    
    factors.append(Factor(lambda d: d.get('macd_divergence', False), 25, 1.0, "- 🧲 **MACD底背离**：日线级别价格创新低但动能衰竭，极其罕见的左侧黄金坑 (触发强加权)", FactorGroup.SPECIAL))
    factors.append(Factor(lambda d: d.get('mcap', 0) > 300e8 and 0 < d.get('pe', -1) < 25 and d.get('pb', 10) < 3, 10, f_val, "- 🏢 **价值蓝筹**：大市值低估值核心资产，防守属性极强", FactorGroup.VAL))
    factors.append(Factor(lambda d: d.get('vol_ratio', 0) > 1.0 and d.get('rs_rating', 0) > 5, 10, f_mom, "- 🚀 **强势领涨**：近期显著强于大盘，资金接力意愿极强", FactorGroup.MOM))
    factors.append(Factor(lambda d: d.get('price_pct', 1.0) < 0.3 and 0 < d.get('pb', 10) < 1.0, 8, f_val, "- ♻️ **困境反转**：股价严重破净且处于绝对低位，安全垫极厚", FactorGroup.VAL))
    
    factors.append(Factor(lambda d: d.get('in_hot_sector', False), 12, f_mom * 1.2, "- 🌡️ **身处主线**：所在板块【{hot_sector_name}】今日强势领涨，踏准市场节奏", FactorGroup.MOM))
    
    factors.append(Factor(lambda d: d.get('price_pct', 1.0) < 0.25, 12, f_rev * rw, "- 🟢 **绝对低位**：目前买入相当于抄底，长线持有安全", FactorGroup.POS))
    factors.append(Factor(lambda d: 0.25 <= d.get('price_pct', 1.0) <= 0.45, 8, f_rev, "- 🟢 **相对低位**：刚刚从底部爬起来，输时间不输钱", FactorGroup.POS))
    factors.append(Factor(lambda d: d.get('price_pct', 0.0) > 0.45, 6, f_mom, "- 📈 **多头趋势**：股价已脱离底部，处于健康的主升浪区间", FactorGroup.POS))
    factors.append(Factor(lambda d: d.get('price_pct', 0.0) > 0.85, 8, f_mom, "- 🚀 **高位突破**：股价处于年度高位，强者恒强趋势极佳", FactorGroup.POS)) 
    
    factors.append(Factor(lambda d: d.get('pe', -1) > 0 and d.get('pe', 100) < 40, 5, f_val, "- 🛡️ **业绩护体**：市盈率健康，不是炒空气的无基本面股", FactorGroup.VAL))
    factors.append(Factor(lambda d: d.get('macd_dea', -1.0) >= -0.05, 5, 1.0, "- 🌊 **多头控盘**：大周期趋势仍强，没有被深套的风险", FactorGroup.TREND)) 
    
    factors.append(Factor(lambda d: -2.0 <= d.get('dist_ma20', 100) <= 6.0, 12, 1.0, "- 🧲 **贴地潜伏**：目前价格紧贴均线支撑，绝佳安全低吸点", FactorGroup.MA20))
    factors.append(Factor(lambda d: 6.0 < d.get('dist_ma20', 0) <= 15.0, 6, f_mom, "- 🚀 **强势发力**：距离20日线有空间，依托短期均线强势上攻", FactorGroup.MA20))
    factors.append(Factor(lambda d: d.get('dist_ma20', 0) < -2.0, -10, f_risk, "- ⚠️ **破位嫌疑**：当前已跌破20日线，需警惕趋势走坏 (扣分)"))
    
    factors.append(Factor(lambda d: 30 <= d.get('rsi', 50) <= 72, 5, 1.0, "- 📊 **温度适中**：RSI处于健康买入区间，正是下手时机", FactorGroup.TREND))
    
    factors.append(Factor(lambda d: d.get('bull_rank', False), 8, f_mom, "- 📈 **顺势而为**：均线多头排列，跟着主力资金大部队走", FactorGroup.TREND))
    
    # 财报窗口时连板因子禁掉
    if zt_enabled:
        factors.append(Factor(lambda d: d.get('has_zt', False), 8, 1.0, "- 🔥 **股性活跃**：该股历史上容易涨停，不会一潭死水", FactorGroup.VOL))
    factors.append(Factor(lambda d: d.get('vol_ratio', 0) >= 1.8, 8, 1.0, "- 🔵 **放量确认**：今天成交量明显放大，大资金开始干活了", FactorGroup.VOL))
    factors.append(Factor(lambda d: d.get('red_days', 0) >= 2, 5, 1.0, "- 🔴 **稳步推升**：最近重心都在上移，主力在偷偷温和建仓", FactorGroup.VOL))
    factors.append(Factor(lambda d: d.get('has_obv_break', False), 10, tw * f_mom, "- 💸 **真金白银**：模型监控到真实的资金在创纪录净流入", FactorGroup.VOL))
    
    factors.append(Factor(lambda d: d.get('has_chip_break', False), 12, tw * f_mom, "- 🏔️ **抛压真空**：上方的套牢盘已割肉离场，向上拉升没阻力", FactorGroup.VCP))
    factors.append(Factor(lambda d: d.get('is_true_vcp', False), 12, 1.0, "- 🎯 **形态确认**：呈现经典 VCP (波动率收敛) 结构，洗盘极度充分", FactorGroup.VCP))
    factors.append(Factor(lambda d: not d.get('is_true_vcp', False) and d.get('vcp_amp', 1.0) < 0.12, 6, 1.0, "- 🟣 **蓄势待发**：近期波动极小，面临短线方向选择", FactorGroup.VCP))
    factors.append(Factor(lambda d: d.get('extreme_shrink_vol', False), 8, 1.0, "- 🧊 **没人砸盘**：爆发前夕成交极度萎缩，散户该卖的都卖了", FactorGroup.VCP)) 
    factors.append(Factor(lambda d: d.get('has_pullback', False), 12, 1.0, "- 🪃 **黄金深坑**：出现温和缩量回踩，主力洗盘给出的上车良机", FactorGroup.VCP))
    factors.append(Factor(lambda d: d.get('lower_shadow_ratio', 0) > 0.03, 5, 1.0, "- 📌 **强力护盘**：跌下去被大资金迅速买回，下方有人兜底", FactorGroup.VCP)) 
    
    factors.append(Factor(lambda d: d.get('rs_rating', 0) > 5,  8, f_mom, "- 🏆 **跑赢大盘**：近60日涨幅超越指数，有资金在持续运作", FactorGroup.MOM))
    
    # --- 【均线粘合发散优化】 ---
    factors.append(Factor(lambda d: d.get('is_ma_converging', False), 8, 1.0, "- 🔍 **均线粘合**：各条均线高度接近，变盘窗口临近", FactorGroup.VCP))
    factors.append(Factor(lambda d: d.get('is_ma_diverging_up', False), 12, tw * f_mom, "- 🌊 **均线首次发散**：粘合后首次向上发散，趋势启动早期信号", FactorGroup.MOM))
    
    # --- 【基本面质量因子】 ---
    roe_boost = 2.0 if m_regime == 'BEAR' else 1.0
    factors.append(Factor(lambda d: d.get('roe', 0) > 15, 10, f_val * roe_boost, "- 💎 **ROE 优异**：净资产收益率 >15%，内生盈利能力强，护城河深厚", FactorGroup.FUNDAMENTAL))
    factors.append(Factor(lambda d: d.get('roe', 0) > 10, 5, f_val * roe_boost, "- 🛡️ **ROE 良好**：净资产收益率 >10%，盈利能力达标", FactorGroup.FUNDAMENTAL))
    factors.append(Factor(lambda d: d.get('revenue_growth', 0) > 20, 10, f_mom, "- 📈 **营收高增**：近季度营收增速 >20%，成长属性显著", FactorGroup.FUNDAMENTAL))
    factors.append(Factor(lambda d: d.get('profit_growth', 0) > 15, 8, f_mom, "- 💵 **利润增长**：近季度净利润增速 >15%，业绩向好", FactorGroup.FUNDAMENTAL))
    factors.append(Factor(lambda d: m_regime != 'BULL' and d.get('dividend_yield', 0) > 2, 5, f_val, "- 💰 **股息防御**：股息率 >2%，非牛市阶段具有抗跌防守属性", FactorGroup.FUNDAMENTAL))
    
    # --- 【多因子共振加成】 ---
    factors.append(Factor(lambda d: d.get('is_pivot_point', False) and d.get('rs_rating', 0) > 10 and d.get('in_hot_sector', False), 5, 1.0, "- 🌟 **三因子共振**：口袋支点 + 强势 RS + 热门板块，胜率显著提升", FactorGroup.RESONANCE))
    
    # --- 【排雷扣分项】 ---
    factors.append(Factor(lambda d: d.get('surge_5d', 0) > 28, -20, f_risk, "- 🚫 **短期暴涨**：近5日涨幅过大透支空间，极易高位站岗 (重度扣分)"))
    factors.append(Factor(lambda d: d.get('consecutive_down', 0) >= 4, -15, f_risk, "- 🔪 **飞刀预警**：近期连续阴线急跌，左侧接飞刀风险大 (重度扣分)"))
    factors.append(Factor(lambda d: d.get('rsi', 50) > 85, -10, f_risk, "- 🌡️ **短期过热**：RSI偏高短线超买，操作需要进一步缩减仓位"))
    factors.append(Factor(lambda d: d.get('rs_rating', 0) < -10, -8, f_risk, "- 📉 **跑输大盘**：近期持续弱于大盘，跟的是被冷落的股票"))
    
    # 财报窗口时低位连板因子也禁掉
    if zt_enabled:
        factors.append(Factor(lambda d: d.get('has_consecutive_zt', False) and d.get('price_pct', 1.0) < 0.40, 10, f_mom, "- 🔥 **低位连板**：刚刚启动的龙头，安全且市场辨识度极高", FactorGroup.MOM))
    factors.append(Factor(lambda d: d.get('has_consecutive_zt', False) and d.get('price_pct', 0.0) >= 0.90 and not (d.get('is_first_dip', False) and m_regime != 'BEAR'), -15, f_risk, "- ⚠️ **高位接盘**：股价已被炒高连板，千万别追容易接盘！"))
    
    # 财报窗口时龙头首阴因子也禁掉
    if zt_enabled and m_regime != 'BEAR':
        factors.append(Factor(lambda d: d.get('is_first_dip', False), 20, f_mom, "- 🐉 **龙头首阴**：连板龙头首次缩量温和回调，量价健康且未破5日线，游资经典接力点！", FactorGroup.SPECIAL))
    factors.append(Factor(lambda d: d.get('upper_shadow_pct', 0) > 35, -15, f_risk, "- ⚠️ **诱多预警**：冲高后大幅跳水，上方抛压极重别上当！"))
    factors.append(Factor(lambda d: d.get('dist_ma20', 0) > 25, -15, f_risk, "- 🚫 **追高预警**：目前涨得太急离均线太远，随时面临暴跌回调"))
    
    factors.append(Factor(lambda d: in_danger and d.get('mcap', 100e8) < 100e8, -8, f_risk, f"- 📅 **财报防雷**：当前属于{danger_label}，小盘股需防业绩变脸 (扣分)"))
    factors.append(Factor(lambda d: d.get('has_financial_red_flag', False), -25, f_risk, "- 🚨 **财务风险**：速动/现金流/商誉等指标异常，存在爆雷风险 (重度扣分)"))
    
    return factors


def get_etf_factors_config(f_mom: float, m_regime: str) -> list[Factor]:
    """ETF专属因子配置"""
    factors = []
    
    f_mom_adj = f_mom * 1.3 if m_regime == 'BULL' else 1.0
    
    factors.append(Factor(lambda d: d.get('rank_20d', 0) >= 0.75, 15, f_mom_adj, "- 🏆 **强势动量**：20日涨幅排名同类前25%，资金持续涌入", FactorGroup.MOM))
    factors.append(Factor(lambda d: 0.5 <= d.get('rank_20d', 0) < 0.75, 8, f_mom_adj, "- 📈 **动量尚可**：20日涨幅排名同类中上水平", FactorGroup.MOM))
    factors.append(Factor(lambda d: d.get('rank_20d', 0) < 0.25, 8, f_mom * 0.5, "- 🟢 **超跌反弹**：20日涨幅排名靠后，存在均值回归机会", FactorGroup.POS))
    
    factors.append(Factor(lambda d: d.get('bull_rank', False), 12, f_mom_adj, "- 📈 **均线多头**：MA20 > MA60，中长期上升趋势确认", FactorGroup.TREND))
    factors.append(Factor(lambda d: d.get('rsi', 50) < 40, 8, f_mom * 0.7, "- 🟢 **RSI低位**：技术指标处于超卖区间，反弹概率大", FactorGroup.TREND))
    factors.append(Factor(lambda d: 40 <= d.get('rsi', 50) <= 65, 6, f_mom, "- ⚖️ **RSI健康**：技术指标处于健康区间，可持续", FactorGroup.TREND))
    
    factors.append(Factor(lambda d: d.get('vol_ratio', 0) >= 1.5, 10, f_mom, "- 🔵 **量能放大**：成交量明显放大，趋势确认", FactorGroup.VOL))
    factors.append(Factor(lambda d: d.get('has_obv_break', False), 12, f_mom, "- 💸 **资金涌入**：OBV突破近期高点，主力介入", FactorGroup.VOL))
    
    factors.append(Factor(lambda d: 0.3 <= d.get('price_pct', 0) <= 0.7, 8, 1.0, "- ⚖️ **位置适中**：处于年度价格区间中部，趋势健康", FactorGroup.POS))
    factors.append(Factor(lambda d: d.get('pct_chg', 0) > 3, 5, f_mom, "- 🚀 **今日强势**：单日涨幅超过3%，动能强劲", FactorGroup.MOM))
    
    factors.append(Factor(lambda d: d.get('surge_5d', 0) > 15, -10, 1.5, "- ⚠️ **短期过热**：近5日涨幅过大，追高风险"))
    factors.append(Factor(lambda d: d.get('rsi', 50) > 75, -8, 1.5, "- ⚠️ **RSI过热**：技术指标超买，注意回调风险"))
    
    return factors


def get_cb_factors_config(f_val: float, f_mom: float, m_regime: str) -> list[Factor]:
    """可转债专属因子配置"""
    factors = []
    
    factors.append(Factor(lambda d: d.get('double_low', 200) < 100, 15, f_val, "- 💎 **双低优选**：价格低+溢价低，下有保底上有弹性", FactorGroup.VAL))
    factors.append(Factor(lambda d: 100 <= d.get('double_low', 200) < 120, 10, f_val, "- 🟢 **双低尚可**：价格和溢价率适中，配置价值", FactorGroup.VAL))
    factors.append(Factor(lambda d: 120 <= d.get('double_low', 200) < 140, 5, f_val, "- ⚖️ **双低一般**：溢价率偏高，股性偏弱", FactorGroup.VAL))
    
    factors.append(Factor(lambda d: d.get('bond_rt', 50) < 15, 10, f_val, "- 🛡️ **债底保护强**：纯债溢价率低于15%，债性显著", FactorGroup.VAL))
    factors.append(Factor(lambda d: d.get('bond_rt', 50) < 25, 6, f_val, "- ⚖️ **债底保护可**：纯债溢价率适中", FactorGroup.VAL))
    
    factors.append(Factor(lambda d: 2e8 <= d.get('scale', 0) <= 10e8, 8, 1.0, "- 📊 **规模适中**：剩余规模2-10亿，流动性好", FactorGroup.VAL))
    
    factors.append(Factor(lambda d: d.get('stock_pct', 0) > 3 and d.get('cb_price', 100) < 130, 10, f_mom, "- 🚀 **正股强势**：正股大涨且转债未过度透支，可关注", FactorGroup.MOM))
    factors.append(Factor(lambda d: d.get('cb_price', 100) < 105, 8, f_val, "- 💰 **价格接近债底**：债底附近配置安全性极高", FactorGroup.VAL))
    
    factors.append(Factor(lambda d: d.get('cb_price', 100) > 150, -12, 1.5, "- 🚫 **价格过高**：超过150元风险大，弹性有限"))
    factors.append(Factor(lambda d: d.get('premium_rt', 0) > 50, -10, 1.5, "- 🚫 **溢价过高**：转股溢价率超过50%，股性极弱"))
    factors.append(Factor(lambda d: d.get('scale', 1e9) < 3e7, -8, 1.5, "- 🚫 **规模过小**：剩余规模低于3000万，流动性风险"))
    
    return factors
