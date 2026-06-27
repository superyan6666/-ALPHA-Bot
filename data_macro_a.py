"""
data_macro_a.py
A股宏观日历事件计算模块
功能：返回未来指定天数内的计划性高波动事件列表
"""

import datetime
import calendar
from typing import List, Dict

# ===========================
# 假设的交易日历辅助函数
# 需根据实际系统实现
# ===========================

def is_weekend(d: datetime.date) -> bool:
    """判断是否为周末（周六/周日）"""
    return d.weekday() >= 5

def is_holiday(d: datetime.date) -> bool:
    """
    判断是否为中国法定假日（含调休）
    实际需维护一个假日字典，或接入交易日历库
    """
    # 示例：仅处理固定假日，春节等需动态计算
    CN_HOLIDAYS = {
        # 元旦
        datetime.date(d.year, 1, 1),
        # 春节（需每年更新，此处占位）
        # 清明节、劳动节、端午节、中秋节、国庆节等
    }
    # 调休上班日（虽为周末但开市）需从假日中排除，此处简化
    return d in CN_HOLIDAYS

def next_business_day(d: datetime.date) -> datetime.date:
    """返回下一个交易日（跳过周末和假日）"""
    d += datetime.timedelta(days=1)
    while is_weekend(d) or is_holiday(d):
        d += datetime.timedelta(days=1)
    return d

def prev_business_day(d: datetime.date) -> datetime.date:
    """返回上一个交易日"""
    d -= datetime.timedelta(days=1)
    while is_weekend(d) or is_holiday(d):
        d -= datetime.timedelta(days=1)
    return d

def is_trading_day(d: datetime.date) -> bool:
    """判断是否为交易日"""
    return not (is_weekend(d) or is_holiday(d))

def next_trading_day_if_holiday(d: datetime.date) -> datetime.date:
    """若目标日非交易日，则顺延至下一交易日"""
    while not is_trading_day(d):
        d = d + datetime.timedelta(days=1)
    return d

# ===========================
# 1. 衍生品到期日
# ===========================

def get_etf_option_expiry(year: int, month: int) -> datetime.date:
    """
    上证50/沪深300/中证500/中证1000 ETF期权及股指期权到期日
    规则：每月第四周的周三
    """
    first_day = datetime.date(year, month, 1)
    # 找到第一个周三 (weekday=2)
    first_wed = first_day + datetime.timedelta(days=(2 - first_day.weekday()) % 7)
    fourth_wed = first_wed + datetime.timedelta(days=21)  # 第四个周三
    return next_trading_day_if_holiday(fourth_wed)

def get_index_future_settlement(year: int, month: int) -> datetime.date:
    """
    股指期货交割日（IF, IC, IH, IM）
    规则：每月第三周的周五
    """
    first_day = datetime.date(year, month, 1)
    first_fri = first_day + datetime.timedelta(days=(4 - first_day.weekday()) % 7)
    third_fri = first_fri + datetime.timedelta(days=14)
    return next_trading_day_if_holiday(third_fri)

def get_triple_witching_a(year: int, month: int) -> datetime.date | None:
    """
    A股版“三巫日”：3/6/9/12月，股指期货+国债期货+ETF期权可能同时到期
    实际上国债期货交割日为3/6/9/12月第二个周五，可能不重合，但波动叠加
    我们返回3/6/9/12月的第三周五（股指期货）作为高波动日
    """
    if month in [3, 6, 9, 12]:
        return get_index_future_settlement(year, month)
    return None

# ===========================
# 2. 货币政策与宏观数据
# ===========================

def get_lpr_date(year: int, month: int) -> datetime.date:
    """
    LPR（贷款市场报价利率）发布日
    规则：每月20日（遇周末/假日顺延至下一交易日）9:15发布
    """
    d = datetime.date(year, month, 20)
    return next_trading_day_if_holiday(d)

def get_mlf_date(year: int, month: int) -> datetime.date:
    """
    MLF（中期借贷便利）操作日（每月15日）
    遇周末/假日顺延，央行通常于当日公布操作利率和规模
    """
    d = datetime.date(year, month, 15)
    return next_trading_day_if_holiday(d)

def get_pmi_date(year: int, month: int) -> datetime.date:
    """
    官方制造业/非制造业PMI发布日
    规则：每月最后一天（若最后一日非交易日，则提前至前一交易日）
    注：实际统计局会在最后一日9:00准时公布，如果最后一日是周末，则提前到周五
    """
    # 当月最后一天
    _, last_day_num = calendar.monthrange(year, month)
    last_day = datetime.date(year, month, last_day_num)
    # 向前找到最近的一个交易日
    while not is_trading_day(last_day):
        last_day -= datetime.timedelta(days=1)
    return last_day

def get_cpi_ppi_date(year: int, month: int) -> datetime.date:
    """
    中国CPI/PPI数据发布日（估算）
    规则：通常在每月9-15日之间，多数落在第二周的周三
    我们采用每月第二周的周三作为估算值
    """
    first_day = datetime.date(year, month, 1)
    first_wed = first_day + datetime.timedelta(days=(2 - first_day.weekday()) % 7)
    second_wed = first_wed + datetime.timedelta(weeks=1)
    return next_trading_day_if_holiday(second_wed)

def get_gdp_and_industrial_data(year: int, month: int) -> datetime.date | None:
    """
    GDP、工业增加值、消费、投资等重磅月度/季度数据
    通常在季度后次月的15-20日公布，但具体日期不定
    我们可以保守地在每月15日标注“宏观数据周”，或按MLF同日提醒
    此处暂返回None，由更精确日历替代
    """
    # 可维护一个固定字典，或统一在15日前后提示
    return None

# ===========================
# 3. 海外事件映射（影响A股次日的关键事件）
# ===========================

# 假设美股FOMC日期字典已存在（可从美股模块导入，此处简化）
FOMC_DATES = {
    2026: [
        datetime.date(2026,1,28), datetime.date(2026,3,18),
        datetime.date(2026,5,6),  datetime.date(2026,6,17),
        datetime.date(2026,7,29), datetime.date(2026,9,23),
        datetime.date(2026,11,5), datetime.date(2026,12,16)
    ]
}

def get_fomc_a_impact(start: datetime.date, end: datetime.date) -> List[Dict]:
    """
    美联储FOMC议息会议对A股的影响提示（次日开盘）
    返回会议日期+1日（A股交易日），提醒隔夜波动
    """
    events = []
    for year_dates in FOMC_DATES.values():
        for fomc_date in year_dates:
            # 直接计算影响日（A股面对决议的次日开盘）
            next_day = fomc_date + datetime.timedelta(days=1)
            impact_date = next_trading_day_if_holiday(next_day)
            if start <= impact_date <= end:
                events.append({
                    "date": impact_date.isoformat(),
                    "event_type": "fomc_impact",
                    "label_zh": f"美联储FOMC决议次日影响 (美东{fomc_date.isoformat()})",
                    "priority": "normal"  # A股间接影响，中等优先级
                })
    return events

def get_nfp_impact(start: datetime.date, end: datetime.date) -> List[Dict]:
    """
    美国非农数据对A股影响（每月第一个周五晚，影响A股下周一）
    这里生成A股受影响日：非农后的下一个A股交易日
    """
    events = []
    # 遍历 start 前一个月到 end 后一个月，确保不遗漏
    d = start.replace(day=1) - datetime.timedelta(days=1)
    d = d.replace(day=1)
    while d <= end + datetime.timedelta(days=32):
        year, month = d.year, d.month
        first_day = datetime.date(year, month, 1)
        # 每月第一个周五
        nfp_date = first_day + datetime.timedelta(days=(4 - first_day.weekday()) % 7)
        # A股受影响日
        next_day = nfp_date + datetime.timedelta(days=1)
        impact_date = next_trading_day_if_holiday(next_day)
        if start <= impact_date <= end:
            events.append({
                "date": impact_date.isoformat(),
                "event_type": "nfp_impact",
                "label_zh": f"美国非农数据隔夜影响 (美东{nfp_date.isoformat()})",
                "priority": "normal"
            })
        # 月份推进
        if month == 12:
            d = datetime.date(year+1, 1, 1)
        else:
            d = datetime.date(year, month+1, 1)
    return events

# ===========================
# 4. 资金面与季节性事件
# ===========================

def get_quarter_end_ranking(year: int, month: int) -> datetime.date | None:
    """
    季末机构排名/资金回笼压力（3/6/9/12月末最后一个交易日）
    """
    if month in [3, 6, 9, 12]:
        return get_pmi_date(year, month)  # PMI日即是月末，复用逻辑
    return None

def get_month_end_liquidity(year: int, month: int) -> datetime.date:
    """
    月末流动性紧张（每月最后一个交易日）
    """
    return get_pmi_date(year, month)

# ===========================
# 5. 节假日与特殊日期
# ===========================

def get_northbound_close_dates(start: datetime.date, end: datetime.date) -> List[datetime.date]:
    """
    北向资金暂停日（港股休市日）
    需维护香港假期列表，此处返回示例占位
    """
    HK_HOLIDAYS = {
        # 示例：2026年香港假期
        datetime.date(2026, 1, 1),  # 元旦
        datetime.date(2026, 2, 17), # 春节(港股)
        # ... 需完整维护
    }
    close_dates = [d for d in HK_HOLIDAYS if start <= d <= end and is_trading_day(d)]
    return close_dates

CN_LONG_HOLIDAYS = {
    # 中国长假（春节、国庆等）及邻近交易日，需每年更新
    # 格式: (开始日, 结束日, 假期名称)
    (datetime.date(2026, 2, 15), datetime.date(2026, 2, 23), "春节"),
    (datetime.date(2026, 10, 1), datetime.date(2026, 10, 7), "国庆节"),
}

def get_pre_holiday_effects(start: datetime.date, end: datetime.date) -> List[Dict]:
    """
    长假前两个交易日提示：交投清淡，建议降低过夜仓位
    """
    events = []
    for holiday_start, holiday_end, name in CN_LONG_HOLIDAYS:
        # 找到长假前的最后一个交易日
        last_trading_day_before = prev_business_day(holiday_start)
        # 提前2天提醒：倒数第2和第1个交易日
        d1 = last_trading_day_before
        d2 = prev_business_day(d1)
        for d in [d2, d1]:
            if start <= d <= end:
                events.append({
                    "date": d.isoformat(),
                    "event_type": "pre_holiday",
                    "label_zh": f"{name}长假前倒数交易日，建议降低仓位",
                    "priority": "normal"
                })
    return events

# ===========================
# 6. 其他可扩展事件（占位）
# ===========================

def get_lockup_expiration_alerts(start: datetime.date, end: datetime.date) -> List[Dict]:
    """
    限售股大规模解禁预警（需接入解禁数据，此处占位）
    返回未来大规模解禁日（例如解禁市值 > 50亿）
    """
    # TODO: 接入解禁数据源
    return []

def get_earnings_season_deadline(start: datetime.date, end: datetime.date) -> List[Dict]:
    """
    年报/中报/三季报披露截止日，业绩雷暴期
    年报：4月30日；中报：8月31日；三季报：10月31日
    若遇非交易日则顺延至前一交易日
    """
    deadlines = [
        (4, 30, "年报披露截止日"),
        (8, 31, "中报披露截止日"),
        (10, 31, "三季报披露截止日")
    ]
    year = start.year
    events = []
    # 考虑跨年的情况，检查 start 到 end 涉及的每一年
    for y in range(start.year, end.year + 1):
        for month, day, label in deadlines:
            d = datetime.date(y, month, day)
            # 若当日非交易日，实际截止日会顺延，但很多公司踩点，因此前几日雷暴密集
            # 我们将截止日当天标注为业绩雷暴提醒
            while not is_trading_day(d):
                d -= datetime.timedelta(days=1)
            
            if start <= d <= end:
                events.append({
                    "date": d.isoformat(),
                    "event_type": "earnings_deadline",
                    "label_zh": label,
                    "priority": "normal"
                })
    return events

# ===========================
# 综合事件生成器
# ===========================

def get_upcoming_a_share_events(days_ahead: int = 7) -> List[Dict]:
    """
    返回未来 days_ahead 天内的所有计划性高波动事件
    """
    today = datetime.date.today()
    end_date = today + datetime.timedelta(days=days_ahead)
    events = []

    # 遍历日期范围内涉及的月份
    current = today.replace(day=1)
    while current <= end_date:
        y, m = current.year, current.month

        # 衍生品到期
        opt_exp = get_etf_option_expiry(y, m)
        if today <= opt_exp <= end_date:
            events.append({
                "date": opt_exp.isoformat(),
                "event_type": "etf_option_expiry",
                "label_zh": "ETF/股指期权到期日",
                "priority": "high",
                "interpretation": "每月第四周三，期权行权及Gamma对冲集中。尾盘30分钟（尤其14:30后）权重成分股易现直线拉升或跳水，建议降低裸卖头寸，谨慎追尾盘突破。"
            })

        fut_settle = get_index_future_settlement(y, m)
        if today <= fut_settle <= end_date:
            events.append({
                "date": fut_settle.isoformat(),
                "event_type": "index_future_settle",
                "label_zh": "股指期货交割日",
                "priority": "high",
                "interpretation": "每月第三周五，主力合约换月移仓。多空平仓与转仓行为易引发午盘后单边波动，可关注基差（升贴水）剧烈变动方向，避免在交割时点追涨杀跌。"
            })

        # 三巫日（季月）
        triple = get_triple_witching_a(y, m)
        if triple and today <= triple <= end_date:
            events.append({
                "date": triple.isoformat(),
                "event_type": "triple_witching_a",
                "label_zh": "季月衍生品集中到期日",
                "priority": "high",
                "interpretation": "季月三重共振（股指期货+ETF期权+个股期权），历年当月波动率平均提升20%以上。可能出现“钓鱼单”引发的瞬间涨跌停，建议降低程序化交易频率，流动性衰竭时暂离盘面。"
            })

        # 货币政策
        lpr = get_lpr_date(y, m)
        if today <= lpr <= end_date:
            events.append({
                "date": lpr.isoformat(),
                "event_type": "lpr",
                "label_zh": "LPR报价公布日",
                "priority": "high",
                "interpretation": "每月20日9:15公布LPR，银行/地产/基建板块开盘即反应。若利率下调超预期，短线利好高负债率行业；若维持不变但市场预期落空，易出现“卖事实”行情，持有相关仓位者可在9:25集合竞价阶段关注盘口异常挂单。"
            })

        mlf = get_mlf_date(y, m)
        if today <= mlf <= end_date:
            events.append({
                "date": mlf.isoformat(),
                "event_type": "mlf",
                "label_zh": "MLF操作日",
                "priority": "high",
                "interpretation": "每月15日公布MLF量价操作（通常9:20）。利率下调视为“降息先导信号”，利空银行、利好成长股；利率不变但超额续作则仅边际改善流动性。盘中可追踪DR007加权均价及国债期货联动，确认市场解读方向。"
            })

        # 宏观数据
        cpi = get_cpi_ppi_date(y, m)
        if today <= cpi <= end_date:
            events.append({
                "date": cpi.isoformat(),
                "event_type": "cpi_ppi",
                "label_zh": "中国CPI/PPI数据发布",
                "priority": "high",
                "interpretation": "通常在9:30前发布。若CPI不及预期、PPI持续负值，市场将博弈“通缩-强刺激”逻辑，消费板块承压但政策敏感型行业（如建材、工程机械）可能逆势。相反，PPI大幅上行会强化上游原材料盈利预期，但压制中游制造板块利润。建议提前检查持仓中的周期股敞口。"
            })

        pmi = get_pmi_date(y, m)
        if today <= pmi <= end_date:
            events.append({
                "date": pmi.isoformat(),
                "event_type": "pmi",
                "label_zh": "官方PMI数据发布日",
                "priority": "normal",
                "interpretation": "当月最后一个交易日9:00发布。若连续两月低于荣枯线，市场将迅速计入“降准/特别国债”等政策预期，建筑建材、能源金属等板块或提前异动。高于50且超预期则可能压制题材炒作，资金回流业绩白马。"
            })

        # 季末资金压力
        q_end = get_quarter_end_ranking(y, m)
        if q_end and today <= q_end <= end_date:
            events.append({
                "date": q_end.isoformat(),
                "event_type": "quarter_end",
                "label_zh": "季末机构排名/资金回笼日",
                "priority": "normal",
                "interpretation": "季末最后一周，机构拉净值与资金回笼并存。高位基金重仓股可能出现“拉抬或反手出货”双面行为，避免在最后半小时跟风基金重仓异动。可观察GC001（国债逆回购）年化利率，若盘中飙升意味着资金面告急，指数回调概率加大。"
            })

        month_end = get_month_end_liquidity(y, m)
        if today <= month_end <= end_date:
            events.append({
                "date": month_end.isoformat(),
                "event_type": "month_end",
                "label_zh": "月末流动性紧张日",
                "priority": "normal",
                "interpretation": "月末银行头寸收紧，国债逆回购利率（如GC001）常现脉冲式上行。若日间突破5%~8%，流动性敏感板块（次新股、券商）首先承压，建议控制隔夜仓位，避免使用高成本杠杆资金。"
            })

        # 月份递增
        if m == 12:
            current = datetime.date(y+1, 1, 1)
        else:
            current = datetime.date(y, m+1, 1)

    # 海外事件映射
    for e in get_fomc_a_impact(today, end_date):
        e["interpretation"] = "美联储议息决议隔夜落地。A股低开概率较高（尤其是鹰派信号），北向资金盘初可能出现集中净流出砸盘。建议盘前设置好止损，开盘15分钟内避免重仓追高，等待北向及离岸人民币汇率（CNH）方向明朗后行动。"
        events.append(e)
    
    for e in get_nfp_impact(today, end_date):
        e["interpretation"] = "非农数据若超预期强劲 → 美元走强 → A股外资重仓股承压（白酒、新能源），早盘或出现估值修正。若大幅不及预期 → 美国衰退叙事升温，出口产业链（纺织、家电）面临需求担忧。无论哪种情况，第一个小时波动率会骤然放大，适合观望。"
        events.append(e)

    # 北向资金关闭
    for d in get_northbound_close_dates(today, end_date):
        events.append({
            "date": d.isoformat(),
            "event_type": "northbound_close",
            "label_zh": "港股休市，北向资金暂停",
            "priority": "normal",
            "interpretation": "港股假期致北向资金暂停，当日市场外资定价缺失。历史统计：北向关闭日小盘股（如中证2000）活跃度上升，但成交量萎缩10-20%，指数易受局部题材炒作影响，不宜重仓跟随板块轮动信号。"
        })

    # 节前效应
    for e in get_pre_holiday_effects(today, end_date):
        e["interpretation"] = "长假前倒数第2日起，两融资金开始净偿还，成交额阶梯式下滑。建议强平线附近账户提前降杠杆，避开最后一日午后流动性枯竭导致的“无量空跌”。持币过节者可在倒数第2日收盘前完成调仓，避免最后一日拥堵。"
        events.append(e)

    # 业绩披露截止日
    for e in get_earnings_season_deadline(today, end_date):
        e["interpretation"] = "业绩披露最后时刻，警惕“爆雷潮”。尤其关注高商誉、高质押比例个股。建议提前检查持仓中未披露业绩的公司，若此前已有预亏公告，截止日前一日盘中可能出现加速下跌，可用条件单保护多头头寸。"
        events.append(e)

    # 限售股解禁（待实现）
    # events.extend(get_lockup_expiration_alerts(today, end_date))

    # 按日期排序
    events.sort(key=lambda x: x["date"])
    return events

# ===========================
# 测试入口
# ===========================
if __name__ == "__main__":
    # 运行测试打印未来14天事件
    print("【A股宏观高危日历扫描】")
    print("-" * 50)
    for e in get_upcoming_a_share_events(14):
        print(f"[{e['date']}] {e['label_zh']} (优先级: {e['priority']})")
        print(f" 💡 解读: {e.get('interpretation', '无')}")
        print("-" * 50)
