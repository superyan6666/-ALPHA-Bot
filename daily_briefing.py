"""
每日投研简报 (Daily Research Briefing)
--------------------------------------
生成并推送一份面向投资者的每日投研简报，包含：
  1. 🌍 宏观市场分析 (上证指数 / 市场情绪 / 北向资金 / 红绿盘 / 涨停跌停)
  2. 🎯 核心股票信号 (涨停强龙 / 右侧爆发 / 稳健型 / 异动超跌)
  3. 🔄 ETF 板块轮动 (宽基 / 行业 ETF 涨跌幅与强弱关系)
  4. 🌋 行业热点雷达 (今日主线板块及领涨龙头)
  5. 💡 操作策略摘要 (仓位建议 + 短线关注清单)

运行方式：
    $ python daily_briefing.py

环境变量：
    - DINGTALK_WEBHOOK  : 钉钉机器人 webhook
    - FEISHU_WEBHOOK    : 飞书机器人 webhook
    - NOTIFY_SEC_KEYWORD: (可选) 安全关键词, 默认 "AI量化"
"""

import os
import sys
import time
import json
import logging
import traceback
from datetime import datetime

import requests
import numpy as np
import pandas as pd
import pytz

# ============================================================
# 0. 环境 & 日志
# ============================================================
# 不覆盖系统代理环境变量，由 requests 自动读取
TZ_BJS = pytz.timezone('Asia/Shanghai')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger('daily_briefing')

UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'


def _json_get(url: str, timeout: float = 15.0, retries: int = 3) -> dict | list | None:
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers={'User-Agent': UA})
            if r.status_code == 200 and r.text.strip():
                return r.json()
            log.debug(f'  [{i+1}/{retries}] status={r.status_code} body[:100]={r.text[:100]}')
        except Exception as e:
            log.debug(f'  [{i+1}/{retries}] {e}')
        time.sleep(0.5 * (i + 1))
    return None


# ============================================================
# 1. 数据采集
#    说明：
#    - 全市场 A 股  : http://vip.stock.finance.sina.com.cn (Market_Center)
#    - 主流 ETF    : http://hq.sinajs.cn/list=sh510300,... (JS 文本格式)
#    - 行业板块    : 在 A 股数据上做关键词分组推断
#    - 上证指数/北向: https://money.finance.sina.com.cn / 东财接口
# ============================================================
def fetch_spot() -> pd.DataFrame:
    """获取全市场 A 股实时行情"""
    log.info('🚀 拉取全市场 A 股实时行情...')

    rows = []
    max_pages = 60
    for page in range(1, max_pages + 1):
        url = (
            f'http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/'
            f'Market_Center.getHQNodeData?page={page}&num=200&sort=changepercent'
            f'&asc=0&node=hs_a&_s_r_a=page'
        )
        data = _json_get(url, timeout=12, retries=2)
        if not data or not isinstance(data, list):
            break
        rows.extend(data)
        if page % 5 == 0:
            log.info(f'  新浪分页: 已获取 {len(rows)} 条 (page {page})')
        time.sleep(0.15)

    if rows:
        df = pd.DataFrame(rows)
        log.info(f'✅ 新浪源: {len(df)} 只股票')
        return df

    raise RuntimeError('所有 A 股实时行情源均失败')


# 主流 ETF 代码清单（宽基 + 常见行业）
MAIN_ETF_CODES = [
    # 宽基
    ('sh510300', '沪深300ETF'),
    ('sh510050', '上证50ETF'),
    ('sh510500', '中证500ETF'),
    ('sh512100', '中证1000ETF'),
    ('sz159915', '创业板ETF'),
    ('sz159949', '创业板50ETF'),
    ('sh588000', '科创50ETF'),
    ('sh510230', '金融ETF'),
    ('sz159920', '恒生ETF'),
    ('sh513100', '纳指ETF'),
    ('sh513500', '标普500ETF'),
    ('sh518880', '黄金ETF'),
    # 行业
    ('sh512880', '证券ETF'),
    ('sh512170', '医疗ETF'),
    ('sh512690', '酒ETF'),
    ('sh512290', '生物医药ETF'),
    ('sh515030', '新能源车ETF'),
    ('sh515050', '5G ETF'),
    ('sh512760', '半导体ETF'),
    ('sz159995', '芯片ETF'),
    ('sz159997', '电子ETF'),
    ('sh512660', '军工ETF'),
    ('sh512800', '银行ETF'),
    ('sh512670', '国防ETF'),
    ('sz159939', '信息技术ETF'),
    ('sz159806', '新能车ETF'),
    ('sh515790', '光伏ETF'),
    ('sh516160', '新能源ETF'),
    ('sh510410', '沪深300ETF'),
    ('sh512000', '券商ETF'),
    ('sh512010', '医药ETF'),
    ('sh512040', '白酒ETF'),
    ('sz159902', '中小板ETF'),
    ('sh560050', 'MSCI中国ETF'),
    ('sh513080', '法国CAC40'),
    ('sh513520', '日经ETF'),
    ('sh513130', '恒生科技ETF'),
]


def fetch_etf_spot() -> pd.DataFrame:
    """获取主流 ETF 的实时行情（hq.sinajs.cn 多标的查询接口）"""
    log.info('🔄 拉取主流 ETF 实时行情...')
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36',
        'Referer': 'https://finance.sina.com.cn',
    }
    symbols = [c for c, _ in MAIN_ETF_CODES]
    # 批量查询，每次最多 80 只
    rows = []
    batch = 80
    for i in range(0, len(symbols), batch):
        batch_symbols = symbols[i:i + batch]
        url = 'http://hq.sinajs.cn/list=' + ','.join(batch_symbols)
        try:
            r = requests.get(url, timeout=15, headers=headers)
            if r.status_code == 200 and r.text.strip():
                # 解析 JS 文本格式:
                # var hq_str_sh510300="沪深300ETF,4.943,4.958,4.984,5.001,4.941,4.982,4.983,1293343039,6434057084, ... ,2026-06-18,15:00:01,00,";
                # 索引: 0=name, 1=open, 2=preclose, 3=current, 4=high, 5=low, 6=buy, 7=sell, 8=volume, 9=amount, ...
                for line in r.text.strip().split('\n'):
                    if '=' not in line or '""' in line:
                        continue
                    left, right = line.split('=', 1)
                    content = right.strip().strip(';').strip('"')
                    if not content:
                        continue
                    parts = content.split(',')
                    if len(parts) < 10:
                        continue
                    symbol = left.split('_')[-1]
                    name = parts[0] or dict(MAIN_ETF_CODES).get(symbol, symbol)
                    try:
                        price = float(parts[3])
                        preclose = float(parts[2])
                        pct = (price - preclose) / preclose * 100 if preclose > 0 else 0.0
                        volume = float(parts[8])
                        amount = float(parts[9])
                    except (ValueError, IndexError):
                        continue
                    rows.append({
                        'symbol': symbol,
                        'code': symbol[2:],  # 去除 sh/sz 前缀
                        'name': name,
                        'trade': price,
                        'changepercent': round(pct, 2),
                        'volume': volume,
                        'amount': amount,
                        'turnoverratio': 0.0,
                        'per': 0.0,
                        'pb': 0.0,
                    })
        except Exception as e:
            log.debug(f'  ETF 查询失败: {e}')
        time.sleep(0.2)

    if rows:
        df = pd.DataFrame(rows).sort_values('changepercent', ascending=False, ignore_index=True)
        log.info(f'✅ 主流 ETF: {len(df)} 只')
        return df

    log.warning('ETF 数据拉取失败，返回空表')
    return pd.DataFrame()


# 基于个股的行业分组（从全量 A 股数据中用关键词归类）
SECTOR_KEYWORDS = {
    'AI/半导体': ['芯片', '半导体', '存储', '集成电路', '电子元器件', '封测', '光刻机', '射频', '功率'],
    'AI算力/通信': ['算力', '服务器', '数据中心', '液冷', '光模块', '通信', '5G', '交换机', '光通信'],
    '软件/信息': ['软件', '云计算', '信创', '信息安全', '操作系统', '数据库', '互联网', '数字经济'],
    '新能源车': ['特斯拉', '新能源汽车', '锂矿', '锂业', '电池', '锂电', '电解液', '正极', '负极', '隔膜', '锂矿', '盐湖'],
    '光伏/风电': ['光伏', '太阳能', '风电', '风电设备', '氢能源', '储能', '逆变器'],
    '白酒/食品': ['白酒', '酒业', '茅台', '五粮液', '啤酒', '乳业', '食品', '调味品', '预制菜', '农业'],
    '医药/创新药': ['制药', '药业', '生物', '创新药', '医药', '医疗', '医院', '疫苗', '中药', '检测'],
    '军工': ['军工', '防务', '航空', '航天', '无人机', '导弹', '卫星', '船舶', '海工'],
    '银行/保险': ['银行', '证券', '保险', '信托', '期货', '金控'],
    '房地产/建材': ['地产', '房产', '置业', '建设', '建材', '水泥', '玻璃', '防水', '家居'],
    '能源/资源': ['煤', '石油', '石化', '天然气', '油气', '铜业', '铝业', '黄金', '稀土', '有色'],
    '消费/零售': ['商贸', '零售', '百货', '超市', '电商', '旅游', '酒店', '餐饮', '美容', '化妆品'],
    '汽车/零部件': ['汽车', '乘用车', '零部件', '汽配', '整车', '特斯拉'],
    '机械/装备': ['机械', '重工', '机器', '装备', '工程', '机床', '工业'],
    '家电/轻工': ['家电', '电器', '照明', '家居', '包装', '造纸', '家具'],
}


def _analyze_sectors_from_spot(spot_df: pd.DataFrame) -> pd.DataFrame:
    """从个股实时行情中推断行业板块强弱"""
    df = spot_df.copy()
    df['pct'] = pd.to_numeric(df.get('changepercent', 0), errors='coerce').fillna(0.0)
    df['amount'] = pd.to_numeric(df.get('amount', 0), errors='coerce').fillna(0.0)

    rows = []
    for sector, keywords in SECTOR_KEYWORDS.items():
        mask = df['name'].astype(str).apply(
            lambda n: any(k in n for k in keywords)
        )
        sub = df[mask]
        if sub.empty or len(sub) < 3:
            continue
        avg_pct = float(sub['pct'].mean())
        up_ratio = float((sub['pct'] > 0).sum()) / len(sub)
        total_amount = float(sub['amount'].sum())
        top = sub.nlargest(5, 'pct')
        leaders = '、'.join(f"{r['name']}({r['pct']:+.1f}%)" for _, r in top.iterrows())
        rows.append({
            'sector': sector,
            'count': int(len(sub)),
            'avg_pct': round(avg_pct, 2),
            'up_ratio': round(up_ratio * 100, 1),
            'amount_yi': round(total_amount / 1e8, 1),
            'leaders': leaders,
        })
    result = pd.DataFrame(rows).sort_values('avg_pct', ascending=False, ignore_index=True)
    return result


def fetch_industry_boards(spot_df: pd.DataFrame) -> pd.DataFrame:
    """行业板块（通过个股关键词分组生成）"""
    log.info('🌋 从全量个股数据推断行业板块强弱...')
    df = _analyze_sectors_from_spot(spot_df)
    if df.empty:
        log.warning('未能推断出有效行业板块')
    else:
        log.info(f'✅ 板块数量: {len(df)}')
    return df


def fetch_index_hist(index_code: str, limit: int = 120) -> pd.DataFrame:
    """获取指数历史 K 线 — 使用新浪历史接口"""
    url = (
        f'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/'
        f'CN_MarketData.getKLineData?symbol={index_code}&scale=240&ma=no&datalen={limit}'
    )
    data = _json_get(url, timeout=15, retries=3)
    if data and isinstance(data, list):
        return pd.DataFrame(data)
    log.warning(f'指数历史 {index_code} 获取失败')
    return pd.DataFrame()


def fetch_northbound_flow() -> float:
    """北向资金净流入 (亿元) — 使用东财接口"""
    try:
        # 北向资金净流入: 沪股通净流入 + 深股通净流入
        url = (
            f'https://push2.eastmoney.com/api/qt/kamt/get?fields1=f1,f2,f3'
            f'&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64'
        )
        data = _json_get(url, timeout=15, retries=3)
        if data and isinstance(data, dict):
            d = data.get('data', {}) or {}
            # 当日净流入
            flow = 0.0
            for key in ['hk2szNetFlowIn', 'hk2shNetFlowIn', 'net_flow_in', 'netflowIn']:
                if key in d and d[key] not in ('-', '', None):
                    try:
                        flow += float(d[key]) / 1e8
                    except (TypeError, ValueError):
                        pass
            return flow
    except Exception as e:
        log.debug(f'北向获取失败: {e}')
    return 0.0


# ============================================================
# 2. 分析模块
# ============================================================
def _safe_float(v) -> float:
    try:
        if v is None or v == '-' or v == '':
            return 0.0
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def analyze_market(spot_df: pd.DataFrame) -> str:
    """市场宏观分析"""
    log.info('📊 进行市场宏观分析...')

    pct = spot_df.get('changepercent')
    if pct is None:
        pct = spot_df.get('pct_chg', pd.Series([0.0]))
    pct_numeric = pd.to_numeric(pct, errors='coerce').fillna(0.0)

    up_count = int((pct_numeric > 0).sum())
    down_count = int((pct_numeric < 0).sum())
    flat_count = int((pct_numeric == 0).sum())
    limit_up = int((pct_numeric >= 9.5).sum())
    limit_down = int((pct_numeric <= -9.5).sum())

    total_amount = spot_df.get('amount', pd.Series([0]))
    total_amount = pd.to_numeric(total_amount, errors='coerce').fillna(0.0)
    total_amount_yi = float(total_amount.sum() / 1e8)

    # 上证指数: sh000001
    idx = fetch_index_hist('sh000001', limit=30)
    idx_close = idx_pct = 0.0
    if not idx.empty and len(idx) >= 2:
        try:
            idx_close = _safe_float(idx.iloc[-1].get('close', 0))
            prev_close = _safe_float(idx.iloc[-2].get('close', 0))
            if prev_close > 0:
                idx_pct = (idx_close - prev_close) / prev_close * 100
        except Exception:
            pass

    north = fetch_northbound_flow()

    # 综合判定
    if idx_pct > 1.0 and limit_up > limit_down * 2:
        state = '🟢 强势拉升，情绪亢奋'
    elif idx_pct > 0.3:
        state = '🟡 震荡偏强，结构性机会'
    elif idx_pct > -0.5:
        state = '🟠 震荡偏弱，防御为主'
    else:
        state = '🔴 风险释放，谨慎观望'

    # 北向描述
    if north > 20:
        north_msg = f'🌊 北向 +{north:.0f}亿 (大举流入)'
    elif north > 5:
        north_msg = f'🌊 北向 +{north:.0f}亿 (温和净流入)'
    elif north < -20:
        north_msg = f'❄️ 北向 {north:.0f}亿 (大幅流出)'
    elif north < -5:
        north_msg = f'❄️ 北向 {north:.0f}亿 (温和净流出)'
    elif north != 0:
        north_msg = f'⚖️ 北向 {north:+.1f}亿 (基本平衡)'
    else:
        north_msg = '⚠️ 北向数据暂缺'

    # 仓位建议
    if idx_pct > 0.5 and north > 0:
        advice = '维持 **中高仓位 (5-7 成)**，聚焦强势主线，避免追高。'
    elif idx_pct > -0.5:
        advice = '维持 **中性仓位 (3-5 成)**，快进快出，严格止盈止损。'
    else:
        advice = '**降仓 (2 成以内)** 或空仓防守，优先保护本金。'

    return (
        f'**💠 市场快照**\n'
        f'- 上证指数：`{idx_close:.2f}` (今日 **{idx_pct:+.2f}%**)\n'
        f'- 综合判定：{state}\n'
        f'- 市场广度：红盘 `{up_count}` / 绿盘 `{down_count}` / 平盘 `{flat_count}`'
        f' (涨停 `{limit_up}` / 跌停 `{limit_down}`)\n'
        f'- 两市成交额：约 `{total_amount_yi:.0f}` 亿元\n'
        f'- {north_msg}\n\n'
        f'**💡 仓位建议**：{advice}'
    )


def analyze_signals(spot_df: pd.DataFrame) -> str:
    """股票信号分析"""
    log.info('🎯 筛选代表性股票信号...')
    if spot_df.empty:
        return '（今日无行情数据）'

    # 字段归一化
    work = pd.DataFrame({
        'code': spot_df.get('code', spot_df.get('symbol', '')).astype(str).str[-6:],
        'name': spot_df.get('name', '').astype(str),
        'pct_chg': pd.to_numeric(spot_df.get('changepercent', spot_df.get('pct_chg', 0)), errors='coerce').fillna(0.0),
        'price': pd.to_numeric(spot_df.get('trade', spot_df.get('price', 0)), errors='coerce').fillna(0.0),
        'amount': pd.to_numeric(spot_df.get('amount', 0), errors='coerce').fillna(0.0),
        'volume': pd.to_numeric(spot_df.get('volume', 0), errors='coerce').fillna(0.0),
        'turnover': pd.to_numeric(spot_df.get('turnoverratio', spot_df.get('turnover', 0)), errors='coerce').fillna(0.0),
        'pe': pd.to_numeric(spot_df.get('per', spot_df.get('pe', 0)), errors='coerce').fillna(0.0),
    })

    work['amount_yi'] = work['amount'] / 1e8

    def _render(subset: pd.DataFrame, limit: int = 5) -> str:
        subset = subset.sort_values('amount_yi', ascending=False).head(limit)
        lines = []
        for _, r in subset.iterrows():
            amt_str = f"{r['amount_yi']:.0f}亿" if r['amount_yi'] > 0 else "数据缺"
            turn_str = f"{r['turnover']:.1f}%" if r['turnover'] > 0 else "—"
            lines.append(
                f"- **{r['name']}** (`{r['code']}`) · `¥{r['price']:.2f}` · "
                f"**{r['pct_chg']:+.2f}%** · 成交 `{amt_str}` · 换手 `{turn_str}`"
            )
        return '\n'.join(lines)

    parts = ['**🎯 代表性股票信号**\n']

    # 涨停强龙
    strong = work[work['pct_chg'] >= 9.5]
    if not strong.empty:
        parts.append('**🟢 涨停强龙 (情绪龙头)**')
        parts.append(_render(strong, 8))
        parts.append('')

    # 右侧强势 (3% - 9.5%)
    right = work[(work['pct_chg'] >= 3) & (work['pct_chg'] < 9.5)]
    if not right.empty:
        parts.append('**🔥 右侧爆发股 (强者恒强)**')
        parts.append(_render(right, 6))
        parts.append('')

    # 稳健型: 0-3%
    robust = work[(work['pct_chg'] > 0) & (work['pct_chg'] < 3)]
    if not robust.empty:
        parts.append('**🛡️ 稳健型 (资金涌入、波动可控)**')
        parts.append(_render(robust, 5))
        parts.append('')

    # 超跌反弹候选
    oversold = work[work['pct_chg'] < -5]
    if not oversold.empty:
        parts.append('**⚠️ 异动超跌榜 (潜在反抽候选，风险高！)**')
        oversold_asc = oversold.sort_values('amount_yi', ascending=False).head(5)
        parts.append(_render(oversold_asc, 5))
        parts.append('')

    if len(parts) <= 1:
        return '（今日无明显信号）'

    parts.append(
        '> ⚠️ **免责声明**：本清单仅基于当日行情做结构化呈现，不构成任何投资建议。'
        '投资有风险，交易请基于自身研究与风险承受能力。\n'
    )
    return '\n'.join(parts)


def analyze_etf_rotation(etf_df: pd.DataFrame) -> str:
    """ETF 板块轮动分析"""
    log.info('🔄 分析 ETF 轮动...')
    if etf_df.empty:
        return '（今日无 ETF 数据）'

    work = pd.DataFrame({
        'code': etf_df.get('code', etf_df.get('symbol', '')).astype(str),
        'name': etf_df.get('name', '').astype(str),
        'pct_chg': pd.to_numeric(etf_df.get('changepercent', 0), errors='coerce').fillna(0.0),
        'price': pd.to_numeric(etf_df.get('trade', etf_df.get('price', 0)), errors='coerce').fillna(0.0),
        'amount': pd.to_numeric(etf_df.get('amount', 0), errors='coerce').fillna(0.0),
    })
    work['amount_yi'] = work['amount'] / 1e8
    work = work.dropna(subset=['pct_chg'])

    # 区分宽基 vs 行业
    broad_keywords = ['沪深300', '中证500', '中证1000', '中证2000', '创业板', '科创', '上证50', 'MSCI', '科创50', '大盘', '中小盘', '中证', 'A股', '红利ETF', '恒生']
    is_broad = work['name'].apply(lambda n: any(k in n for k in broad_keywords))
    broad = work[(is_broad) & (work['amount_yi'] > 0.5)].sort_values('pct_chg', ascending=False)
    industry = work[(~is_broad) & (work['amount_yi'] > 0.3)].sort_values('pct_chg', ascending=False)

    def _render(subset: pd.DataFrame, limit: int = 8) -> str:
        lines = []
        for _, r in subset.head(limit).iterrows():
            pct = float(r['pct_chg'])
            emoji = '🟢' if pct > 2 else ('🟡' if pct > 0 else ('🔴' if pct < -2 else '⚫'))
            lines.append(
                f"- {emoji} **{r['name']}** (`{r['code']}`) · "
                f"`¥{r['price']:.3f}` · **{pct:+.2f}%** · 成交 `{r['amount_yi']:.1f}亿`"
            )
        return '\n'.join(lines)

    parts = ['**🔄 ETF 板块轮动**\n']
    if not broad.empty:
        parts.append('**📊 宽基指数 ETF**')
        parts.append(_render(broad, 10))
        parts.append('')
    if not industry.empty:
        parts.append('**🥇 领涨行业 ETF TOP 8**')
        parts.append(_render(industry.head(8), 8))
        parts.append('')
        parts.append('**🥉 领跌行业 ETF (规避区)**')
        parts.append(_render(industry.tail(5).iloc[::-1], 5))
        parts.append('')

    if len(parts) <= 1:
        return '（今日无可用 ETF 数据）'
    return '\n'.join(parts)


def analyze_hot_sectors(board_df: pd.DataFrame, spot_df: pd.DataFrame) -> str:
    """行业热点分析（board_df 列: sector,count,avg_pct,up_ratio,amount_yi,leaders）"""
    log.info('🌋 分析行业热点...')
    if board_df is None or board_df.empty:
        return '（今日无行业板块数据，可参考上方个股信号）'

    parts = ['**🌋 行业热点雷达**\n']
    top = board_df.head(10)
    bottom = board_df.tail(5)

    parts.append('**🥇 今日主线板块 (Top 10)**')
    for _, row in top.iterrows():
        pct = float(row['avg_pct'])
        emoji = '🟢' if pct > 2 else ('🟡' if pct > 0 else ('🔴' if pct < -2 else '⚫'))
        parts.append(
            f"- {emoji} **{row['sector']}** · 成分股 `{int(row['count'])}` 只 · "
            f"**{pct:+.2f}%** · 上涨家数占比 `{float(row['up_ratio']):.0f}%` · 板块成交 `{float(row['amount_yi']):.0f}` 亿"
        )
        parts.append(f"  · 代表龙头: {row['leaders']}")

    if not bottom.empty and len(board_df) > 10:
        parts.append('')
        parts.append('**🥉 今日最弱板块 (规避区)**')
        for _, row in bottom.iloc[::-1].iterrows():
            pct = float(row['avg_pct'])
            parts.append(
                f"- 🔴 **{row['sector']}** · 成分股 `{int(row['count'])}` 只 · "
                f"**{pct:+.2f}%** · 成交 `{float(row['amount_yi']):.0f}` 亿"
            )

    parts.append('')
    return '\n'.join(parts)


def summarize_strategy(market_msg: str, top_signals: str, board_df: pd.DataFrame) -> str:
    """策略摘要"""
    log.info('💡 汇总策略摘要...')
    hottest_names = []
    if board_df is not None and not board_df.empty:
        for _, r in board_df.head(3).iterrows():
            pct = float(r['avg_pct'])
            hottest_names.append(f"**{r['sector']} ({pct:+.2f}%)**")

    lines = ['**💡 今日策略摘要**\n']
    if hottest_names:
        lines.append('- 主线方向：关注 ' + '、'.join(hottest_names) + '，以 **跟随趋势** 为主，不做逆势抄底')
    else:
        lines.append('- 主线方向：关注当日强势板块，以 **跟随趋势** 为主，不做逆势抄底')
    lines.append('- 选股风格：优先 **高成交 + 正斜率** 的龙头股，避免缩量阴跌的弱势品种')
    lines.append('- 交易纪律：单票仓位不超过 15%，总仓位不超过 7 成，严格止损')
    lines.append('- 关注点：明日盘前关注外围消息、汇率、北向资金开盘动向')
    lines.append('')
    lines.append('> 本文由 AI 量化投研系统自动生成，仅供复盘研究，不构成投资建议。')
    return '\n'.join(lines)


# ============================================================
# 3. 推送网关 (钉钉 / 飞书)
# ============================================================
def push_to_bots(title: str, content: str) -> None:
    ding_url = os.environ.get('DINGTALK_WEBHOOK', '').strip()
    feishu_url = os.environ.get('FEISHU_WEBHOOK', '').strip()
    sec_keyword = os.environ.get('NOTIFY_SEC_KEYWORD', 'AI量化').strip()

    targets = []
    if ding_url:
        targets.append((ding_url, False, '钉钉'))
    if feishu_url:
        targets.append((feishu_url, True, '飞书'))
    if not targets:
        log.warning('⚠️ 未配置任何 Webhook 环境变量 (DINGTALK_WEBHOOK / FEISHU_WEBHOOK), 仅在控制台输出。')
        return

    # 分块推送, 避免超长消息被拒
    CHUNK = 18000
    chunks = [content[i:i + CHUNK] for i in range(0, len(content), CHUNK)]
    if len(chunks) > 5:
        chunks = chunks[:5]
        chunks[-1] += '\n\n> ⚠️ (内容超出承载极限, 尾部已截断)'

    for idx, chunk in enumerate(chunks):
        piece_title = title if len(chunks) == 1 else f'{title} (Part {idx + 1}/{len(chunks)})'
        for url, is_feishu, name in targets:
            ok = False
            for attempt in range(3):
                try:
                    if is_feishu:
                        payload = {
                            'msg_type': 'interactive',
                            'card': {
                                'config': {'wide_screen_mode': True},
                                'header': {
                                    'title': {'tag': 'plain_text', 'content': piece_title},
                                    'template': 'blue',
                                },
                                'elements': [{'tag': 'markdown', 'content': chunk}],
                            },
                        }
                    else:
                        final_title = piece_title if sec_keyword in piece_title else f'{sec_keyword} | {piece_title}'
                        final_text = chunk if sec_keyword in chunk else f'### {sec_keyword}\n\n{chunk}'
                        payload = {
                            'msgtype': 'markdown',
                            'markdown': {
                                'title': final_title,
                                'text': final_text,
                            },
                        }
                    r = requests.post(url, json=payload, timeout=15, headers={'User-Agent': UA})
                    if r.status_code == 200:
                        resp = r.json()
                        if is_feishu and resp.get('code', 0) == 0:
                            ok = True; break
                        if not is_feishu and resp.get('errcode', 0) == 0:
                            ok = True; break
                        log.warning(f'{name} server err: {resp}')
                    else:
                        log.warning(f'{name} HTTP {r.status_code}: {r.text[:200]}')
                except Exception as e:
                    log.warning(f'{name} 推送异常: {e}')
                time.sleep(1.0 * (attempt + 1))
            log.info(f'✅ {name} 推送成功 (第 {idx+1}/{len(chunks)} 块)' if ok else f'❌ {name} 推送失败 (第 {idx+1}/{len(chunks)} 块)')
        if idx < len(chunks) - 1:
            time.sleep(0.5)


# ============================================================
# 4. 主调度
# ============================================================
def build_briefing_body(now_str: str, market_msg: str, signal_msg: str, etf_msg: str,
                        sector_msg: str, strategy_msg: str) -> str:
    body = (
        f'## 📘 每日投研简报\n'
        f'> **{now_str}** (自动生成，仅供研究参考)\n\n'
        f'{market_msg}\n\n'
        f'---\n\n'
        f'{signal_msg}\n\n'
        f'---\n\n'
        f'{etf_msg}\n\n'
        f'---\n\n'
        f'{sector_msg}\n\n'
        f'---\n\n'
        f'{strategy_msg}\n'
    )
    return body


def main():
    now_str = datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M')
    log.info('=' * 60)
    log.info(f'🚀 每日投研简报任务启动 @ {now_str}')
    log.info(f'   http_proxy={os.environ.get("http_proxy","<unset>")}')
    log.info(f'   https_proxy={os.environ.get("https_proxy","<unset>")}')
    log.info('=' * 60)

    # 主数据: A 股行情
    try:
        spot_df = fetch_spot()
    except Exception as e:
        log.error(f'❌ A 股数据采集失败: {e}')
        traceback.print_exc()
        push_to_bots(
            f'【AI量化】每日投研简报 - {now_str} (采集失败)',
            f'**⚠️ 今日行情数据采集异常**\n\n错误信息：`{e}`\n\n请稍后手动重试或关注后续推送。',
        )
        sys.exit(1)

    # 辅助数据: ETF + 行业板块
    try:
        etf_df = fetch_etf_spot()
    except Exception as e:
        log.warning(f'ETF 数据采集失败: {e}')
        etf_df = pd.DataFrame()

    try:
        board_df = fetch_industry_boards(spot_df)
    except Exception as e:
        log.warning(f'行业板块数据采集失败: {e}')
        board_df = pd.DataFrame()

    # 分析
    try:
        market_msg = analyze_market(spot_df)
        signal_msg = analyze_signals(spot_df)
        etf_msg = analyze_etf_rotation(etf_df)
        sector_msg = analyze_hot_sectors(board_df, spot_df)
        strategy_msg = summarize_strategy(market_msg, signal_msg, board_df)
    except Exception as e:
        log.error(f'❌ 分析模块异常: {e}')
        traceback.print_exc()
        market_msg = f'**⚠️ 分析异常: {e}**'
        signal_msg = etf_msg = sector_msg = strategy_msg = ''

    title = f'【AI量化】每日投研简报 - {now_str}'
    body = build_briefing_body(now_str, market_msg, signal_msg, etf_msg, sector_msg, strategy_msg)

    log.info('📤 推送简报至钉钉 / 飞书 ...')
    push_to_bots(title, body)

    log.info('=' * 60)
    print(body)
    log.info('=' * 60)
    log.info('✅ 每日投研简报任务完成')


if __name__ == '__main__':
    main()
