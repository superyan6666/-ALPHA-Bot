#!/usr/bin/env python3
"""
每日投研简报生成器
整合市场分析、股票信号、ETF轮动、行业热点等内容，并发送到钉钉
"""

import os
import sys
import logging
from datetime import datetime
import pytz

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

TZ_BJS = pytz.timezone('Asia/Shanghai')

def generate_daily_briefing():
    """生成每日投研简报"""
    try:
        from main import (
            get_signals,
            generate_macro_section,
            NotificationGateway,
            config,
            DataProxy
        )
    except ImportError as e:
        log.error(f"导入模块失败: {e}")
        return False
    
    now = datetime.now(TZ_BJS)
    now_str = now.strftime('%Y-%m-%d %H:%M')
    
    log.info("=" * 60)
    log.info("🚀 每日投研简报生成器启动")
    log.info(f"📅 时间: {now_str}")
    log.info("=" * 60)
    
    # 1. 市场宏观分析
    log.info("\n[1/4] 正在生成宏观市场分析...")
    try:
        macro_content = generate_macro_section()
        log.info("✅ 宏观市场分析完成")
    except Exception as e:
        log.warning(f"⚠️ 宏观市场分析失败: {e}")
        macro_content = "⚠️ 宏观数据获取失败，请稍后重试"
    
    # 2. 股票信号
    log.info("\n[2/4] 正在扫描股票信号...")
    try:
        signals, watchlist, pushed, pool_size, market_msg, total_market = get_signals()
        total_signals = sum(len(sigs) for sigs in signals.values()) if signals else 0
        log.info(f"✅ 股票信号扫描完成: 发现 {total_signals} 个信号")
    except Exception as e:
        log.warning(f"⚠️ 股票信号扫描失败: {e}")
        signals, watchlist, market_msg = {}, [], ""
        total_signals, pool_size, total_market = 0, 0, 0
    
    # 3. 行业热点分析
    log.info("\n[3/4] 正在分析行业热点...")
    try:
        proxy = DataProxy()
        hot_sectors = proxy.get_hot_sectors()
        
        if hot_sectors:
            from collections import Counter
            sector_counts = Counter(hot_sectors.values())
            top_sectors = sector_counts.most_common(5)
            sector_content = "### 🔥 行业热点\n"
            for sector, count in top_sectors:
                sector_content += f"- **{sector}**：{count} 只股票入选\n"
            log.info(f"✅ 行业热点分析完成: {len(top_sectors)} 个热门板块")
        else:
            sector_content = "### 🔥 行业热点\n暂无热门板块数据"
            log.info("⚠️ 未获取到行业热点数据")
    except Exception as e:
        log.warning(f"⚠️ 行业热点分析失败: {e}")
        sector_content = "### 🔥 行业热点\n行业数据获取失败"
    
    # 4. ETF轮动信号
    log.info("\n[4/4] 正在生成ETF轮动建议...")
    try:
        etf_content = generate_etf_rotation_section()
        log.info("✅ ETF轮动建议生成完成")
    except Exception as e:
        log.warning(f"⚠️ ETF轮动建议生成失败: {e}")
        etf_content = "### 📈 ETF轮动建议\n暂无ETF轮动数据"
    
    # 构建完整简报
    briefing_content = f"""# 📊 每日投研简报

> **生成时间**: {now_str}

---

{macro_content}

---

{sector_content}

---

{etf_content}

---

## 📌 股票信号汇总

"""
    
    # 添加信号详情
    if signals and any(signals.values()):
        for category, sig_list in signals.items():
            if sig_list:
                briefing_content += f"\n### {category}\n"
                for sig in sig_list[:10]:  # 每类最多展示10个
                    try:
                        briefing_content += f"- **{sig.code}** ({sig.name}): ¥{sig.price:.2f} | 得分: {sig.score}\n"
                    except:
                        briefing_content += f"- {sig.code}\n"
        briefing_content += f"\n> 共发现 **{total_signals}** 个信号，候选池 **{pool_size}** 只股票\n"
    else:
        briefing_content += "\n✅ 今日未发现符合条件的股票信号\n"
    
    if market_msg:
        briefing_content += f"\n---\n\n### 📊 市场状态\n{market_msg}\n"
    
    # 发送到钉钉
    log.info("\n" + "=" * 60)
    log.info("📤 正在发送简报到钉钉...")
    
    if config.DINGTALK_WEBHOOK or config.FEISHU_WEBHOOK:
        try:
            NotificationGateway.send(
                "📊 每日投研简报",
                briefing_content,
                template="blue"
            )
            log.info("✅ 简报已成功发送!")
        except Exception as e:
            log.error(f"❌ 发送失败: {e}")
            return False
    else:
        log.warning("⚠️ 未配置 WEBHOOK，简报仅本地输出:")
        print("\n" + "=" * 60)
        print(briefing_content)
        print("=" * 60 + "\n")
    
    return True


def generate_etf_rotation_section() -> str:
    """生成ETF轮动建议"""
    import akshare as ak
    import socket
    
    # 设置请求超时
    socket.setdefaulttimeout(10)
    
    # 主要ETF列表
    etf_list = [
        ('510300', '沪深300ETF'),
        ('510500', '中证500ETF'),
        ('159915', '创业板ETF'),
        ('512880', '证券ETF'),
        ('512690', '酒ETF'),
        ('159766', '旅游ETF'),
        ('515790', '光伏ETF'),
        ('159996', '家电ETF'),
        ('512170', '医疗ETF'),
        ('515030', '新能源ETF'),
    ]
    
    etf_data = []
    for code, name in etf_list:
        try:
            df = ak.fund_etf_hist_sina(symbol=code)
            if df is not None and len(df) > 0:
                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
                pct_chg = ((latest['close'] - prev['close']) / prev['close'] * 100) if prev['close'] > 0 else 0
                
                # 计算5日均线
                if len(df) >= 5:
                    ma5 = df['close'].tail(5).mean()
                    trend = "↑" if latest['close'] > ma5 else "↓"
                else:
                    trend = "-"
                
                etf_data.append({
                    'code': code,
                    'name': name,
                    'price': latest['close'],
                    'pct_chg': pct_chg,
                    'trend': trend
                })
        except Exception:
            continue
    
    if not etf_data:
        return "### 📈 ETF轮动建议\n暂无ETF数据（网络连接超时）"
    
    # 排序：按涨幅
    etf_data.sort(key=lambda x: x['pct_chg'], reverse=True)
    
    content = "### 📈 ETF轮动建议\n\n| 代码 | 名称 | 最新价 | 涨跌幅 | 趋势 |\n|------|------|--------|--------|------|\n"
    for etf in etf_data[:8]:
        content += f"| {etf['code']} | {etf['name']} | ¥{etf['price']:.3f} | {etf['pct_chg']:+.2f}% | {etf['trend']} |\n"
    
    # 添加轮动建议
    top_etf = etf_data[0]
    content += f"\n> 💡 **轮动建议**: 当前强势板块为 **{top_etf['name']}**，建议关注相关ETF机会。\n"
    
    return content


if __name__ == '__main__':
    try:
        success = generate_daily_briefing()
        sys.exit(0 if success else 1)
    except Exception as e:
        log.error(f"❌ 简报生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)