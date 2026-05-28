#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
每日投研简报生成器
包含：市场分析、股票信号、ETF轮动、行业热点等内容
并发送到钉钉通知
"""

import os
import sys

# 设置环境变量，确保使用正常模式运行
os.environ.setdefault('RUN_MODE', 'normal')
os.environ.setdefault('GITHUB_EVENT_NAME', 'workflow_dispatch')  # 设置为手动模式，允许在任何时间运行

def generate_briefing_content(sigs, watch, pool_size, total_mkt, m_msg):
    """生成简报内容"""
    from main import _today_str
    
    now_ts = _today_str()
    header = (
        f"## 🤖 AI量化保姆级盘后总结\n"
        f"> **{now_ts}**\n>\n"
        f"> ⚠️ **郑重声明**：本报告由量化模型自动生成，仅供技术交流与策略复盘，**绝不构成任何投资建议**。股市有风险，入市需谨慎，盈亏请自负。\n\n"
    )
    
    pass_rate = len(sigs) / max(pool_size, 1) * 100 if pool_size > 0 else 0
    header += f"**🔬 漏斗数据**：全市场白名单 `{total_mkt}` 只，异动提取 `{pool_size}` 只，完美过线 `{len(sigs)}` 只 (B+级以上优选率 **{pass_rate:.1f}%**)\n\n"
    
    if m_msg:
        header += f"{m_msg}\n\n---\n\n"
    
    content = header
    
    if sigs:
        MAX_DISPLAY = 5
        display_signals = sigs[:MAX_DISPLAY]
        hidden_count = len(sigs) - len(display_signals)
        
        avg_score = sum(s.score for s in display_signals) / len(display_signals)
        quality_tag = "🥇 **绝佳** (建议严格按剧本执行)" if avg_score >= 80 \
            else "🥈 **尚可** (建议严格限价，减半仓位)"
            
        content += f"### 📈 今日核心精选 (Top 5)\n**精选均分：{avg_score:.0f} 分** | {quality_tag}\n\n"
        
        cold_gate = (
            "> **🛑 买入前冷静自检（30秒）**\n"
            "> 1. 这笔闲钱 **3年内** 绝对不会急用？\n"
            "> 2. 就算不小心 **亏掉30%** 也不会睡不着？\n"
            "> 3. 能管住手，**绝不因为下跌反复盯盘**？\n"
            "> \n"
            "> *✅ 三项全对 ➡️ 允许按下方计划执行*\n"
            "> *❌ 有一项不对 ➡️ 请立即把买入预算砍掉一半！*\n\n"
            "---\n\n"
        )
        content += cold_gate
        
        for s in display_signals:
            warn_msg = "> ⚡ **【风险警示】** 该股为创业板(波动±20%)，心脏不好请务必**缩减仓位**！\n\n" if str(s.code).startswith('300') else ""
            
            content += (
                f"#### 🎯 {s.name} (`{s.code}`)\n"
                f"{warn_msg}"
                f"- **综合评级**：`{s.score}` 分 {s.level}\n"
                f"- **今日收盘**：`¥{s.price}` ({s.pct_chg})\n\n"
                f"**💡 为什么机器选出它？**\n{s.reasons}\n\n"
                f"**🛡️ 小白专属操作剧本**\n"
                f"{s.hold_period_msg}\n"
                f"{s.money_risk_msg}\n\n"
                f"{s.tranche_plan_msg}\n\n"
                f"{s.plan_b_msg}\n\n"
                f"> **纪律红线 (V11.0 吊灯止损架构)**\n"
                f"> 🛡️ **防震仓锁 (T+5)**：建仓后 5 个交易日内，只要未跌破死线 `¥{s.stop_loss}`，无论怎么洗盘坚决死拿！\n"
                f"> 🎯 **动态止盈 (吊灯)**：创出新高后，以最高价回撤 2.5~3 倍 ATR 为动态离场线，彻底废除均线退出法。\n"
                f"> 🚫 **防空防守**：明日开盘直接高开 **> 4%** 说明资金抢跑，直接放弃，绝不追高！\n\n"
                "---\n\n"
            )
        
        if hidden_count > 0:
            hidden_names = "、".join([f"{s.name}(`{s.code}` **{s.score}分**)" for s in sigs[MAX_DISPLAY:]])
            content += f"\n\n*⚠️ 受限于篇幅，以下 **{hidden_count} 只** 达标个股被系统折叠（已按分数排序）：*\n> {hidden_names}\n"
            
    else:
        content += "✅ 今日未发现 B+ 级以上核心机会，正式推荐列表空仓防守中。\n"
    
    if watch:
        watch_lines = "\n".join(
            f"- `{code}` **{name}** (¥{price}) 得分: **{score}**"
            for name, code, score, price in watch[:5]
        )
        content += (
            f"\n\n---\n### 👁️ 候补观察池（只看不买）\n"
            f"{watch_lines}\n\n"
            f"*注：以上标的评级不足 70 分，系统判断波动或风险偏大，暂不提供操作剧本。待其评级升至发车线后再考虑介入。*"
        )
    
    content += (
        "\n\n---\n### 🤔 每日灵魂拷问\n"
        "如果明天买入的股票跌了 5%，我会焦虑得睡不着觉吗？\n\n"
        "> **如果会，请把你准备买入的金额【再砍掉一半】！投资是为了生活更好，不是花钱找罪受。**"
    )
    
    return content

def main():
    from main import get_signals, send_dingtalk, save_pushed_state, config, log
    
    log.info("📊 每日投研简报生成器启动...")
    
    try:
        # 获取股票信号和市场信息
        sigs, watch, pushed, pool_size, m_msg, total_mkt = get_signals()
        
        # 生成简报内容
        content = generate_briefing_content(sigs, watch, pool_size, total_mkt, m_msg)
        
        # 打印到控制台
        print("\n" + "="*80)
        print("📊 每日投研简报")
        print("="*80)
        print(content)
        print("="*80 + "\n")
        
        # 如果配置了钉钉 Webhook，则发送通知
        if config.DINGTALK_WEBHOOK:
            # 发送钉钉通知
            send_dingtalk(sigs, watch, pool_size, total_mkt, m_msg)
            log.info("✅ 每日投研简报已发送到钉钉！")
        else:
            log.warning("⚠️ 未配置 DINGTALK_WEBHOOK，仅在控制台输出简报内容")
        
        # 保存推送状态
        if sigs:
            save_pushed_state(pushed)
        
        log.info("✅ 每日投研简报生成完成！")
        
    except Exception as e:
        log.critical(f"❌ 生成每日投研简报失败: {e}", exc_info=True)
        from main import NotificationGateway, _today_str
        error_msg = f"🚨 **每日投研简报生成失败**\n\n**时间**: {_today_str()}\n**异常信息**: {str(e)[:300]}..."
        
        if config.DINGTALK_WEBHOOK:
            NotificationGateway.send("🚨 每日投研简报告警", error_msg, template="red")
        
        sys.exit(1)

if __name__ == '__main__':
    main()
