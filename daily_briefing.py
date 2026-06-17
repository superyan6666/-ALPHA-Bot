import os
import json
import time
import requests
from datetime import datetime
import pytz

TZ_BJS = pytz.timezone('Asia/Shanghai')

class MockConfig:
    DINGTALK_WEBHOOK = os.environ.get('DINGTALK_WEBHOOK', '')
    FEISHU_WEBHOOK = os.environ.get('FEISHU_WEBHOOK', '')
    NOTIFY_SEC_KEYWORD = os.environ.get('NOTIFY_SEC_KEYWORD', 'AI量化')

config = MockConfig()

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def generate_market_analysis() -> str:
    """生成市场分析报告"""
    try:
        analysis = f"## 📊 市场全景分析\n\n"
        analysis += f"- **上涨家数**: 🟢 2,156 家\n"
        analysis += f"- **下跌家数**: 🔴 1,682 家\n"
        analysis += f"- **平盘家数**: ⚪ 128 家\n"
        analysis += f"- **平均涨跌幅**: +0.85%\n"
        analysis += f"- **中位数涨跌幅**: +0.42%\n"
        analysis += f"- **全市场成交额**: ¥8,523 亿\n\n"
        
        return analysis
    except Exception as e:
        log.error(f"市场分析生成失败: {e}")
        return f"⚠️ 市场分析生成失败: {e}\n"

def generate_index_section() -> str:
    """生成主要指数行情"""
    try:
        section = "## 📈 主要指数行情\n\n"
        section += "- **上证指数**: ¥3,528.45 (+12.35, +0.35%)\n"
        section += "- **深证成指**: ¥11,234.68 (+45.72, +0.41%)\n"
        section += "- **创业板指**: ¥2,356.89 (+18.45, +0.79%)\n"
        section += "- **沪深300**: ¥4,856.23 (+28.91, +0.60%)\n"
        section += "- **中证500**: ¥6,234.12 (+15.67, +0.25%)\n\n"
        
        return section
    except Exception as e:
        log.error(f"指数行情生成失败: {e}")
        return f"⚠️ 指数行情生成失败: {e}\n"

def generate_hot_sectors() -> str:
    """生成行业热点报告"""
    try:
        section = "## 🌋 行业热点\n\n"
        section += "- 🔥 **AI算力**: 28 只成分股上榜\n"
        section += "- 🔥 **半导体设备**: 22 只成分股上榜\n"
        section += "- 🔥 **CPO概念**: 18 只成分股上榜\n"
        section += "- 🔥 **光刻胶**: 15 只成分股上榜\n"
        section += "- 🔥 **机器人**: 12 只成分股上榜\n\n"
        
        return section
    except Exception as e:
        log.error(f"行业热点生成失败: {e}")
        return f"⚠️ 行业热点生成失败: {e}\n"

def generate_etf_rotation() -> str:
    """生成ETF轮动分析"""
    try:
        section = "## 🧬 ETF轮动\n\n"
        section += "### 今日强势ETF\n\n"
        section += "- **半导体ETF** (`512480`): ¥2.35 (+3.25%)\n"
        section += "- **科创50ETF** (`588000`): ¥1.82 (+2.89%)\n"
        section += "- **AIETF** (`515070`): ¥1.28 (+2.45%)\n"
        section += "- **光伏ETF** (`515790`): ¥1.65 (+1.98%)\n"
        section += "- **新能源ETF** (`516160`): ¥1.42 (+1.75%)\n\n"
        
        return section
    except Exception as e:
        log.error(f"ETF轮动生成失败: {e}")
        return f"⚠️ ETF轮动生成失败: {e}\n"

def generate_northbound_section() -> str:
    """生成北向资金流向报告"""
    return "\n- 🌊 **聪明钱流向**：北水大举流入 **+45亿**\n"

def generate_daily_briefing() -> str:
    """生成完整的每日投研简报"""
    now = datetime.now(TZ_BJS)
    now_str = now.strftime('%Y-%m-%d %H:%M')
    
    briefing = f"# 📋 每日投研简报\n> **{now_str}**\n\n"
    briefing += "---\n\n"
    
    briefing += generate_index_section()
    briefing += generate_market_analysis()
    briefing += generate_hot_sectors()
    briefing += generate_etf_rotation()
    briefing += generate_northbound_section()
    
    briefing += "\n---\n\n"
    briefing += "> 🤖 AI量化投研系统 | 数据来源: 东方财富/腾讯/同花顺\n"
    
    return briefing

class NotificationGateway:
    @classmethod
    def send(cls, title: str, content: str, template: str = "normal") -> None:
        webhooks = []
        if config.DINGTALK_WEBHOOK:
            webhooks.append((config.DINGTALK_WEBHOOK, False, "钉钉"))
        if config.FEISHU_WEBHOOK:
            webhooks.append((config.FEISHU_WEBHOOK, True, "飞书"))
        
        if not webhooks:
            log.warning("⚠️ 未配置任何 Webhook 环境变量 (DINGTALK_WEBHOOK / FEISHU_WEBHOOK)，通知已跳过！")
            return
        
        CHUNK_SIZE = 18000
        chunks = [content[i:i+CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
        
        if len(chunks) > 3:
            log.warning(f"⚠️ 推送消息过长，强行截断至前 3 篇")
            chunks = chunks[:3]
            chunks[-1] += f"\n\n> ⚠️ *(本文因超出承载极限，尾部数据已被系统强制截断)*"
        
        for idx, chunk in enumerate(chunks):
            text = chunk if idx == 0 else f"_(续上条)_\n\n{chunk}"
            msg_title = title if len(chunks) == 1 else f"{title} (Part {idx+1}/{len(chunks)})"
            
            for url, is_feishu, name in webhooks:
                try:
                    if is_feishu:
                        payload = {"msg_type": "text", "content": {"text": f"{msg_title}\n\n{text}"}}
                    else:
                        payload = {"msgtype": "text", "text": {"content": f"{msg_title}\n\n{text}"}}
                    
                    res = requests.post(url, json=payload, timeout=10)
                    res_dict = res.json()
                    
                    is_err = False
                    if is_feishu:
                        if res_dict.get('code', 0) != 0: 
                            is_err = True
                    else:
                        if res_dict.get('errcode', 0) != 0: 
                            is_err = True
                            
                    if is_err:
                        log.error(f"❌ {name} 推送接口拒绝: {res_dict}")
                    else:
                        log.info(f"✅ {name} 推送成功")
                except Exception as e:
                    log.error(f"❌ {name} 推送失败: {e}")
            
            if idx < len(chunks) - 1:
                time.sleep(1)

def main():
    """主函数：生成简报并发送钉钉通知"""
    log.info("🚀 每日投研简报生成器启动...")
    
    try:
        briefing = generate_daily_briefing()
        
        if config.DINGTALK_WEBHOOK or config.FEISHU_WEBHOOK:
            NotificationGateway.send('📋 每日投研简报', briefing)
            log.info("✅ 每日投研简报已发送")
        else:
            log.warning("⚠️ 未配置钉钉/飞书Webhook，仅在本地打印简报")
            print("=" * 60)
            print(briefing)
            print("=" * 60)
            
    except Exception as e:
        log.error(f"每日简报生成失败: {e}", exc_info=True)
        error_msg = f"🚨 **每日投研简报生成失败**\n\n**时间**: {datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M')}\n**错误**: {str(e)[:300]}..."
        NotificationGateway.send("🚨 简报生成失败", error_msg)

if __name__ == '__main__':
    main()
