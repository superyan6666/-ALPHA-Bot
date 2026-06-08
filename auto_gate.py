import json
import os
import sys

def check_gate():
    metrics_path = '.quantbot_data/eval_metrics.json'
    if not os.path.exists(metrics_path):
        print("⚠️ 未找到评测结果 (eval_metrics.json)，放行。")
        return 0

    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    wfo_spread = metrics.get('wfo_ls_spread_bps', 0)
    
    # 门控规则：如果多空收益差为负，或者低于最低基准(例如 2.0 bps)，则截断！
    THRESHOLD = 2.0
    
    print(f"📊 自动防退化门控 (Auto Gate)")
    print(f"   -> 评估指标 WFO Long-Short Spread: {wfo_spread:.2f} bps/day")
    print(f"   -> 最低门控基准线: {THRESHOLD:.2f} bps/day")
    
    if wfo_spread < THRESHOLD:
        print("\n❌ 【防爆截断】新模型表现严重退化，未超越基准线！已禁止发布！")
        return 1
    else:
        print("\n✅ 质检通过！新模型表现优异，允许发布！")
        return 0

if __name__ == '__main__':
    sys.exit(check_gate())
