
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
import json
from ml_engine import PyTorchDLModel
from feature_engine import build_ml_features
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def evaluate_pool(pool, pool_name):
    """
    计算特定股票池在 T+1, T+5, T+20 的平均未来收益 (bps)。
    """
    if len(pool) == 0:
        log.warning(f"[{pool_name}] 股票池为空！")
        return
        
    ret_1 = pool['fwd_ret_t1'].mean() * 10000
    ret_5 = pool['fwd_ret_t5'].mean() * 10000
    ret_20 = pool['fwd_ret_t20'].mean() * 10000
    
    log.info(f"[{pool_name}] 样本数: {len(pool):,d} | T+1 收益: {ret_1:6.2f} bps | T+5 收益: {ret_5:6.2f} bps | T+20 收益: {ret_20:6.2f} bps")


def run_veto_simulation():
    log.info("=" * 60)
    log.info("启动择时否决机制仿真 (Execution Veto Simulation)")
    log.info("=" * 60)
    
    # 1. 加载数据
    panel = pd.read_parquet('.quantbot_data/ashare_daily.parquet')
    if 'volume' in panel.columns and 'vol' not in panel.columns:
        panel.rename(columns={'volume': 'vol'}, inplace=True)
        
    panel['code'] = panel['code'].str.replace('sh.', '', regex=False)\
                                 .str.replace('sz.', '', regex=False)\
                                 .str.replace('bj.', '', regex=False)
    panel = panel.sort_values(['code', 'date'])
    panel['fwd_ret_t1'] = panel.groupby('code')['close'].shift(-1) / (panel.groupby('code')['open'].shift(-1) + 1e-5) - 1
    panel['fwd_ret_t5'] = panel.groupby('code')['close'].shift(-5) / (panel.groupby('code')['open'].shift(-1) + 1e-5) - 1
    panel['fwd_ret_t20'] = panel.groupby('code')['close'].shift(-20) / (panel.groupby('code')['open'].shift(-1) + 1e-5) - 1
    
    df = build_ml_features(panel)
    df['date'] = pd.to_datetime(df['date'])
    
    # 选择样本外的时间段进行仿真 (例如最近1-2年，避免消耗过多内存)
    oos_df = df[df['date'] >= '2025-01-01'].copy()
    log.info(f"仿真区间: {oos_df['date'].min().date()} 至 {oos_df['date'].max().date()} ({len(oos_df)} 样本)")
    
    # 2. 加载模型结构和权重
    horizons = [1, 5, 10, 20]
    features_dict = {}
    models_dict = {}
    
    for h in horizons:
        try:
            with open(f'.quantbot_data/prod_pt_meta_t{h}.json', 'r') as f:
                features = json.load(f)['features']
                features_dict[h] = features
                
            model = PyTorchDLModel(len(features))
            model.load_model(f'.quantbot_data/prod_pt_model_t{h}.pth')
            models_dict[h] = model
            log.info(f"✅ 成功加载 T+{h} 模型及其 {len(features)} 个特征。")
        except Exception as e:
            log.error(f"加载 T+{h} 模型失败: {e}")
            return
            
    # 3. 生成预测与截面排名
    log.info("正在生成各个周期的预测分数...")
    for h in horizons:
        oos_df[f'pred_t{h}'] = models_dict[h].predict(oos_df, features_dict[h])
        # 计算截面排名 (0.0 到 1.0)
        oos_df[f'rank_t{h}'] = oos_df.groupby('date')[f'pred_t{h}'].rank(pct=True)

    # 4. 仿真执行逻辑 (Veto System)
    # [主脑过滤]：由极其稳健的 T+10 和 T+20 强强联手，合成绝对基本盘！
    oos_df['core_rank'] = (oos_df['rank_t10'] + oos_df['rank_t20']) / 2.0
    oos_df['final_core_rank'] = oos_df.groupby('date')['core_rank'].rank(pct=True)
    baseline_pool = oos_df[oos_df['final_core_rank'] >= 0.90].copy()
    
    # [择时滤网 A：防追高]：如果 T+1 极度看多 (排名前 5%)，说明今天刚拉大阳线，属于“追高”，大概率次日回落。
    veto_t1_pool = baseline_pool[baseline_pool['rank_t1'] >= 0.95]
    
    # [择时滤网 B：防阴跌]：如果 T+5 严重看空 (排名后 10%)，说明接下来一周动量极差。
    veto_t5_pool = baseline_pool[baseline_pool['rank_t5'] <= 0.10]
    
    # [最终买入池]：经过双重否决后的干净筹码
    final_filtered_pool = baseline_pool[
        (baseline_pool['rank_t1'] < 0.95) & 
        (baseline_pool['rank_t5'] > 0.10)
    ]
    
    # 5. 打印回测结果
    log.info("\n" + "=" * 40)
    log.info("仿真回测结果 (收益率 = Q5多头池持仓收益):")
    log.info("=" * 40)
    
    evaluate_pool(baseline_pool, "1. 原始底仓池 (T+10 & T+20 强强联手 Top10%)")
    
    log.info("-" * 40)
    evaluate_pool(veto_t1_pool, "🚫 被 T+1 否决的追高票 (应有极差的 T+1 收益)")
    evaluate_pool(veto_t5_pool, "🚫 被 T+5 否决的阴跌票 (应有极差的 T+5 收益)")
    
    log.info("-" * 40)
    evaluate_pool(final_filtered_pool, "🏆 终极执行池 (过掉双重滤网后)")
    
    # 计算胜率提升
    base_t20_ret = baseline_pool['fwd_ret_t20'].mean() * 10000
    final_t20_ret = final_filtered_pool['fwd_ret_t20'].mean() * 10000
    improvement = final_t20_ret - base_t20_ret
    
    log.info("=" * 40)
    log.info(f"🎯 择时滤网价值: 为主脑核心持仓贡献了 +{improvement:.2f} bps 的超额安全垫！")
    log.info("=" * 40)

if __name__ == '__main__':
    run_veto_simulation()
