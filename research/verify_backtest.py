
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import DataProxy, C
from feature_engine import build_ml_features
from ml_engine import PyTorchDLModel
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)

def main():
    log.info("🚀 启动策略快速回测验证 (CSI 300 对标)")
    dp = DataProxy()
    
    # 1. 加载基准 (沪深300)
    log.info("加载沪深300指数数据...")
    csi300 = dp.get_index('sh000300')
    if csi300 is None or csi300.empty:
        log.error("无法获取沪深300数据，回测退出。")
        return
        
    csi300 = csi300.rename(columns={C.H_DATE: 'date', C.H_CLOSE: 'close'})
    csi300['date'] = pd.to_datetime(csi300['date'])
    csi300 = csi300.sort_values('date').set_index('date')
    csi300['pct_chg'] = csi300['close'].pct_change().fillna(0)
    
    # 2. 加载测试标的池与特征
    parquet_path = ".quantbot_data/ashare_daily.parquet"
    log.info(f"加载测试股票池数据 {parquet_path}...")
    panel = pd.read_parquet(parquet_path)
    panel = panel.rename(columns={'date': 'date', 'code': 'code', 'open': 'open', 'close': 'close', 'high': 'high', 'low': 'low', 'volume': 'vol'})
    panel['date'] = pd.to_datetime(panel['date'])
    
    # 只回测 2023 年以来的数据
    start_dt = pd.to_datetime("2023-01-01")
    panel = panel[panel['date'] >= start_dt].copy()
    panel = panel.sort_values(['date', 'code'])
    
    # 预计算特征
    log.info("计算截面 ML 特征...")
    panel = build_ml_features(panel)
    
    # 加载模型
    h = 20
    model_path = f'.quantbot_data/prod_pt_model_t{h}.pth'
    meta_path = f'.quantbot_data/prod_pt_meta_t{h}.json'
    
    with open(meta_path, 'r') as f:
        features = json.load(f)['features']
        
    model = PyTorchDLModel(input_dim=len(features))
    model.load_model(model_path)
    
    # 推理打分
    log.info(f"执行 T+{h} 模型全量打分...")
    panel[f'xgb_score_t{h}'] = model.predict(panel, features)
    
    # 3. 构建每日投资组合 (Top 30 等权)
    log.info("构建每日等权投资组合...")
    # 过滤无效分数
    valid_panel = panel.dropna(subset=[f'xgb_score_t{h}', 'pct_chg'])
    
    # 计算个股明日实际收益 (用开盘价买入，收盘价卖出的日内模拟，或者简单的 close to close)
    # 既然是 T 预测，T+1 执行，我们的实际收益应该是 T+1 日的 pct_chg (收盘 vs 前收盘)
    valid_panel['next_ret'] = valid_panel.groupby('code')['pct_chg'].shift(-1) / 100.0
    valid_panel = valid_panel.dropna(subset=['next_ret'])
    
    # 每天选取得分最高的前 30 只股票
    def select_top_30(group):
        return group.nlargest(30, f'xgb_score_t{h}')['next_ret'].mean()
        
    strategy_daily_ret = valid_panel.groupby('date').apply(select_top_30)
    
    # 对齐日期
    common_dates = strategy_daily_ret.index.intersection(csi300.index)
    strategy_daily_ret = strategy_daily_ret.loc[common_dates]
    csi300_daily_ret = csi300.loc[common_dates, 'pct_chg']
    
    # 计算累计收益
    strategy_cum = (1 + strategy_daily_ret).cumprod()
    csi300_cum = (1 + csi300_daily_ret).cumprod()
    
    log.info(f"=== 回测结果 (2023-01-01 至今) ===")
    log.info(f"策略累计净值: {strategy_cum.iloc[-1]:.4f}")
    log.info(f"沪深300净值: {csi300_cum.iloc[-1]:.4f}")
    
    # 生成对标图表
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_cum.index, strategy_cum.values, label='Level 3 Quant Strategy', color='red')
    plt.plot(csi300_cum.index, csi300_cum.values, label='CSI 300 Benchmark', color='blue', alpha=0.7)
    plt.title('Strategy Performance vs CSI 300 (2023-2024)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    chart_path = 'artifacts/backtest_result.png'
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig(chart_path)
    log.info(f"图表已保存至 {chart_path}")

if __name__ == "__main__":
    main()
