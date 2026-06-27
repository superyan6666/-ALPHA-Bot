
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def bootstrap_monte_carlo(daily_returns, n_simulations=10000, block_size=20):
    """
    Block Bootstrap on daily returns.
    """
    logging.info(f"Running Bootstrap Monte Carlo with {n_simulations} simulations...")
    n_days = len(daily_returns)
    simulated_returns = []
    
    for _ in tqdm(range(n_simulations)):
        # Generate random blocks
        sim_ret = []
        while len(sim_ret) < n_days:
            start_idx = np.random.randint(0, n_days - block_size)
            sim_ret.extend(daily_returns[start_idx:start_idx+block_size])
            
        sim_ret = sim_ret[:n_days]
        simulated_returns.append(sim_ret)
        
    simulated_returns = np.array(simulated_returns)
    
    # Calculate metrics for each simulation
    cagrs = []
    max_dds = []
    sharpes = []
    
    for i in range(n_simulations):
        ret = simulated_returns[i]
        cum_ret = np.cumprod(1 + ret)
        
        ann_ret = np.mean(ret) * 252
        ann_vol = np.std(ret) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        roll_max = np.maximum.accumulate(cum_ret)
        drawdown = cum_ret / roll_max - 1.0
        max_dd = np.min(drawdown)
        
        cagrs.append(ann_ret)
        max_dds.append(max_dd)
        sharpes.append(sharpe)
        
    return np.array(cagrs), np.array(max_dds), np.array(sharpes)

def zero_intelligence_monte_carlo(df, start_date, end_date, n_simulations=1000):
    """
    Random monkey picking 20 stocks everyday.
    """
    logging.info(f"Running Zero-Intelligence Monte Carlo with {n_simulations} simulations...")
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df_eval = df[mask].copy()
    
    # We only care about date and fwd_ret_t1
    dates = df_eval['date'].unique()
    dates = np.sort(dates)
    
    # Create a mapping from date to available returns
    date_to_returns = df_eval.groupby('date')['fwd_ret_t1'].apply(list).to_dict()
    
    cagrs = []
    for _ in tqdm(range(n_simulations)):
        sim_daily_ret = []
        for d in dates:
            available_rets = date_to_returns.get(d, [])
            if len(available_rets) == 0:
                sim_daily_ret.append(0.0)
                continue
            
            # Randomly pick up to 20 stocks
            n_picks = min(20, len(available_rets))
            picks = np.random.choice(available_rets, size=n_picks, replace=False)
            sim_daily_ret.append(np.mean(picks))
            
        sim_daily_ret = np.array(sim_daily_ret)
        ann_ret = np.mean(sim_daily_ret) * 252
        cagrs.append(ann_ret)
        
    return np.array(cagrs)

def main():
    # 1. Load actual portfolio returns
    try:
        portfolio_df = pd.read_csv('.quantbot_data/portfolio_daily_wfo.csv')
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df = portfolio_df.sort_values('date')
        actual_daily_ret = portfolio_df['fwd_ret_t1'].values
        actual_cagr = np.mean(actual_daily_ret) * 252
        actual_max_dd = np.min(np.cumprod(1+actual_daily_ret) / np.maximum.accumulate(np.cumprod(1+actual_daily_ret)) - 1)
        actual_sharpe = actual_cagr / (np.std(actual_daily_ret) * np.sqrt(252))
        
        logging.info(f"Actual Strategy - CAGR: {actual_cagr:.2%}, MaxDD: {actual_max_dd:.2%}, Sharpe: {actual_sharpe:.2f}")
    except Exception as e:
        logging.error(f"Could not load portfolio_daily_wfo.csv: {e}")
        return

    # 2. Run Bootstrap MC
    cagrs, max_dds, sharpes = bootstrap_monte_carlo(actual_daily_ret, n_simulations=10000, block_size=20)
    
    # 3. Run Zero-Intelligence MC
    logging.info("Loading ashare_daily.parquet for Zero-Intelligence MC...")
    df = pd.read_parquet('.quantbot_data/ashare_daily.parquet', columns=['date', 'code', 'close', 'open'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date'])
    df['fwd_ret_t1'] = df.groupby('code')['close'].shift(-1) / (df.groupby('code')['open'].shift(-1) + 1e-5) - 1
    df = df.dropna(subset=['fwd_ret_t1'])
    
    start_date = portfolio_df['date'].min()
    end_date = portfolio_df['date'].max()
    zi_cagrs = zero_intelligence_monte_carlo(df, start_date, end_date, n_simulations=1000)
    
    # 4. Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Bootstrap CAGR
    axes[0, 0].hist(cagrs, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(actual_cagr, color='red', linestyle='dashed', linewidth=2, label=f'Actual: {actual_cagr:.2%}')
    axes[0, 0].axvline(np.percentile(cagrs, 5), color='orange', linestyle='dashed', linewidth=2, label=f'5% CI: {np.percentile(cagrs, 5):.2%}')
    axes[0, 0].set_title('Bootstrap Distribution: CAGR')
    axes[0, 0].legend()
    
    # Bootstrap MaxDD
    axes[0, 1].hist(max_dds, bins=50, color='salmon', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(actual_max_dd, color='red', linestyle='dashed', linewidth=2, label=f'Actual: {actual_max_dd:.2%}')
    axes[0, 1].axvline(np.percentile(max_dds, 5), color='orange', linestyle='dashed', linewidth=2, label=f'5% CI (Worst Case): {np.percentile(max_dds, 5):.2%}')
    axes[0, 1].set_title('Bootstrap Distribution: Max Drawdown')
    axes[0, 1].legend()
    
    # Bootstrap Sharpe
    axes[1, 0].hist(sharpes, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(actual_sharpe, color='red', linestyle='dashed', linewidth=2, label=f'Actual: {actual_sharpe:.2f}')
    axes[1, 0].axvline(np.percentile(sharpes, 5), color='orange', linestyle='dashed', linewidth=2, label=f'5% CI: {np.percentile(sharpes, 5):.2f}')
    axes[1, 0].set_title('Bootstrap Distribution: Sharpe Ratio')
    axes[1, 0].legend()
    
    # Zero-Intelligence CAGR
    zi_p_value = np.sum(zi_cagrs >= actual_cagr) / len(zi_cagrs)
    axes[1, 1].hist(zi_cagrs, bins=50, color='plum', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(actual_cagr, color='red', linestyle='dashed', linewidth=2, label=f'Strategy: {actual_cagr:.2%}')
    axes[1, 1].axvline(np.mean(zi_cagrs), color='gray', linestyle='dashed', linewidth=2, label=f'ZI Mean: {np.mean(zi_cagrs):.2%}')
    axes[1, 1].set_title(f'Zero-Intelligence (Monkey) CAGR\nP-Value: {zi_p_value:.4f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('monte_carlo_results.png')
    logging.info("Saved plots to monte_carlo_results.png")
    
    # Write report
    report = f'''
========================================
MONTE CARLO BACKTEST RESULTS
========================================

1. Bootstrap Analysis (10,000 paths, block size 20)
----------------------------------------
CAGR: Actual {actual_cagr:.2%} -> 95% Confidence Lower Bound: {np.percentile(cagrs, 5):.2%}
Max Drawdown: Actual {actual_max_dd:.2%} -> 95% Confidence Worst Case: {np.percentile(max_dds, 5):.2%}
Sharpe: Actual {actual_sharpe:.2f} -> 95% Confidence Lower Bound: {np.percentile(sharpes, 5):.2f}

2. Zero-Intelligence "Monkey" Test (1,000 simulations)
----------------------------------------
Monkey Mean CAGR: {np.mean(zi_cagrs):.2%}
Strategy CAGR: {actual_cagr:.2%}
P-Value (Probability of achieving this by pure luck): {zi_p_value:.4f}
========================================
'''
    logging.info(report)
    with open('monte_carlo_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

if __name__ == "__main__":
    main()
