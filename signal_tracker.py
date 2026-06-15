import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

log = logging.getLogger(__name__)

HISTORY_PATH = '.quantbot_data/signal_history.jsonl'
DATA_LAKE_PATH = '.quantbot_data/ashare_daily.parquet'

class SignalTracker:
    """信号反馈闭环追踪器 (Signal Outcome Tracker)
    
    职责：
    1. log_signal() — 在信号推送时追加写入 JSONL 归档
    2. backfill_outcomes() — 从 ashare_daily.parquet 回填 T+1/5/10/20 实际收益
    3. update_outcome() — 当 AdvisoryTracker 判定止盈/止损/到期时更新结果
    4. generate_report() — 生成月度信号体检报告
    5. get_training_feedback() — 导出已完成回填的信号作为训练反馈 DataFrame
    """
    
    def __init__(self, history_path=HISTORY_PATH):
        self.history_path = history_path
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
    
    def log_signal(self, signals_dict: dict, ranks_dict: dict = None, market_context: str = ""):
        """在信号推送后立即调用，将信号追加到 JSONL。
        
        Args:
            signals_dict: {'Resonance': [Signal, ...]} from get_signals()
            ranks_dict: {code: {'rank_t1': float, 'rank_t5': float, ...}} per-stock ranks
            market_context: market analysis summary string
        """
        today_str = datetime.now().strftime('%Y-%m-%d')
        
        records = []
        for category, signal_list in signals_dict.items():
            for sig in signal_list:
                ranks = ranks_dict.get(sig.code, {}) if ranks_dict else {}
                record = {
                    'signal_date': today_str,
                    'category': category,
                    'code': sig.code,
                    'name': sig.name,
                    'entry_price': float(sig.price),
                    'score': float(sig.score),
                    'rank_t1': float(ranks.get('rank_t1', 0)),
                    'rank_t5': float(ranks.get('rank_t5', 0)),
                    'rank_t10': float(ranks.get('rank_t10', 0)),
                    'rank_t20': float(ranks.get('rank_t20', 0)),
                    'core_rank': float(ranks.get('core_rank', 0)),
                    'stop_loss': float(sig.stop_loss),
                    'target1': float(sig.target1),
                    'market_context': market_context[:200] if market_context else '',
                    # Outcome fields — initially null, backfilled later
                    'actual_ret_t1': None,
                    'actual_ret_t5': None,
                    'actual_ret_t10': None,
                    'actual_ret_t20': None,
                    'csi300_ret_t1': None,
                    'csi300_ret_t5': None,
                    'csi300_ret_t10': None,
                    'csi300_ret_t20': None,
                    'excess_ret_t20': None,
                    'hit_target': None,
                    'hit_stop': None,
                    'max_gain_pct': None,
                    'max_drawdown_pct': None,
                    'outcome_status': 'PENDING',  # PENDING -> PARTIAL -> COMPLETE
                    'last_updated': today_str
                }
                records.append(record)
        
        if not records:
            log.info("[SignalTracker] 本次无信号，跳过归档。")
            return
            
        with open(self.history_path, 'a', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        
        log.info(f"[SignalTracker] ✅ 已归档 {len(records)} 条信号到 {self.history_path}")
    
    def _load_history(self) -> list[dict]:
        """Load all records from JSONL."""
        if not os.path.exists(self.history_path):
            return []
        records = []
        with open(self.history_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        log.warning(f"[SignalTracker] 跳过损坏行: {line[:50]}...")
        return records
    
    def _save_history(self, records: list[dict]):
        """Overwrite entire JSONL with updated records."""
        with open(self.history_path, 'w', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    def backfill_outcomes(self, data_lake_path: str = DATA_LAKE_PATH):
        """从数据湖回填信号的实际收益。
        
        对每条 PENDING/PARTIAL 状态的信号：
        - 查找 signal_date 之后的价格序列
        - 计算 T+1/5/10/20 的实际收益率 (close_t+n / entry_price - 1)
        - 计算 max_gain 和 max_drawdown
        - 如果 T+20 已有数据，标记为 COMPLETE
        """
        records = self._load_history()
        if not records:
            log.info("[SignalTracker] 无历史信号，跳过回填。")
            return
        
        pending = [r for r in records if r.get('outcome_status') != 'COMPLETE']
        if not pending:
            log.info("[SignalTracker] 所有信号已回填完毕。")
            return
        
        # Load data lake
        if not os.path.exists(data_lake_path):
            log.error(f"[SignalTracker] 数据湖 {data_lake_path} 不存在，无法回填。")
            return
        
        log.info(f"[SignalTracker] 加载数据湖进行回填 ({len(pending)} 条待处理)...")
        lake = pd.read_parquet(data_lake_path)
        lake['date'] = pd.to_datetime(lake['date'])
        # Normalize code format
        if lake['code'].str.contains('.').any():
            lake['code'] = lake['code'].str.replace('sh.', '', regex=False).str.replace('sz.', '', regex=False).str.replace('bj.', '', regex=False)
        lake = lake.sort_values(['code', 'date'])
        
        # Build virtual market index to compute excess returns
        log.info("[SignalTracker] 构建等权虚拟大盘指数...")
        # Since 'close' pct_change across different groups requires sorting by code first,
        # it's easier to just compute daily return per stock then take cross-sectional mean:
        lake['daily_ret'] = lake.groupby('code')['close'].pct_change()
        daily_mkt_ret = lake.groupby('date')['daily_ret'].mean(skipna=True).fillna(0)
        market_idx = daily_mkt_ret.reset_index(name='daily_ret').sort_values('date')
        market_idx['mkt_index'] = (1 + market_idx['daily_ret']).cumprod()
        
        today = datetime.now()
        updated_count = 0
        
        for rec in records:
            if rec.get('outcome_status') == 'COMPLETE':
                continue
            
            code = rec['code']
            signal_date = pd.to_datetime(rec['signal_date'])
            entry_price = rec['entry_price']
            
            # 如果 entry_price 缺失（如从 git 恢复的记录），从数据湖补全
            if not entry_price or entry_price <= 0:
                entry_day = lake[(lake['code'] == code) & (lake['date'] == signal_date)]
                if not entry_day.empty:
                    entry_price = float(entry_day.iloc[0]['close'])
                    rec['entry_price'] = entry_price
                else:
                    log.warning(f"[SignalTracker] {code} 在 {rec['signal_date']} 无数据，跳过回填。")
                    continue
            
            # Get stock's price series after signal date
            stock_data = lake[(lake['code'] == code) & (lake['date'] > signal_date)].sort_values('date')
            
            if stock_data.empty:
                continue
            
            closes = stock_data['close'].values
            dates = stock_data['date'].values
            
            # Calculate returns for each horizon
            horizons = {'t1': 1, 't5': 5, 't10': 10, 't20': 20}
            max_available = len(closes)
            
            # Entry market index value
            mkt_entry_row = market_idx[market_idx['date'] == signal_date]
            mkt_entry_val = float(mkt_entry_row['mkt_index'].iloc[0]) if not mkt_entry_row.empty else 1.0
            
            for suffix, offset in horizons.items():
                field = f'actual_ret_{suffix}'
                csi_field = f'csi300_ret_{suffix}'
                exc_field = f'excess_ret_{suffix}'
                
                if rec.get(field) is not None and rec.get(exc_field) is not None:
                    continue  # Already filled
                if max_available >= offset:
                    ret = (float(closes[offset - 1]) / entry_price - 1)
                    rec[field] = round(ret, 6)
                    
                    # Compute market return over the same period
                    exit_date = dates[offset - 1]
                    mkt_exit_row = market_idx[market_idx['date'] == exit_date]
                    if not mkt_exit_row.empty and mkt_entry_val > 0:
                        mkt_exit_val = float(mkt_exit_row['mkt_index'].iloc[0])
                        m_ret = (mkt_exit_val / mkt_entry_val) - 1
                        rec[csi_field] = round(m_ret, 6)
                        rec[exc_field] = round(ret - m_ret, 6)
            
            # Max gain and max drawdown over available period (up to 20 days)
            lookback = min(max_available, 20)
            if lookback > 0:
                price_series = closes[:lookback].astype(float)
                gains = (price_series / entry_price - 1)
                rec['max_gain_pct'] = round(float(np.max(gains)), 6)
                rec['max_drawdown_pct'] = round(float(np.min(gains)), 6)
                
                # Hit target/stop check
                target = rec.get('target1', 0)
                stop = rec.get('stop_loss', 0)
                if target and target > 0:
                    rec['hit_target'] = bool(np.any(price_series >= target))
                if stop and stop > 0:
                    rec['hit_stop'] = bool(np.any(price_series <= stop))
            
            # Determine status
            if rec.get('actual_ret_t20') is not None:
                rec['outcome_status'] = 'COMPLETE'
            elif any(rec.get(f'actual_ret_{s}') is not None for s in ['t1', 't5', 't10']):
                rec['outcome_status'] = 'PARTIAL'
            
            rec['last_updated'] = today.strftime('%Y-%m-%d')
            updated_count += 1
        
        self._save_history(records)
        log.info(f"[SignalTracker] ✅ 回填完成，更新了 {updated_count} 条记录。")
    
    def update_outcome(self, code: str, entry_date: str, status: str, exit_price: float):
        """当 AdvisoryTracker 判定止盈/止损/到期时调用。
        
        Args:
            code: 股票代码
            entry_date: 建仓日期 YYYY-MM-DD
            status: 'HIT_TARGET' | 'HIT_STOP' | 'EXPIRED'
            exit_price: 退出时价格
        """
        records = self._load_history()
        for rec in records:
            if rec['code'] == code and rec['signal_date'] == entry_date:
                entry_price = rec.get('entry_price', 0)
                if entry_price > 0:
                    rec['exit_ret'] = round((exit_price / entry_price - 1), 6)
                rec['advisory_status'] = status
                rec['exit_price'] = float(exit_price)
                rec['last_updated'] = datetime.now().strftime('%Y-%m-%d')
                break
        
        self._save_history(records)
        log.info(f"[SignalTracker] 📌 已更新 {code} ({entry_date}) 的退出状态: {status}")
    
    def generate_report(self) -> str:
        """生成信号绩效报告。"""
        records = self._load_history()
        if not records:
            return "📊 暂无历史信号数据。"
        
        df = pd.DataFrame(records)
        total = len(df)
        complete = len(df[df['outcome_status'] == 'COMPLETE'])
        pending = len(df[df['outcome_status'] == 'PENDING'])
        
        lines = [
            "# 📊 信号绩效体检报告",
            f"\n**统计时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**信号总量**: {total} 条 (完成回填: {complete}, 待回填: {pending})",
            ""
        ]
        
        if complete > 0:
            completed = df[df['outcome_status'] == 'COMPLETE'].copy()
            
            for horizon in ['t1', 't5', 't10', 't20']:
                col = f'actual_ret_{horizon}'
                if col in completed.columns:
                    valid = completed[col].dropna()
                    if len(valid) > 0:
                        win_rate = (valid > 0).mean() * 100
                        avg_ret = valid.mean() * 100
                        lines.append(f"### {horizon.upper()} 周期")
                        lines.append(f"- 胜率: {win_rate:.1f}% (n={len(valid)})")
                        lines.append(f"- 平均收益: {avg_ret:.2f}%")
                        lines.append(f"- 最佳: {valid.max()*100:.2f}% | 最差: {valid.min()*100:.2f}%")
                        lines.append("")
            
            # Target/Stop stats
            if 'hit_target' in completed.columns:
                ht = completed['hit_target'].dropna()
                if len(ht) > 0:
                    lines.append(f"### 止盈止损统计")
                    lines.append(f"- 触达目标价: {ht.sum()}/{len(ht)} ({ht.mean()*100:.1f}%)")
                if 'hit_stop' in completed.columns:
                    hs = completed['hit_stop'].dropna()
                    if len(hs) > 0:
                        lines.append(f"- 触达止损线: {hs.sum()}/{len(hs)} ({hs.mean()*100:.1f}%)")
                lines.append("")
            
            # Score bucketing — does higher score actually predict better returns?
            if 'score' in completed.columns and 'actual_ret_t20' in completed.columns:
                completed['score_bucket'] = pd.cut(completed['score'], bins=[0, 80, 90, 95, 100], labels=['<80', '80-90', '90-95', '95+'])
                bucket_stats = completed.groupby('score_bucket', observed=True)['actual_ret_t20'].agg(['mean', 'count'])
                lines.append("### 得分分档 vs T+20 实际收益")
                lines.append("| 得分区间 | 平均收益 | 样本数 |")
                lines.append("|---------|---------|-------|")
                for bucket, row in bucket_stats.iterrows():
                    lines.append(f"| {bucket} | {row['mean']*100:.2f}% | {int(row['count'])} |")
                lines.append("")
        
        return '\n'.join(lines)
    
    def get_training_feedback(self, min_age_days: int = 20) -> pd.DataFrame:
        """导出已完成回填的信号，供未来训练反馈使用。
        
        Returns:
            DataFrame with columns: [code, signal_date, score, actual_ret_t20, excess_ret_t20, ...]
        """
        records = self._load_history()
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        complete = df[df['outcome_status'] == 'COMPLETE'].copy()
        
        if complete.empty:
            log.info("[SignalTracker] 无已完成信号可供训练反馈。")
            return pd.DataFrame()
        
        log.info(f"[SignalTracker] 导出 {len(complete)} 条已完成信号作为训练反馈。")
        return complete
