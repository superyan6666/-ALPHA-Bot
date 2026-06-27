import pandas as pd
import numpy as np
import logging
from scipy import stats
import xgboost as xgb
import concurrent.futures

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class InitialScreener:
    def __init__(self, target_ret_window=20, max_workers=4):
        self.target_ret_window = target_ret_window
        self.max_workers = max_workers
        
    def _calc_single_ic(self, df, factor_name, is_single_asset=False):
        if is_single_asset:
            temp_df = df.copy()
            temp_df['Month'] = pd.to_datetime(temp_df['date']).dt.to_period('M')
            ic = temp_df.groupby('Month').apply(lambda x: x[factor_name].corr(x['target'], method='spearman') if len(x) > 5 else np.nan)
            
            # 单标的时间序列标准化后计算换手
            z_score = (temp_df[factor_name] - temp_df[factor_name].mean()) / (temp_df[factor_name].std() + 1e-8)
            turnover = z_score.diff().abs().mean()
        else:
            ic = df.groupby('date').apply(lambda g: g[factor_name].corr(g['target'], method='spearman'))
            
            # 截面 Rank 换手率计算 (0~1)
            temp_df = df.copy()
            temp_df['rank'] = temp_df.groupby('date')[factor_name].rank(pct=True)
            temp_df = temp_df.sort_values(['code', 'date'])
            turnover = temp_df.groupby('code')['rank'].diff().abs().mean()
            
        ic = ic.dropna()
        if len(ic) < 10:
            return None
            
        mean_ic = ic.mean()
        std_ic = ic.std()
        ir = mean_ic / (std_ic + 1e-8)
        t_stat = mean_ic / (std_ic / np.sqrt(len(ic)) + 1e-8)
        
        # 换手率扣减：假设千分之 1.5 的单边摩擦，20天持有期，Net IC 折损 = turnover * 2(双边) * 0.0015 * 20 = turnover * 0.06
        turnover_penalty = (turnover * 0.06) if pd.notna(turnover) else 0
        # 惩罚只能削弱预测力，不能改变方向
        net_ic = np.sign(mean_ic) * max(0, abs(mean_ic) - turnover_penalty)
        
        return {
            'factor': factor_name,
            'mean_ic': net_ic, # 用 Net IC 替代原始 IC 参与后续排序
            'raw_ic': mean_ic,
            'turnover': turnover,
            'ir': ir,
            't_stat': t_stat
        }

    def screen(self, df, factor_cols):
        log.info(f"Calculating target return for {self.target_ret_window} days...")
        df['next_open'] = df.groupby('code')['open'].shift(-1)
        df['close_tn'] = df.groupby('code')['close'].shift(-self.target_ret_window)
        df['fwd_ret'] = df['close_tn'] / (df['next_open'] + 1e-8) - 1.0
        
        is_single_asset = (df['code'].nunique() == 1)
        if is_single_asset:
            df['target'] = df['fwd_ret']
        else:
            df['market_fwd_ret'] = df.groupby('date')['fwd_ret'].transform('mean')
            df['target'] = df['fwd_ret'] - df['market_fwd_ret']
        
        log.info("Applying Fundamental Stock Universe Screen (Top 80% Cap, Top 80% Turn)...")
        if 'F_float_cap' in df.columns and 'turn' in df.columns:
            df['cap_rank'] = df.groupby('date')['F_float_cap'].rank(pct=True)
            df['turn_rank'] = df.groupby('date')['turn'].rank(pct=True)
            screened_df = df[(df['cap_rank'] > 0.2) & (df['turn_rank'] > 0.2)].copy()
        else:
            log.warning("Fundamental columns (F_float_cap, turn) missing, skipping universe filter.")
            screened_df = df.copy()
            
        screened_df = screened_df.dropna(subset=['target'] + factor_cols)
        log.info(f"Rows after initial screen and dropna: {len(screened_df)} / {len(df)}")
        
        log.info(f"Calculating Rank IC for {len(factor_cols)} factors using max_workers={self.max_workers}...")
        ic_results = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._calc_single_ic, screened_df[['date', 'code', 'target', f]], f, is_single_asset): f for f in factor_cols}
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res and abs(res['mean_ic']) > 0.015 and abs(res['t_stat']) > 1.5:
                    ic_results.append(res)
                    
        passed_df = pd.DataFrame(ic_results)
        if passed_df.empty:
            log.warning("No factors passed the initial IC screen!")
            return screened_df, []
            
        passed_df = passed_df.sort_values(by='mean_ic', key=abs, ascending=False)
        log.info(f"Initial Screen Passed: {len(passed_df)} factors out of {len(factor_cols)}")
        print(passed_df.to_string(index=False))
        
        return screened_df, passed_df['factor'].tolist()

MIN_FEEDBACK_SAMPLES = 60
FEEDBACK_DECAY = 0.95  # per 20-trading-day bucket

class FineScreener:
    def __init__(self, corr_threshold=0.8):
        self.corr_threshold = corr_threshold

    def _load_feedback(self) -> pd.DataFrame:
        """Load real-world signal outcomes from SignalTracker."""
        try:
            from signal_tracker import SignalTracker
            fb = SignalTracker().get_training_feedback()
            if fb.empty:
                return pd.DataFrame()
            # Normalise column names for joining
            fb = fb.rename(columns={'signal_date': 'date'})
            fb['date'] = pd.to_datetime(fb['date'])
            return fb[['code', 'date', 'excess_ret_t20']].dropna()
        except Exception as e:
            log.warning(f"[FineScreener] 无法加载实盘反馈 (非阻塞): {e}")
            return pd.DataFrame()

    def _compute_decay_weights(self, dates_series: pd.Series) -> pd.Series:
        """Exponential decay by 20-trading-day buckets from today."""
        today = pd.Timestamp.now().normalize()
        days_ago = (today - dates_series.dt.normalize()).dt.days.clip(lower=0)
        buckets = (days_ago // 20).astype(float)
        return FEEDBACK_DECAY ** buckets

    def screen(self, df, candidate_factors):
        log.info(f"Starting Fine Screening on {len(candidate_factors)} candidates...")
        dates = sorted(df['date'].unique())
        split_idx = int(len(dates) * 0.8)
        train_dates = dates[:split_idx]

        train_df = df[df['date'].isin(train_dates)].copy()

        # ── [Phase 2-A] 实盘反馈注入 ──────────────────────────────────────────
        feedback = self._load_feedback()
        using_real_labels = False
        sample_weights = None

        if len(feedback) >= MIN_FEEDBACK_SAMPLES:
            log.info(f"[FineScreener] ✅ 实盘反馈样本充足 ({len(feedback)} 条 ≥ {MIN_FEEDBACK_SAMPLES})，启动标签替换 + 衰减加权。")
            train_df['date'] = pd.to_datetime(train_df['date'])
            feedback['date'] = pd.to_datetime(feedback['date'])
            merged = train_df.merge(
                feedback.rename(columns={'excess_ret_t20': 'real_excess'}),
                on=['code', 'date'], how='left'
            )
            # Replace label where real data exists
            merged['real_target'] = np.where(
                merged['real_excess'].notna(),
                merged['real_excess'],
                merged['target']
            )
            # Compute decay weights: real rows get full decay weight, backtest rows get weight=1
            decay_w = self._compute_decay_weights(merged['date'])
            real_mask = merged['real_excess'].notna()
            merged.loc[real_mask, 'sample_weight'] = decay_w[real_mask]
            merged.loc[~real_mask, 'sample_weight'] = 1.0

            train_df = merged
            y_col = 'real_target'
            sample_weights = train_df['sample_weight'].values
            using_real_labels = True
        else:
            n = len(feedback)
            log.info(f"[FineScreener] ⚠️  实盘反馈样本不足 ({n} 条 < {MIN_FEEDBACK_SAMPLES})，继续使用回测标签，待积累更多真实记录后自动升级。")
            y_col = 'target'
        # ─────────────────────────────────────────────────────────────────────

        X_train = train_df[candidate_factors]
        y_train = train_df[y_col]

        log.info("Training XGBoost Regressor to extract SHAP/Gain Importance (limiting cores to 4)...")
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=42
        )
        fit_kwargs = {'sample_weight': sample_weights} if using_real_labels else {}
        model.fit(X_train, y_train, **fit_kwargs)

        importance = model.feature_importances_
        fi_df = pd.DataFrame({'factor': candidate_factors, 'gain': importance})
        fi_df = fi_df.sort_values(by='gain', ascending=False)

        log.info(f"Handling collinearity (Pearson > {self.corr_threshold}) via Orthogonalization...")
        corr_matrix = train_df[fi_df['factor']].corr(method='pearson').abs()

        selected_factors = []
        ortho_weights = {}
        
        for factor in fi_df['factor']:
            if len(selected_factors) == 0:
                selected_factors.append(factor)
                continue
                
            is_collinear = [sel for sel in selected_factors if corr_matrix.loc[factor, sel] > self.corr_threshold]
            
            if not is_collinear:
                selected_factors.append(factor)
            else:
                log.info(f"Orthogonalizing {factor} against {is_collinear}...")
                valid_mask = train_df[[factor] + is_collinear].notna().all(axis=1)
                X = train_df.loc[valid_mask, is_collinear].values
                y = train_df.loc[valid_mask, factor].values
                
                if len(y) > 10:
                    X_b = np.c_[np.ones((len(X), 1)), X]
                    theta, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=None)
                    intercept = theta[0]
                    betas = theta[1:]
                    
                    ortho_weights[factor] = {
                        'intercept': float(intercept),
                        'betas': {is_collinear[i]: float(betas[i]) for i in range(len(is_collinear))}
                    }
                    
                    pred = intercept + sum(train_df[col] * beta for col, beta in zip(is_collinear, betas))
                    train_df[factor] = train_df[factor] - pred
                    df[factor] = df[factor] - pred # Update the original df as well
                    
                selected_factors.append(factor)

        log.info(f"Fine Screen Passed: {len(selected_factors)} factors remaining.")
        final_importance = fi_df[fi_df['factor'].isin(selected_factors)]
        print(final_importance.to_string(index=False))

        return selected_factors, ortho_weights
