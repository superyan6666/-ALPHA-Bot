import os
import json
import logging
import pandas as pd
from datetime import datetime
from factor_library import calculate_factors, registry
from factor_screener import InitialScreener, FineScreener

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class PipelineManager:
    def __init__(self, data_path: str, output_path: str = 'promoted_factors.json'):
        self.data_path = data_path
        self.output_path = output_path
        
    def load_data(self):
        log.info(f"Loading base data from {self.data_path}...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file {self.data_path} not found. Please ensure base data exists.")
        
        # Determine if we are loading csi300_price.csv
        if "csi300_price.csv" in self.data_path:
            data_dir = os.path.dirname(self.data_path)
            price_df = pd.read_csv(self.data_path)
            price_df.rename(columns={'Date': 'date'}, inplace=True)
            price_df['date'] = pd.to_datetime(price_df['date'])
            
            df = price_df.set_index('date')
            for filename in ["csi300_pe.csv", "cn_10y_yield.csv", "vix_data.csv"]:
                filepath = os.path.join(data_dir, filename)
                if os.path.exists(filepath):
                    extra_df = pd.read_csv(filepath)
                    extra_df.rename(columns={'Date': 'date'}, inplace=True)
                    extra_df['date'] = pd.to_datetime(extra_df['date'])
                    df = df.join(extra_df.set_index('date'), how='left')
            
            df = df.sort_index().ffill().dropna(subset=['close']).reset_index()
            df['code'] = '000300.SH'
            df['open'] = df['close']
            df['volume'] = 1e6
        else:
            df = pd.read_csv(self.data_path)
            if 'Date' in df.columns:
                df.rename(columns={'Date': 'date'}, inplace=True)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
        return df

    def run_pipeline(self):
        log.info("="*50)
        log.info("ALPHA FACTORY PIPELINE STARTED")
        log.info("="*50)
        
        try:
            df = self.load_data()
        except Exception as e:
            log.error(f"Failed to load data: {e}")
            return
            
        # Stage 1: Generation & Incubation
        log.info("\n--- STAGE 1: Factor Generation ---")
        df = calculate_factors(df)
        factor_cols = [c for c in df.columns if c.startswith('F_') and c != 'F_float_cap']
        
        # Stage 1.5: Factor Decay Monitor
        df = self.monitor_decay(df)
        
        # Stage 2: Initial Screening (Rank IC)
        log.info("\n--- STAGE 2: Initial Screening ---")
        screener1 = InitialScreener(target_ret_window=20, max_workers=4)
        screened_df, passed_initial = screener1.screen(df, factor_cols)
        
        if not passed_initial:
            log.error("Pipeline Aborted: No factors passed Stage 2.")
            return
            
        # Stage 3: Fine Screening (XGBoost & Collinearity)
        log.info("\n--- STAGE 3: Fine Screening ---")
        screener2 = FineScreener(corr_threshold=0.7) # Using 0.7 as per plan
        promoted_factors, ortho_weights = screener2.screen(screened_df, passed_initial)
        
        # Stage 3.5: Gene Atlas Building & Diversity Enforcement
        log.info("\n--- STAGE 3.5: Gene Atlas & Diversity ---")
        try:
            from gene_atlas import GeneAtlas
            atlas = GeneAtlas()
            run_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            # Build using all active factors from Stage 1/2? No, use factor_cols (all generated ones)
            # wait, the plan says: `GeneAtlas.build(screened_df, passed_fine)` ?
            # Actually building with all available factor_cols gives a complete universe atlas
            atlas.build(screened_df, factor_cols, target='excess_ret_t20', pipeline_run_id=run_id)
            promoted_factors = atlas.enforce_diversity(promoted_factors, max_per_gene=2)
        except Exception as e:
            log.error(f"[GeneAtlas] 基因图谱构建或多样性门控失败: {e}，将跳过多样性控制")
        
        # Stage 4 & 5: Registration & Promotion
        log.info("\n--- STAGE 5: Registration & Promotion ---")
        self._register_promoted(promoted_factors, ortho_weights)
        
        log.info("="*50)
        log.info(f"PIPELINE COMPLETE. {len(promoted_factors)} factors successfully promoted to production.")
        log.info("="*50)

    def _register_promoted(self, promoted_list, ortho_weights=None):
        registry_data = {}
        if os.path.exists(self.output_path):
            with open(self.output_path, 'r', encoding='utf-8') as f:
                registry_data = json.load(f)
                
        run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        from gene_atlas import GeneAtlas
        atlas = GeneAtlas()
        
        for f in promoted_list:
            gene_info = atlas.get_gene(f)
            gene_id = gene_info['gene_id'] if gene_info else None
            gene_label = gene_info['gene_label'] if gene_info else ""
            
            if f not in registry_data:
                registry_data[f] = {
                    "promoted_at": run_date,
                    "status": "ACTIVE",
                    "history": [],
                    "gene_id": gene_id,
                    "gene_label": gene_label
                }
            else:
                # Update gene info
                registry_data[f]["gene_id"] = gene_id
                registry_data[f]["gene_label"] = gene_label
            
            # Record this successful run
            registry_data[f]["history"].append({
                "date": run_date,
                "event": "PROMOTED_VIA_PIPELINE"
            })
            
            if ortho_weights and f in ortho_weights and ortho_weights[f]:
                registry_data[f]["ortho_weights"] = ortho_weights[f]
            
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(registry_data, f, indent=4, ensure_ascii=False)
            
        log.info(f"Successfully recorded promoted factors to {self.output_path}")

    def monitor_decay(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("\n--- STAGE 1.5: Factor Decay Monitor ---")
        if not os.path.exists(self.output_path):
            log.info("No promoted_factors.json found. Skipping decay monitor.")
            return df
            
        with open(self.output_path, 'r', encoding='utf-8') as f:
            registry_data = json.load(f)

        tracked_factors = [f"F_{k}" for k, v in registry_data.items() if v.get('status') in ['ACTIVE', 'WATCH', 'DEGRADED']]
        eval_cols = [c for c in tracked_factors if c in df.columns]
        
        if not eval_cols:
            log.info("No valid tracked factors found in current data.")
            return df

        recent_df = df.copy()
        if 'date' in recent_df.columns:
            recent_df = recent_df.sort_values('date')
            dates = recent_df['date'].unique()
            if len(dates) > 60:
                recent_df = recent_df[recent_df['date'].isin(dates[-60:])]

        log.info(f"Evaluating decay on {len(recent_df)} recent records for {len(eval_cols)} factors.")
        
        recent_df['next_open'] = recent_df.groupby('code')['open'].shift(-1)
        recent_df['close_tn'] = recent_df.groupby('code')['close'].shift(-20)
        recent_df['fwd_ret'] = recent_df['close_tn'] / (recent_df['next_open'] + 1e-8) - 1.0
        
        is_single_asset = (recent_df['code'].nunique() == 1)
        if is_single_asset:
            recent_df['target'] = recent_df['fwd_ret']
        else:
            recent_df['market_fwd_ret'] = recent_df.groupby('date')['fwd_ret'].transform('mean')
            recent_df['target'] = recent_df['fwd_ret'] - recent_df['market_fwd_ret']
            
        recent_df = recent_df.dropna(subset=['target'] + eval_cols)
        
        import concurrent.futures
        screener = InitialScreener(target_ret_window=20, max_workers=4)
        ic_results = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(screener._calc_single_ic, recent_df[['date', 'code', 'target', f]], f, is_single_asset): f for f in eval_cols}
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    ic_results[res['factor']] = res
                    
        run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # === 动态相对淘汰机制 ===
        if ic_results:
            ic_df = pd.DataFrame(list(ic_results.values()))
            ic_df['ic_rank'] = ic_df['mean_ic'].rank(pct=True)
            ic_results_enriched = ic_df.set_index('factor').to_dict('index')
        else:
            ic_results_enriched = {}
        
        for k, v in registry_data.items():
            col_name = f"F_{k}"
            if col_name not in ic_results_enriched:
                continue
                
            res = ic_results_enriched[col_name]
            mean_ic = res['mean_ic']
            t_stat = res['t_stat']
            ic_rank = res.get('ic_rank', 0.5)
            
            old_status = v.get('status', 'ACTIVE')
            new_status = old_status
            
            # 只有处在倒数 10% 或者 IC 绝对值为负，才会被标记为劣质
            is_poor = (ic_rank <= 0.1) or (mean_ic < 0)
            is_good = (ic_rank >= 0.5) and (mean_ic > 0.01)
            
            if old_status == 'ACTIVE' and is_poor:
                new_status = 'WATCH'
            elif old_status == 'WATCH':
                if is_poor:
                    watch_since = v.get('watch_since')
                    if watch_since:
                        days_in_watch = (datetime.now() - datetime.strptime(watch_since, "%Y-%m-%d %H:%M:%S")).days
                        if days_in_watch > 90:
                            new_status = 'DEGRADED'
                    else:
                        v['watch_since'] = run_date
                elif is_good:
                    new_status = 'ACTIVE'
                    v.pop('watch_since', None)
            elif old_status == 'DEGRADED' and is_good:
                new_status = 'ACTIVE'
                v.pop('watch_since', None)
            elif old_status == 'RECOVERED' and is_poor:
                new_status = 'WATCH'
                
            if new_status != old_status:
                log.info(f"Factor {k} transitioned: {old_status} -> {new_status} (IC: {mean_ic:.4f}, T: {t_stat:.2f})")
                v['status'] = new_status
                v.setdefault("history", []).append({
                    "date": run_date,
                    "event": f"TRANSITION_{old_status}_TO_{new_status}",
                    "mean_ic": round(mean_ic, 4),
                    "t_stat": round(t_stat, 2)
                })
                
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(registry_data, f, indent=4, ensure_ascii=False)
            
        return df

if __name__ == "__main__":
    # Point this to a valid pre-processed parquet or csv with stock data
    # For now, it assumes research/data/merged_data.csv exists or similar.
    # Note: In real usage, the user should point this to their canonical dataset.
    manager = PipelineManager(data_path="research/data/csi300_price.csv")
    manager.run_pipeline()
