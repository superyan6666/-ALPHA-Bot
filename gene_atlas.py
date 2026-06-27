import os
import json
import logging
import glob
from datetime import datetime
from collections import Counter
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
import shutil

log = logging.getLogger(__name__)

LABEL_MAP = {
    "price_momentum": "动量家族",
    "volume": "量价结构",
    "value": "价值家族",
    "macro": "宏观基因",
    "fundamental": "基本面",
    "quality": "质量基因",
}

class GeneAtlas:
    def __init__(self, data_dir=".quantbot_data"):
        self.data_dir = data_dir
        self.atlas_path = os.path.join(data_dir, "gene_atlas.json")
        self.history_dir = os.path.join(data_dir, "gene_atlas_history")
        self.cache_path = os.path.join(data_dir, "ic_series_cache.parquet")
        os.makedirs(self.history_dir, exist_ok=True)
        
        self.atlas_data = self.load()

    def _compute_ic_series(self, df: pd.DataFrame, factor_cols: list, target: str) -> pd.DataFrame:
        """
        计算截面 Rank IC 时序
        返回 DataFrame: index 为时间(年月)，columns 为各个因子
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['year_month'] = df['date'].dt.to_period('M')
        
        ic_records = []
        groups = df.groupby('year_month')
        for ym, g in groups:
            if len(g) < 30: continue
            rec = {'year_month': ym}
            valid_g = g.dropna(subset=[target])
            if len(valid_g) < 30: continue
            
            for f in factor_cols:
                if f in valid_g.columns:
                    ic = valid_g[f].corr(valid_g[target], method='spearman')
                    rec[f] = ic
            ic_records.append(rec)
            
        if not ic_records:
            return pd.DataFrame(columns=factor_cols)
            
        ic_df = pd.DataFrame(ic_records).set_index('year_month')
        
        # Save to cache as intermediate product
        ic_df_save = ic_df.copy()
        ic_df_save.index = ic_df_save.index.astype(str)
        ic_df_save.to_parquet(self.cache_path)
        
        n_months = len(ic_df)
        if n_months < 24:
            log.warning(f"[GeneAtlas] IC 序列仅 {n_months} 个月度样本（< 24），聚类结果可能不稳定。")
            
        return ic_df

    def _cluster(self, ic_df: pd.DataFrame):
        """
        双轨距离，Ward 聚类，Permutation 检验
        """
        ic_df = ic_df.fillna(0)
        factors = ic_df.columns.tolist()
        n_factors = len(factors)
        
        if n_factors < 2:
            return {f: 1 for f in factors}, 1, 0.0

        # Pearson + Spearman 均值距离
        corr_pearson = ic_df.corr(method='pearson').fillna(0)
        corr_spearman = ic_df.corr(method='spearman').fillna(0)
        
        dist_pearson = 1 - corr_pearson
        dist_spearman = 1 - corr_spearman
        
        diff_max = (dist_pearson - dist_spearman).abs().max().max()
        if diff_max > 0.25:
            log.warning("[GeneAtlas] 检测到显著非线性协同，距离矩阵已取 Pearson+Spearman 均值。")
            
        dist_matrix = ((dist_pearson + dist_spearman) / 2).to_numpy(copy=True)
        
        # 确保对角线为0，截断
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix = np.clip(dist_matrix, 0, 2)
        
        condensed_dist = squareform(dist_matrix, checks=False)
        Z = linkage(condensed_dist, method='ward')
        
        if n_factors == 2:
            if dist_matrix[0, 1] < 0.5:
                return {factors[0]: 1, factors[1]: 1}, 1, 0.0
            else:
                return {factors[0]: 1, factors[1]: 2}, 2, 0.0
        
        best_k = min(n_factors, int(np.ceil(np.sqrt(n_factors))))
        best_score = -1.0
        stability_score = 0.0
        
        search_min = 2
        search_max = min(n_factors - 1, int(np.floor(np.sqrt(n_factors))) + 5)
        
        if search_max >= search_min and n_factors >= 3:
            for k in range(search_min, search_max + 1):
                labels = fcluster(Z, k, criterion='maxclust')
                if len(set(labels)) < 2:
                    continue
                    
                s_real = silhouette_score(dist_matrix, labels, metric='precomputed')
                
                # Permutation test
                s_nulls = []
                for _ in range(50):
                    idx = np.random.permutation(n_factors)
                    null_dist = dist_matrix[idx, :][:, idx]
                    null_condensed = squareform(null_dist, checks=False)
                    null_Z = linkage(null_condensed, method='ward')
                    null_labels = fcluster(null_Z, k, criterion='maxclust')
                    if len(set(null_labels)) > 1:
                        s_nulls.append(silhouette_score(null_dist, null_labels, metric='precomputed'))
                    
                if s_nulls:
                    mu = np.mean(s_nulls)
                    sigma = np.std(s_nulls) + 1e-8
                    if s_real > mu + 2 * sigma:
                        if s_real > best_score:
                            best_score = s_real
                            best_k = k
                            stability_score = (s_real - mu) / sigma
        
        if best_score < 0:
            log.info(f"[GeneAtlas] 未找到显著更优簇数，回退至保守值 {best_k}")
        else:
            log.info(f"[GeneAtlas] 自动搜索最优簇数: {best_k}, Stability Score={stability_score:.2f}")

        final_labels = fcluster(Z, best_k, criterion='maxclust')
        assignments = {factors[i]: int(final_labels[i]) for i in range(n_factors)}
        return assignments, best_k, stability_score

    def _generate_labels(self, assignments, ic_df):
        try:
            from factor_library import registry
            registry_data = registry.get_all_factors()
        except ImportError:
            registry_data = {}
            
        cat_map = {}
        for k, v in registry_data.items():
            cat_map[f"F_{k}"] = v['category']
            cat_map[k] = v['category']
            
        ic_means = ic_df.mean().to_dict()
        
        clusters = {}
        for f, gid in assignments.items():
            if gid not in clusters:
                clusters[gid] = []
            clusters[gid].append(f)
            
        labels_map = {}
        for gid, f_list in clusters.items():
            cats = [cat_map.get(f, '') for f in f_list if cat_map.get(f)]
            if cats:
                top_cat, top_count = Counter(cats).most_common(1)[0]
                if top_count / len(cats) > 0.5:
                    label_name = LABEL_MAP.get(top_cat, top_cat)
                    labels_map[gid] = f"G{gid}·{label_name}"
                else:
                    labels_map[gid] = f"G{gid}·混合基因"
            else:
                if f_list:
                    best_f = max(f_list, key=lambda f: abs(ic_means.get(f, 0)))
                    best_name = best_f.replace('F_', '')[:8]
                    labels_map[gid] = f"G{gid}·{best_name}簇"
                else:
                    labels_map[gid] = f"G{gid}·空"
                    
        return labels_map, clusters

    def build(self, df: pd.DataFrame, factor_cols: list, target: str = 'excess_ret_t20', pipeline_run_id: str = ""):
        log.info(f"STAGE 3.5: Gene Atlas Building... factors: {len(factor_cols)}")
        if len(factor_cols) == 0:
            return
            
        ic_df = self._compute_ic_series(df, factor_cols, target)
        if ic_df.empty:
            log.warning("[GeneAtlas] IC series is empty, cannot build atlas.")
            return

        assignments, n_clusters, stability = self._cluster(ic_df)
        labels_map, clusters = self._generate_labels(assignments, ic_df)
        
        self.save(assignments, labels_map, clusters, {
            "n_monthly_samples": len(ic_df),
            "target": target,
            "n_factors": len(factor_cols),
            "n_clusters": n_clusters,
            "stability_score": round(stability, 2),
            "pipeline_run_id": pipeline_run_id
        })
        log.info(f"[GeneAtlas] 最优基因簇数: {n_clusters}，因子族谱已保存至 gene_atlas.json")

    def save(self, assignments, labels_map, clusters_dict, meta_info):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        factors_data = {}
        for f, gid in assignments.items():
            factors_data[f] = {
                "gene_id": gid,
                "gene_label": labels_map[gid]
            }
            
        clusters_data = {}
        for gid, c_list in clusters_dict.items():
            clusters_data[str(gid)] = {
                "label": labels_map[gid],
                "components": c_list
            }
            
        atlas = {
            "_meta": {
                "version": "1.0.0",
                "build_date": now_str,
                "pipeline_run_id": meta_info.get("pipeline_run_id", ""),
                "n_monthly_samples": meta_info.get("n_monthly_samples", 0),
                "target": meta_info.get("target", ""),
                "n_factors": meta_info.get("n_factors", 0),
                "n_clusters": meta_info.get("n_clusters", 0),
                "stability_score": meta_info.get("stability_score", 0),
                "linkage": "ward"
            },
            "factors": factors_data,
            "clusters": clusters_data
        }
        
        if os.path.exists(self.atlas_path):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            hist_path = os.path.join(self.history_dir, f"gene_atlas_{ts}.json")
            shutil.copy2(self.atlas_path, hist_path)
            
            hists = sorted(glob.glob(os.path.join(self.history_dir, "gene_atlas_*.json")))
            if len(hists) > 30:
                for h in hists[:-30]:
                    os.remove(h)
        
        with open(self.atlas_path, 'w', encoding='utf-8') as f:
            json.dump(atlas, f, indent=4, ensure_ascii=False)
            
        self.atlas_data = atlas

    def load(self):
        if not os.path.exists(self.atlas_path):
            return {}
        try:
            with open(self.atlas_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "_meta" not in data or "factors" not in data:
                log.warning("[GeneAtlas] gene_atlas.json 缺少 _meta 或 factors 字段，基因图谱可能已过期，建议重新构建。")
            return data
        except Exception as e:
            log.warning(f"[GeneAtlas] 无法解析 gene_atlas.json: {e}")
            return {}

    def get_gene(self, factor_name: str) -> dict:
        if not self.atlas_data or "factors" not in self.atlas_data:
            return None
        return self.atlas_data["factors"].get(factor_name)

    def enforce_diversity(self, ranked_factors: list, max_per_gene: int = 2) -> list:
        if not self.atlas_data or "factors" not in self.atlas_data:
            return ranked_factors
            
        ic_corr_matrix = None
        if os.path.exists(self.cache_path):
            try:
                ic_df = pd.read_parquet(self.cache_path)
                ic_corr_matrix = ic_df.corr(method='spearman')
            except Exception:
                pass
                
        selected = []
        for f in ranked_factors:
            gene_info = self.get_gene(f)
            if not gene_info:
                selected.append(f)
                continue
                
            gid = gene_info['gene_id']
            label = gene_info['gene_label']
            
            same_gene_passed = [x for x in selected if self.get_gene(x) and self.get_gene(x)['gene_id'] == gid]
            
            if len(same_gene_passed) >= max_per_gene:
                blocker = same_gene_passed[-1]
                corr_val = 0.0
                if ic_corr_matrix is not None and f in ic_corr_matrix.columns and blocker in ic_corr_matrix.columns:
                    corr_val = ic_corr_matrix.loc[f, blocker]
                    
                log.info(f"[基因门控] {f} 被拦截，与同簇保留因子 {blocker} 的 IC 相关={corr_val:.3f}，归入 {label}")
                continue
                
            selected.append(f)
            
        return selected
