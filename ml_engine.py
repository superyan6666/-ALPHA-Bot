import xgboost as xgb
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)

class XGBoostLTR:
    def __init__(self, **kwargs):
        """
        Initialize XGBoost Learning-to-Rank model.
        Uses rank:pairwise objective to learn relative ordering within each date.
        """
        default_params = {
            'tree_method': 'hist',
            'objective': 'rank:pairwise',
            'learning_rate': 0.1,
            'max_depth': 4,
            'n_estimators': 200,
            'reg_alpha': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        default_params.update(kwargs)
        self.model = xgb.XGBRanker(**default_params)
        
    def train(self, df_train, feature_cols, target_col='fwd_ret_real', group_col='date'):
        log.info(f"Training XGBRanker on {len(df_train)} samples with {len(feature_cols)} features...")
        # Drop NaNs
        df_train = df_train.dropna(subset=feature_cols + [target_col, group_col]).copy()
        
        # XGBRanker requires the dataset to be sorted by the group column
        df_train = df_train.sort_values(by=group_col)
        
        X_train = df_train[feature_cols]
        y_train = df_train[target_col]
        
        # Compute group sizes (number of stocks per day)
        groups = df_train.groupby(group_col).size().values
        
        self.model.fit(X_train, y_train, group=groups)
        log.info("XGBRanker training completed.")
        return self.model
        
    def predict(self, df_test, feature_cols):
        # Fill NaNs in features with median or 0 to allow prediction
        X_test = df_test[feature_cols].fillna(0)
        return self.model.predict(X_test)
        
    def get_feature_importance(self, feature_cols):
        importance = self.model.feature_importances_
        imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importance})
        imp_df = imp_df.sort_values('importance', ascending=False)
        return imp_df
        
    def save_model(self, filepath: str):
        log.info(f"Saving XGBRanker model to {filepath}")
        self.model.save_model(filepath)
        
    def load_model(self, filepath: str):
        log.info(f"Loading XGBRanker model from {filepath}")
        self.model.load_model(filepath)

def apply_liquidity_gate(panel, amihud_col='amihud_20', threshold_pct=0.90):
    """
    LiquidityGate: Filters out the top (1 - threshold_pct) least liquid stocks.
    e.g., threshold_pct=0.90 means drop the top 10% highest Amihud stocks daily.
    """
    log.info(f"Applying LiquidityGate (Dropping top {1 - threshold_pct:.0%} highest {amihud_col} stocks)...")
    
    # Calculate daily cross-sectional rank of Amihud (higher = less liquid)
    panel['amihud_rank'] = panel.groupby('date')[amihud_col].rank(pct=True)
    
    # Keep only those <= threshold
    initial_count = len(panel)
    filtered_panel = panel[panel['amihud_rank'] <= threshold_pct].copy()
    dropped_count = initial_count - len(filtered_panel)
    
    log.info(f"LiquidityGate dropped {dropped_count} rows out of {initial_count}.")
    return filtered_panel
