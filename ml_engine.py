import xgboost as xgb
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class PyTorchDLModel:
    def __init__(self, input_dim: int, **kwargs):
        """
        [CRUCIBLE PROTOCOL] Deep Learning Engine initialized.
        Uses a lightweight feed-forward/LSTM structure to adhere to ARM 4C 24G constraints.
        """
        self.device = torch.device("cpu") # ARM local runs on CPU
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def train(self, df_train, feature_cols, target_col='fwd_ret_real', group_col='date', epochs=5, batch_size=2048):
        log.info(f"Training PyTorchDLModel on {len(df_train)} samples...")
        df_train = df_train.dropna(subset=feature_cols + [target_col]).copy()
        
        X_train = torch.tensor(df_train[feature_cols].values, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(df_train[target_col].values, dtype=torch.float32).view(-1, 1).to(self.device)
        
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            log.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")
        return self.model
        
    def predict(self, df_test, feature_cols):
        X_test = torch.tensor(df_test[feature_cols].fillna(0).values, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_test).cpu().numpy().flatten()
        return preds
        
    def save_model(self, filepath: str):
        log.info(f"Saving PyTorch model to {filepath}")
        torch.save(self.model.state_dict(), filepath)
        
    def load_model(self, filepath: str):
        log.info(f"Loading PyTorch model from {filepath}")
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))


class _LSTMNet(nn.Module):
    """LSTM network for temporal cross-sectional ranking."""
    def __init__(self, input_dim, hidden_dim=48, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, batch, hidden_dim)
        out = h_n[-1]  # last layer hidden state: (batch, hidden_dim)
        return self.head(out)


class LSTMModel:
    """
    [CRUCIBLE Level 3] LSTM-based temporal model.
    Takes a lookback window of features per stock to capture sequential patterns.
    """
    def __init__(self, input_dim: int, seq_len: int = 10, hidden_dim: int = 48, **kwargs):
        self.device = torch.device("cpu")
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.model = _LSTMNet(input_dim, hidden_dim=hidden_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)

    def _build_sequences(self, df, feature_cols, target_col=None):
        """Build (stock, date) → lookback sequences. Returns X_seq, y, valid_indices."""
        df = df.sort_values(['code', 'date']).copy()
        codes = df['code'].unique()
        
        all_X, all_y, all_idx = [], [], []
        for code in codes:
            sub = df[df['code'] == code]
            if len(sub) < self.seq_len:
                continue
            vals = sub[feature_cols].values.astype(np.float32)
            for i in range(self.seq_len, len(sub)):
                seq = vals[i - self.seq_len:i]  # (seq_len, n_features)
                if np.isnan(seq).any():
                    continue
                all_X.append(seq)
                all_idx.append(sub.index[i])
                if target_col:
                    all_y.append(sub[target_col].iloc[i])

        X_seq = np.stack(all_X) if all_X else np.empty((0, self.seq_len, len(feature_cols)))
        y_arr = np.array(all_y, dtype=np.float32) if all_y else np.empty(0)
        return X_seq, y_arr, all_idx

    def train(self, df_train, feature_cols, target_col='fwd_ret_real', group_col='date',
              epochs=5, batch_size=2048):
        log.info(f"Building LSTM sequences (seq_len={self.seq_len}) on {len(df_train)} samples...")
        X_seq, y_arr, _ = self._build_sequences(df_train, feature_cols, target_col)
        log.info(f"LSTM sequences built: {len(X_seq)} valid sequences")

        if len(X_seq) == 0:
            log.warning("No valid sequences built. Skipping training.")
            return self.model

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_arr, dtype=torch.float32).view(-1, 1).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for bx, by in loader:
                self.optimizer.zero_grad()
                preds = self.model(bx)
                loss = self.criterion(preds, by)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            log.info(f"LSTM Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")
        return self.model

    def predict(self, df_test, feature_cols):
        X_seq, _, valid_idx = self._build_sequences(df_test, feature_cols)
        if len(X_seq) == 0:
            return np.zeros(len(df_test))

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_t).cpu().numpy().flatten()

        # Map predictions back to original indices
        result = pd.Series(np.nan, index=df_test.index)
        for i, idx in enumerate(valid_idx):
            if idx in result.index:
                result.loc[idx] = preds[i]
        return result.fillna(0).values


class EnsembleModel:
    """
    [CRUCIBLE Level 3] Ensemble of MLP + LSTM.
    Blends predictions from multiple model types for diversification.
    """
    def __init__(self, input_dim: int, seq_len: int = 10, weights=None, **kwargs):
        self.mlp = PyTorchDLModel(input_dim)
        self.mlp.optimizer = optim.Adam(self.mlp.model.parameters(), lr=0.0003)
        self.lstm = LSTMModel(input_dim, seq_len=seq_len)
        self.weights = weights or [0.6, 0.4]  # MLP weight, LSTM weight

    def train(self, df_train, feature_cols, target_col='fwd_ret_real', group_col='date',
              epochs=5, batch_size=2048):
        log.info(f"[Ensemble] Training MLP (weight={self.weights[0]})...")
        self.mlp.train(df_train, feature_cols, target_col, group_col, epochs, batch_size)
        log.info(f"[Ensemble] Training LSTM (weight={self.weights[1]})...")
        self.lstm.train(df_train, feature_cols, target_col, group_col, epochs, batch_size)

    def predict(self, df_test, feature_cols):
        mlp_preds = self.mlp.predict(df_test, feature_cols)
        lstm_preds = self.lstm.predict(df_test, feature_cols)
        # Rank-based blending to avoid scale mismatch
        from scipy.stats import rankdata
        mlp_ranks = rankdata(mlp_preds) / len(mlp_preds)
        lstm_ranks = rankdata(lstm_preds) / len(lstm_preds)
        return self.weights[0] * mlp_ranks + self.weights[1] * lstm_ranks

class XGBoostLTR:
    def __init__(self, **kwargs):
        """
        [DEPRECATED] Legacy XGBoost Learning-to-Rank model.
        Migrating to PyTorchDLModel for next-gen alpha discovery.
        """
        log.warning("XGBoostLTR is deprecated under CRUCIBLE rules. Please use PyTorchDLModel.")
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
        df_train = df_train.dropna(subset=feature_cols + [target_col, group_col]).copy()
        df_train = df_train.sort_values(by=group_col)
        X_train = df_train[feature_cols]
        y_train = df_train[target_col]
        groups = df_train.groupby(group_col).size().values
        
        self.model.fit(X_train, y_train, group=groups)
        return self.model
        
    def predict(self, df_test, feature_cols):
        X_test = df_test[feature_cols].fillna(0)
        return self.model.predict(X_test)
        
    def save_model(self, filepath: str):
        self.model.save_model(filepath)
        
    def load_model(self, filepath: str):
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
