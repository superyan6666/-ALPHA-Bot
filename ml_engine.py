# ml_engine.py — Deep Learning Engine Upgrade (Phase 5)
# 包含改进后的 MLP、分类器、LSTM、集成模型与兼容层

import xgboost as xgb
import pandas as pd
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
import copy

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 可复用组件
# ═════════════════════════════════════════════════════════════════════════════
class GatedResidualBlock(nn.Module):
    """门控残差块：通过 Sigmoid 门控动态调节信息流，防止噪音因子干扰。"""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        # 初始化门控偏置为正，让训练初期门控接近 1（保留信息）
        nn.init.constant_(self.gate.bias, 2.0)  # sigmoid(2.0) ≈ 0.88
        # 主通路用 kaiming 初始化（后续 ReLU）
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        residual = x
        out = self.linear(x)
        out = self.dropout(out)
        gate = torch.sigmoid(self.gate(x))  # (0,1)
        out = out * gate
        out = self.norm(out + residual)      # 残差连接后 LayerNorm
        return out


def _huber_ranking_loss(preds, targets, alpha=0.2):
    """混合损失：HuberLoss + 成对排序损失，引导模型关注相对顺序。"""
    huber = nn.HuberLoss(delta=0.5)(preds, targets)
    diff = preds.unsqueeze(1) - preds.unsqueeze(0)
    labels = (targets.unsqueeze(1) > targets.unsqueeze(0)).float()
    ranking = torch.mean(torch.log(1 + torch.exp(-diff * labels)))
    return huber + alpha * ranking


# ═════════════════════════════════════════════════════════════════════════════
# 1. 核心 MLP 回归模型 (升级版)
# ═════════════════════════════════════════════════════════════════════════════
class PyTorchDLModel:
    def __init__(self, input_dim: int, hidden_dim1=128, hidden_dim2=64, dropout=0.3, **kwargs):
        """
        升级版 PyTorch DL 模型：门控残差 + LayerNorm + 稳健损失 + 早停
        """
        self.device = torch.device("cpu")
        self.input_bn = nn.BatchNorm1d(input_dim)

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.ln1 = nn.LayerNorm(hidden_dim1)
        self.drop1 = nn.Dropout(dropout)

        self.gated_block = GatedResidualBlock(hidden_dim1, dropout=dropout)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.ln2 = nn.LayerNorm(hidden_dim2)
        self.drop2 = nn.Dropout(dropout)

        self.fc_out = nn.Linear(hidden_dim2, 1)

        self._init_weights()
        self.model = nn.Sequential(
            self.input_bn,
            self.fc1,
            self.ln1,
            nn.ReLU(),
            self.drop1,
            self.gated_block,
            self.fc2,
            self.ln2,
            nn.ReLU(),
            self.drop2,
            self.fc_out,
            nn.Tanh()
        ).to(self.device)

        self.criterion = _huber_ranking_loss  # 混合损失
        self.optimizer = None
        self.best_model_state = None

    def _init_weights(self):
        # 隐藏层用 Kaiming 初始化（适合 ReLU）
        for m in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # 输出层用 Xavier 初始化（适合 Tanh）
        nn.init.xavier_normal_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)

    def train(self, df_train, feature_cols, target_col='fwd_ret_real', group_col='date',
              epochs=100, batch_size=2048, early_stop_patience=10, val_size=0.2, lr=0.001):
        log.info(f"Training PyTorchDLModel (upgraded) on {len(df_train)} samples...")

        # 数据清洗
        df_train = df_train.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols + [target_col, group_col]).copy()
        if group_col in df_train.columns:
            df_train[target_col] = df_train.groupby(group_col)[target_col].rank(pct=True) * 2.0 - 1.0

        # 按日期切分训练/验证（时序验证）
        dates = sorted(df_train[group_col].unique())
        split_idx = int(len(dates) * (1 - val_size))
        train_dates = set(dates[:split_idx])
        val_dates = set(dates[split_idx:])

        train_df = df_train[df_train[group_col].isin(train_dates)]
        val_df = df_train[df_train[group_col].isin(val_dates)]

        if len(val_df) < 30:
            log.warning("Validation set too small (<30), disabling early stopping.")
            val_df = train_df  # 退化为全量训练但忽略早停
            early_stop_patience = epochs

        X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(train_df[target_col].values, dtype=torch.float32).view(-1, 1).to(self.device)
        X_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(val_df[target_col].values, dtype=torch.float32).view(-1, 1).to(self.device)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=3, factor=0.5)

        best_val_spearman = -1.0
        patience_counter = 0
        self.best_model_state = copy.deepcopy(self.model.state_dict())

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()

            # 验证
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val).cpu().numpy().flatten()
            val_true = y_val.cpu().numpy().flatten()
            if len(val_true) > 1:
                val_spearman, _ = spearmanr(val_preds, val_true)
                if np.isnan(val_spearman):
                    val_spearman = 0.0
            else:
                val_spearman = 0.0

            log.info(f"Epoch {epoch+1:3d} | Train Loss: {total_loss/len(train_loader):.4f} | Val Spearman: {val_spearman:.4f}")

            scheduler.step(val_spearman)

            if val_spearman > best_val_spearman:
                best_val_spearman = val_spearman
                patience_counter = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                log.info(f"  >> New best model saved (Spearman={val_spearman:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    log.info(f"Early stopping triggered after {early_stop_patience} epochs without improvement.")
                    break

        # 加载最佳权重
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            log.info(f"Loaded best model with Val Spearman: {best_val_spearman:.4f}")
        else:
            log.warning("No best model found; using final state.")

        return self.model

    def predict(self, df_test, feature_cols):
        X_df = df_test[feature_cols].replace([np.inf, -np.inf], np.nan)
        valid_mask = X_df.notna().all(axis=1)
        preds = np.full(len(df_test), np.nan)
        if valid_mask.sum() > 0:
            X_valid = torch.tensor(X_df[valid_mask].values, dtype=torch.float32).to(self.device)
            self.model.eval()
            with torch.no_grad():
                valid_preds = self.model(X_valid).cpu().numpy().flatten()
            preds[valid_mask] = valid_preds
        nan_count = (~valid_mask).sum()
        if nan_count > 0:
            log.warning(f"⚠️ PyTorch 引擎拒绝为 {nan_count} 只特征异常股票提供预测分数。")
        return preds

    def save_model(self, filepath: str):
        log.info(f"Saving PyTorch model to {filepath}")
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath: str):
        log.info(f"Loading PyTorch model from {filepath}")
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))


# ═════════════════════════════════════════════════════════════════════════════
# 2. 分类器 (Meta-Critic) - 同步升级
# ═════════════════════════════════════════════════════════════════════════════
class PyTorchClassifier:
    def __init__(self, input_dim: int, num_classes: int = 3, hidden_dim1=128, hidden_dim2=64, dropout=0.3, **kwargs):
        self.device = torch.device("cpu")
        self.input_bn = nn.BatchNorm1d(input_dim)

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.ln1 = nn.LayerNorm(hidden_dim1)
        self.drop1 = nn.Dropout(dropout)

        self.gated_block = GatedResidualBlock(hidden_dim1, dropout=dropout)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.ln2 = nn.LayerNorm(hidden_dim2)
        self.drop2 = nn.Dropout(dropout)

        self.fc_out = nn.Linear(hidden_dim2, num_classes)

        self._init_weights()
        self.model = nn.Sequential(
            self.input_bn,
            self.fc1,
            self.ln1,
            nn.ReLU(),
            self.drop1,
            self.gated_block,
            self.fc2,
            self.ln2,
            nn.ReLU(),
            self.drop2,
            self.fc_out
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.best_model_state = None
        self.num_classes = num_classes

    def _init_weights(self):
        for m in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        nn.init.xavier_normal_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)

    def train(self, df_train, feature_cols, target_col='target_class', epochs=100, batch_size=2048,
              early_stop_patience=10, val_size=0.2, lr=0.001, group_col='date'):
        log.info(f"Training PyTorchClassifier (upgraded) on {len(df_train)} samples...")
        df_train = df_train.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols + [target_col, group_col]).copy()
        if len(df_train) < 10:
            log.warning("Insufficient data to train PyTorchClassifier. Skipping.")
            return self.model

        # 时间序列切分
        dates = sorted(df_train[group_col].unique())
        split_idx = int(len(dates) * (1 - val_size))
        train_dates = set(dates[:split_idx])
        val_dates = set(dates[split_idx:])
        train_df = df_train[df_train[group_col].isin(train_dates)]
        val_df = df_train[df_train[group_col].isin(val_dates)]

        if len(val_df) < 10:
            val_df = train_df
            early_stop_patience = epochs

        X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(train_df[target_col].values, dtype=torch.long).to(self.device)
        X_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(val_df[target_col].values, dtype=torch.long).to(self.device)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=3, factor=0.5)

        best_val_acc = -1.0
        patience_counter = 0
        self.best_model_state = copy.deepcopy(self.model.state_dict())

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val)
                val_preds = torch.argmax(val_logits, dim=1)
                val_acc = (val_preds == y_val).float().mean().item()

            log.info(f"Classifier Epoch {epoch+1:3d} | Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                log.info(f"  >> New best classifier saved (Acc={val_acc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    log.info(f"Classifier early stopping after {early_stop_patience} epochs.")
                    break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            log.info(f"Loaded best classifier with Val Acc: {best_val_acc:.4f}")
        return self.model

    def predict(self, df_test, feature_cols):
        X_df = df_test[feature_cols].replace([np.inf, -np.inf], np.nan)
        valid_mask = X_df.notna().all(axis=1)
        preds = np.full((len(df_test), self.num_classes), np.nan)
        if valid_mask.sum() > 0:
            X_valid = torch.tensor(X_df[valid_mask].values, dtype=torch.float32).to(self.device)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(X_valid)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds[valid_mask] = probs
        return preds

    def save_model(self, filepath: str):
        log.info(f"Saving PyTorch classifier to {filepath}")
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath: str):
        log.info(f"Loading PyTorch classifier from {filepath}")
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))


# ═════════════════════════════════════════════════════════════════════════════
# 3. LSTM 模型 (保持原有逻辑，仅微调初始化和损失函数)
# ═════════════════════════════════════════════════════════════════════════════
class _LSTMNet(nn.Module):
    """LSTM network for temporal cross-sectional ranking."""
    def __init__(self, input_dim, hidden_dim=48, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        # 初始化
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, 0)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1]
        return self.head(out)


class LSTMModel:
    """LSTM-based temporal model (improved initialization)."""
    def __init__(self, input_dim: int, seq_len: int = 10, hidden_dim: int = 48, **kwargs):
        self.device = torch.device("cpu")
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.model = _LSTMNet(input_dim, hidden_dim=hidden_dim).to(self.device)
        self.criterion = nn.HuberLoss(delta=0.5)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0003, weight_decay=1e-4)

    def _build_sequences(self, df, feature_cols, target_col=None):
        df = df.sort_values(['code', 'date']).copy()
        codes = df['code'].unique()

        all_X, all_y, all_idx = [], [], []
        for code in codes:
            sub = df[df['code'] == code]
            if len(sub) < self.seq_len:
                continue
            vals = sub[feature_cols].values.astype(np.float32)
            for i in range(self.seq_len, len(sub)):
                seq = vals[i - self.seq_len:i]
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
        result = pd.Series(np.nan, index=df_test.index)
        for i, idx in enumerate(valid_idx):
            if idx in result.index:
                result.loc[idx] = preds[i]
        return result.fillna(0).values


# ═════════════════════════════════════════════════════════════════════════════
# 4. Ensemble 集成模型
# ═════════════════════════════════════════════════════════════════════════════
class EnsembleModel:
    """Ensemble of upgraded MLP + LSTM."""
    def __init__(self, input_dim: int, seq_len: int = 10, weights=None, **kwargs):
        self.mlp = PyTorchDLModel(input_dim)
        self.lstm = LSTMModel(input_dim, seq_len=seq_len)
        self.weights = weights or [0.6, 0.4]

    def train(self, df_train, feature_cols, target_col='fwd_ret_real', group_col='date',
              epochs=100, batch_size=2048):
        log.info(f"[Ensemble] Training MLP (weight={self.weights[0]})...")
        self.mlp.train(df_train, feature_cols, target_col, group_col, epochs=epochs, batch_size=batch_size)
        log.info(f"[Ensemble] Training LSTM (weight={self.weights[1]})...")
        self.lstm.train(df_train, feature_cols, target_col, group_col, epochs=epochs, batch_size=batch_size)

    def predict(self, df_test, feature_cols):
        mlp_preds = self.mlp.predict(df_test, feature_cols)
        lstm_preds = self.lstm.predict(df_test, feature_cols)
        from scipy.stats import rankdata
        mlp_ranks = rankdata(mlp_preds) / len(mlp_preds)
        lstm_ranks = rankdata(lstm_preds) / len(lstm_preds)
        return self.weights[0] * mlp_ranks + self.weights[1] * lstm_ranks


# ═════════════════════════════════════════════════════════════════════════════
# 5. XGBoostLTR (Deprecated, 保持兼容)
# ═════════════════════════════════════════════════════════════════════════════
class XGBoostLTR:
    def __init__(self, **kwargs):
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
        X_test = df_test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.model.predict(X_test)

    def save_model(self, filepath: str):
        self.model.save_model(filepath)

    def load_model(self, filepath: str):
        self.model.load_model(filepath)


# ═════════════════════════════════════════════════════════════════════════════
# 6. 流动性过滤门 (保持不变)
# ═════════════════════════════════════════════════════════════════════════════
def apply_liquidity_gate(panel, amihud_col='amihud_20', threshold_pct=0.90):
    log.info(f"Applying LiquidityGate (Dropping top {1 - threshold_pct:.0%} highest {amihud_col} stocks)...")
    panel['amihud_rank'] = panel.groupby('date')[amihud_col].rank(pct=True)
    initial_count = len(panel)
    filtered_panel = panel[panel['amihud_rank'] <= threshold_pct].copy()
    dropped_count = initial_count - len(filtered_panel)
    log.info(f"LiquidityGate dropped {dropped_count} rows out of {initial_count}.")
    return filtered_panel
