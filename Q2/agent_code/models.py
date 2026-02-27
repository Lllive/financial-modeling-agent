"""
PyTorch MLP分类器实现
用于Agent的神经网络模型组件
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _MLPNet(nn.Module):
    """3层残差式MLP网络"""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPClassifier:
    """
    sklearn风格的PyTorch MLP分类器包装器
    支持 predict_proba() / fit() 接口，与sklearn Pipeline兼容
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
        device: str = None,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout = dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_losses = []
        self.val_aucs = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 30,
        batch_size: int = 1024,
        lr: float = 1e-3,
    ):
        """训练MLP网络"""
        from sklearn.metrics import roc_auc_score

        device = torch.device(self.device)

        # 构建网络
        self.model = _MLPNet(self.input_dim, self.hidden_dims, self.dropout).to(device)

        # 计算正类权重（处理类别不平衡）
        pos_weight = torch.tensor(
            [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
            dtype=torch.float32
        ).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # 数据集
        X_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
        y_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

        self.train_losses = []
        self.val_aucs = []

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)

            avg_loss = epoch_loss / len(X_train)
            self.train_losses.append(avg_loss)
            scheduler.step()

            # 验证集AUC
            if X_val is not None and y_val is not None:
                proba = self.predict_proba(X_val)
                val_auc = roc_auc_score(y_val, proba)
                self.val_aucs.append(val_auc)

                if (epoch + 1) % 5 == 0:
                    print(f"    Epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f} | val_AUC={val_auc:.4f}")
            else:
                if (epoch + 1) % 5 == 0:
                    print(f"    Epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回正类概率（形状: N,）"""
        device = torch.device(self.device)
        self.model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X.astype(np.float32)).to(device)
            logits = self.model(X_t)
            proba = torch.sigmoid(logits).cpu().numpy()
        return proba

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
