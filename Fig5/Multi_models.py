import os
import ast
import json
import math
import random
from typing import Dict, Optional, List, Tuple

# =========================
# 0. 在 import torch 之前设置环境变量
# =========================
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
    f1_score,
)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. 全局配置
# ============================================================
SEED = 42

# ---------- 二分类训练数据 ----------
BINARY_DATA_PATH = "Traindata_2d_Sensitivity_300.csv"

# ---------- 8分类预测数据 ----------
MULTI8_DATA_PATH = "Traindata_8d.csv"
G = 8

MULTI_TRUE_LABEL_COL = None
MULTI_TRUE_LABEL_COL_CANDIDATES = [
    "label", "true_label", "target", "y", "Y",
    "alpha*", "alpha_star", "best_label", "final_label", "optimal_label"
]

# ---------- 保存目录 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "Model_Compare")

MAX_EPOCHS = 200
PATIENCE = 10

MODEL_NAMES = ["LSTM", "CNN", "GRU", "Transformer", "MLP"]

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 2. 固定随机种子：保证可复现
# ============================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 避免 Transformer 在部分环境下走到非确定性 SDP 内核
    if hasattr(torch.backends, "cuda"):
        try:
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(False)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_generator(seed: int = 42):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# 3. 数据读取与预处理函数（二分类）
# ============================================================
def load_binary_data(data_path: str):
    data = pd.read_csv(data_path)

    infection_data = []
    for i in range(14):
        day_col = f"gamI-Day{i + 1}"
        parsed = data[day_col].apply(ast.literal_eval)
        infection_data.append(np.stack(parsed.to_numpy()))
    X_infected = np.stack(infection_data, axis=1)

    Ts = np.repeat(data["Ts"].values[:, np.newaxis], 14, axis=1)[:, :, np.newaxis]
    N1 = np.repeat(data["N1"].values[:, np.newaxis], 14, axis=1)[:, :, np.newaxis]
    N2 = np.repeat(data["N2"].values[:, np.newaxis], 14, axis=1)[:, :, np.newaxis]

    X_all = np.concatenate([X_infected, Ts, N1, N2], axis=2)

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(data["alpha*"].values)

    return data, X_all.astype(np.float32), y_all.astype(np.int64), label_encoder


def fit_transform_3d(X: np.ndarray, scaler: MinMaxScaler):
    X_flat = X.reshape(-1, X.shape[2])
    X_scaled = scaler.fit_transform(X_flat)
    return X_scaled.reshape(X.shape)


def transform_3d(X: np.ndarray, scaler: MinMaxScaler):
    X_flat = X.reshape(-1, X.shape[2])
    X_scaled = scaler.transform(X_flat)
    return X_scaled.reshape(X.shape)


def make_class_weights(y: np.ndarray, num_classes: int):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)

    total = len(y)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total / (num_classes * counts[c])
        else:
            weights[c] = 0.0
    return torch.tensor(weights, dtype=torch.float32)


def build_single_loader(X, y, batch_size: int, shuffle: bool, seed: int = 42):
    g = make_generator(seed)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return loader


def build_loaders(X_train, y_train, X_val, y_val, batch_size: int, seed: int = 42):
    train_loader = build_single_loader(X_train, y_train, batch_size, shuffle=True, seed=seed)
    val_loader = build_single_loader(X_val, y_val, batch_size, shuffle=False, seed=seed)
    return train_loader, val_loader


# ============================================================
# 4. 模型定义
# ============================================================
def get_activation(name: str):
    if name == "ReLU":
        return nn.ReLU()
    if name == "Sigmoid":
        return nn.Sigmoid()
    if name == "Tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class BiLSTMClassifier2d(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout1: float, activation_name: str, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout1 if num_layers > 1 else 0.0,
        )
        self.dropout1 = nn.Dropout(dropout1)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.act = get_activation(activation_name)
        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        h_last = self.dropout1(h_last)
        h_last = self.fc1(h_last)
        h_last = self.act(h_last)
        h_last = self.dropout2(h_last)
        return self.fc2(h_last)


class LSTMClassifier2d(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout1: float, activation_name: str, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout1 if num_layers > 1 else 0.0,
        )
        self.dropout1 = nn.Dropout(dropout1)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.act = get_activation(activation_name)
        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        h_last = self.dropout1(h_last)
        h_last = self.fc1(h_last)
        h_last = self.act(h_last)
        h_last = self.dropout2(h_last)
        return self.fc2(h_last)


class GRUClassifier2d(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout1: float, activation_name: str, num_classes: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout1 if num_layers > 1 else 0.0,
        )
        self.dropout1 = nn.Dropout(dropout1)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.act = get_activation(activation_name)
        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        _, h_n = self.gru(x)
        h_last = h_n[-1]
        h_last = self.dropout1(h_last)
        h_last = self.fc1(h_last)
        h_last = self.act(h_last)
        h_last = self.dropout2(h_last)
        return self.fc2(h_last)


class CNNClassifier2d(nn.Module):
    def __init__(self, input_size: int, num_filters: int, kernel_size: int, dropout1: float, activation_name: str, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.act = get_activation(activation_name)
        self.dropout = nn.Dropout(dropout1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_filters, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier2d(nn.Module):
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, ff_dim: int, dropout1: float, activation_name: str, num_classes: int):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"Transformer 要求 d_model % nhead == 0，但当前 d_model={d_model}, nhead={nhead}")
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout1,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout1)
        self.fc1 = nn.Linear(d_model, 32)
        self.act = get_activation(activation_name)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNClassifier2d(nn.Module):
    def __init__(self, input_size: int, num_channels: int, num_levels: int, kernel_size: int, dropout1: float, activation_name: str, num_classes: int):
        super().__init__()
        layers = []
        in_ch = input_size
        for i in range(num_levels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, num_channels, kernel_size, dilation, dropout1))
            in_ch = num_channels
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_channels, 32)
        self.act = get_activation(activation_name)
        self.dropout = nn.Dropout(dropout1)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)


class MLPClassifier2d(nn.Module):
    def __init__(self, seq_len: int, input_size: int, hidden_dim: int, dropout1: float, activation_name: str, num_classes: int):
        super().__init__()
        in_dim = seq_len * input_size
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.act = get_activation(activation_name)
        self.dropout = nn.Dropout(dropout1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc3(x)


# ============================================================
# 5. 模型与超参数网格
# ============================================================
def build_model(model_name: str, params: Dict, input_size: int, seq_len: int, num_classes: int):
    if model_name == "BiLSTM":
        return BiLSTMClassifier2d(
            input_size=input_size,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout1=params["dropout1"],
            activation_name=params["activation"],
            num_classes=num_classes,
        )
    if model_name == "LSTM":
        return LSTMClassifier2d(
            input_size=input_size,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout1=params["dropout1"],
            activation_name=params["activation"],
            num_classes=num_classes,
        )
    if model_name == "GRU":
        return GRUClassifier2d(
            input_size=input_size,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout1=params["dropout1"],
            activation_name=params["activation"],
            num_classes=num_classes,
        )
    if model_name == "CNN":
        return CNNClassifier2d(
            input_size=input_size,
            num_filters=params["num_filters"],
            kernel_size=params["kernel_size"],
            dropout1=params["dropout1"],
            activation_name=params["activation"],
            num_classes=num_classes,
        )
    if model_name == "Transformer":
        return TransformerClassifier2d(
            input_size=input_size,
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_layers=params["num_layers"],
            ff_dim=params["ff_dim"],
            dropout1=params["dropout1"],
            activation_name=params["activation"],
            num_classes=num_classes,
        )

    if model_name == "MLP":
        return MLPClassifier2d(
            seq_len=seq_len,
            input_size=input_size,
            hidden_dim=params["hidden_dim"],
            dropout1=params["dropout1"],
            activation_name=params["activation"],
            num_classes=num_classes,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


# def get_model_param_grids() -> Dict[str, Dict]:
#     return {
#         "BiLSTM": {
#             "learning_rate": [0.0001, 0.001, 0.01],
#             "hidden_size": [64, 128],
#             "num_layers": [1, 2],
#             "batch_size": [32, 64],
#             "dropout1": [0.2],
#             "activation": ["ReLU"],
#         },
#         "LSTM": {
#             "learning_rate": [0.0001, 0.001, 0.01],
#             "hidden_size": [64, 128],
#             "num_layers": [1, 2],
#             "batch_size": [32, 64],
#             "dropout1": [0.2],
#             "activation": ["ReLU"],
#         },
#         "GRU": {
#             "learning_rate": [0.0001, 0.001, 0.01],
#             "hidden_size": [64, 128],
#             "num_layers": [1, 2],
#             "batch_size": [32, 64],
#             "dropout1": [0.2],
#             "activation": ["ReLU"],
#         },
#         "CNN": {
#             "learning_rate": [0.0001, 0.001, 0.01],
#             "num_filters": [32, 64],
#             "kernel_size": [3, 5],
#             "batch_size": [32, 64],
#             "dropout1": [0.2],
#             "activation": ["ReLU"],
#         },
#         "Transformer": {
#             "learning_rate": [0.0001, 0.001],
#             "d_model": [32, 64],
#             "nhead": [4],
#             "num_layers": [1, 2],
#             "ff_dim": [64, 128],
#             "batch_size": [32, 64],
#             "dropout1": [0.1, 0.2],
#             "activation": ["ReLU"],
#         },
#         "MLP": {
#             "learning_rate": [0.0001, 0.001, 0.01],
#             "hidden_dim": [64, 128],
#             "batch_size": [32, 64],
#             "dropout1": [0.2],
#             "activation": ["ReLU"],
#         },
#     }
def get_model_param_grids() -> Dict[str, Dict]:
    return {
        "BiLSTM": {
            "learning_rate": [0.01],
            "hidden_size": [64],
            "num_layers": [2],
            "batch_size": [64],
            "dropout1": [0.2],
            "activation": ["ReLU"],
        },
        "LSTM": {
            "learning_rate": [0.01],
            "hidden_size": [64],
            "num_layers": [2],
            "batch_size": [32],
            "dropout1": [0.2],
            "activation": ["ReLU"],
        },
        "GRU": {
            "learning_rate": [0.01],
            "hidden_size": [64],
            "num_layers": [1],
            "batch_size": [64],
            "dropout1": [0.2],
            "activation": ["ReLU"],
        },
        "CNN": {
            "learning_rate": [0.001],
            "num_filters": [32],
            "kernel_size": [5],
            "batch_size": [32],
            "dropout1": [0.2],
            "activation": ["ReLU"],
        },
        "Transformer": {
            "learning_rate": [0.001],
            "d_model": [64],
            "nhead": [4],
            "num_layers": [2],
            "ff_dim": [64],
            "batch_size": [32],
            "dropout1": [0.1],
            "activation": ["ReLU"],
        },
        "MLP": {
            "learning_rate": [0.001],
            "hidden_dim": [128],
            "batch_size": [64],
            "dropout1": [0.2],
            "activation": ["ReLU"],
        },
    }

# ============================================================
# 6. 训练 / 验证 / 推断函数（二分类）
# ============================================================
def evaluate_loss_and_accuracy(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_true.append(yb.cpu().numpy())
            all_pred.append(preds.cpu().numpy())

    avg_loss = total_loss / len(loader)
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc


def get_positive_prob_and_true(model, loader, device, positive_class_index: int = 1):
    model.eval()
    probs_all = []
    y_all = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:, positive_class_index]

            probs_all.append(probs.cpu().numpy())
            y_all.append(yb.cpu().numpy())

    return np.concatenate(probs_all), np.concatenate(y_all)


def train_one_split(
    model_name: str,
    X_train,
    y_train,
    X_val,
    y_val,
    params: Dict,
    num_classes: int,
    input_size: int,
    seq_len: int,
    seed: int = 42,
    max_epochs: int = 200,
    patience: int = 10,
    device: str = "cpu",
):
    seed_everything(seed)

    scaler = MinMaxScaler()
    X_train_scaled = fit_transform_3d(X_train, scaler)
    X_val_scaled = transform_3d(X_val, scaler)

    train_loader, val_loader = build_loaders(
        X_train_scaled, y_train, X_val_scaled, y_val,
        batch_size=params["batch_size"],
        seed=seed,
    )

    class_weights = make_class_weights(y_train, num_classes).to(device)

    model = build_model(
        model_name=model_name,
        params=params,
        input_size=input_size,
        seq_len=seq_len,
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_loss = float("inf")
    best_state = None
    wait = 0
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(max_epochs):
        model.train()
        running_train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        train_loss = running_train_loss / len(train_loader)
        val_loss, val_acc = evaluate_loss_and_accuracy(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            best_epoch = epoch + 1
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    val_loss_best, val_acc_best = evaluate_loss_and_accuracy(model, val_loader, criterion, device)

    return {
        "model": model,
        "scaler": scaler,
        "best_state": best_state,
        "best_val_loss": float(val_loss_best),
        "best_val_acc": float(val_acc_best),
        "best_epoch": int(best_epoch),
        "history": history,
        "val_loader": val_loader,
    }


def choose_threshold_by_youden_j(model, val_loader, device, positive_class_index: int = 1):
    val_prob, val_true = get_positive_prob_and_true(model, val_loader, device, positive_class_index)

    if len(np.unique(val_true)) < 2:
        print("Warning: validation set has only one class; threshold is set to 0.5.")
        return 0.5, None

    y_true_bin = (val_true == positive_class_index).astype(int)
    fpr, tpr, thresholds = roc_curve(y_true_bin, val_prob, pos_label=1)
    finite_mask = np.isfinite(thresholds)

    fpr = fpr[finite_mask]
    tpr = tpr[finite_mask]
    thresholds = thresholds[finite_mask]

    if len(thresholds) == 0:
        print("Warning: no finite ROC thresholds found; threshold is set to 0.5.")
        return 0.5, None

    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_idx])

    info = {
        "threshold": best_threshold,
        "youden_j": float(j_scores[best_idx]),
        "tpr": float(tpr[best_idx]),
        "fpr": float(fpr[best_idx]),
    }
    return best_threshold, info


def evaluate_on_holdout(model, scaler, X_test, y_test, batch_size, device, threshold=0.5, positive_class_index: int = 1):
    X_test_scaled = transform_3d(X_test, scaler)
    test_loader = build_single_loader(X_test_scaled, y_test, batch_size, shuffle=False, seed=SEED)
    test_prob, test_true = get_positive_prob_and_true(model, test_loader, device, positive_class_index)

    test_pred = np.where(test_prob >= threshold, positive_class_index, 1 - positive_class_index)
    acc = accuracy_score(test_true, test_pred)

    if len(np.unique(test_true)) >= 2:
        y_true_bin = (test_true == positive_class_index).astype(int)
        test_auc = roc_auc_score(y_true_bin, test_prob)
    else:
        test_auc = np.nan

    return {
        "test_true": test_true,
        "test_prob": test_prob,
        "test_pred": test_pred,
        "accuracy": float(acc),
        "auc": float(test_auc) if not np.isnan(test_auc) else np.nan,
    }


# ============================================================
# 7. 多分类（8分类）辅助函数
# ============================================================
def series_to_float_array(series: pd.Series) -> np.ndarray:
    if np.issubdtype(series.dtype, np.number):
        return series.to_numpy(dtype=np.float32)

    vals = []
    for v in series:
        if isinstance(v, (int, float, np.integer, np.floating)):
            vals.append(float(v))
        else:
            s = str(v).strip()
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, np.ndarray)):
                    if len(parsed) != 1:
                        raise ValueError(f"列 {series.name} 中出现了长度不为1的列表，无法转成标量。")
                    vals.append(float(parsed[0]))
                else:
                    vals.append(float(parsed))
            except Exception:
                vals.append(float(s))
    return np.asarray(vals, dtype=np.float32)


def build_pair_input_multiclass(df: pd.DataFrame, left_id: int, right_id: int) -> np.ndarray:
    infection_data = []
    for day in range(1, 15):
        col_left = f"gamI{left_id}-Day{day}"
        col_right = f"gamI{right_id}-Day{day}"
        arr_left = series_to_float_array(df[col_left])
        arr_right = series_to_float_array(df[col_right])
        infection_data.append(np.stack([arr_left, arr_right], axis=-1))

    X_infected = np.stack(infection_data, axis=1)
    Ts = np.repeat(series_to_float_array(df["Ts"])[:, np.newaxis], 14, axis=1)[:, :, np.newaxis]
    N_left = np.repeat(series_to_float_array(df[f"N{left_id}"])[:, np.newaxis], 14, axis=1)[:, :, np.newaxis]
    N_right = np.repeat(series_to_float_array(df[f"N{right_id}"])[:, np.newaxis], 14, axis=1)[:, :, np.newaxis]

    X_all = np.concatenate([X_infected, Ts, N_left, N_right], axis=2)
    return X_all.astype(np.float32)


def build_tensor_only_loader(X: np.ndarray, batch_size: int, seed: int = 42):
    g = make_generator(seed)
    x_tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(x_tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return loader


def predict_pairwise_right_probs(model, scaler, X_pair: np.ndarray, batch_size: int, device, right_choice_index: int = 1):
    X_scaled = transform_3d(X_pair, scaler)
    loader = build_tensor_only_loader(X_scaled, batch_size=batch_size, seed=SEED)

    probs_all = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:, right_choice_index]
            probs_all.append(probs.cpu().numpy())
    return np.concatenate(probs_all)


def detect_multiclass_true_label_col(df: pd.DataFrame, g: int) -> Optional[str]:
    if MULTI_TRUE_LABEL_COL is not None:
        if MULTI_TRUE_LABEL_COL not in df.columns:
            raise ValueError(f"指定的 MULTI_TRUE_LABEL_COL={MULTI_TRUE_LABEL_COL} 不在数据列中。")
        return MULTI_TRUE_LABEL_COL

    valid_sets = [set(range(1, g + 1)), set(range(0, g))]

    for col in MULTI_TRUE_LABEL_COL_CANDIDATES:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().all():
                uniq = set(vals.astype(int).unique())
                if any(uniq.issubset(vs) for vs in valid_sets):
                    return col

    for col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().all():
            uniq = set(vals.astype(int).unique())
            if len(uniq) > 1 and any(uniq.issubset(vs) for vs in valid_sets):
                return col

    return None


def normalize_multiclass_true_labels(y_true_raw: np.ndarray, g: int) -> np.ndarray:
    y_true_raw = np.asarray(y_true_raw).astype(int)
    uniq = set(np.unique(y_true_raw))
    if uniq.issubset(set(range(1, g + 1))):
        return y_true_raw
    if uniq.issubset(set(range(0, g))):
        return y_true_raw + 1
    raise ValueError(f"8分类真实标签不在 1..{g} 或 0..{g - 1} 范围内，实际 unique={sorted(list(uniq))}")


def predict_multiclass_probs_from_binary(
    real_data: pd.DataFrame,
    model,
    scaler,
    g: int,
    batch_size: int,
    device,
    right_choice_index: int = 1,
):
    n = len(real_data)
    pairwise_right_probs = {}

    for left in range(1, g):
        for right in range(left + 1, g + 1):
            X_pair = build_pair_input_multiclass(real_data, left, right)
            p_right = predict_pairwise_right_probs(
                model=model,
                scaler=scaler,
                X_pair=X_pair,
                batch_size=batch_size,
                device=device,
                right_choice_index=right_choice_index,
            )
            pairwise_right_probs[(left, right)] = p_right

    class_probs = np.zeros((n, g), dtype=np.float64)

    for i in range(n):
        dist = np.zeros(g + 1, dtype=np.float64)

        p_12_right = pairwise_right_probs[(1, 2)][i]
        dist[1] = 1.0 - p_12_right
        dist[2] = p_12_right

        for challenger in range(3, g + 1):
            new_dist = np.zeros(g + 1, dtype=np.float64)
            for champion in range(1, challenger):
                if dist[champion] == 0:
                    continue
                p_right = pairwise_right_probs[(champion, challenger)][i]
                new_dist[champion] += dist[champion] * (1.0 - p_right)
                new_dist[challenger] += dist[champion] * p_right
            dist = new_dist

        probs = dist[1:g + 1]
        denom = probs.sum()
        if denom <= 0:
            probs = np.ones(g, dtype=np.float64) / g
        else:
            probs = probs / denom
        class_probs[i] = probs

    pred_labels = np.argmax(class_probs, axis=1) + 1
    return pred_labels.astype(int), class_probs


# ============================================================
# 8. 结果保存与评估
# ============================================================
def save_binary_outputs(model_name: str, model_dir: str, label_encoder, retrain_result: Dict, test_result: Dict, best_threshold: float, positive_class_index: int, negative_class_index: int):
    binary_save_dir = os.path.join(model_dir, "binary_holdout_cv")
    os.makedirs(binary_save_dir, exist_ok=True)

    target_names = [str(x) for x in label_encoder.classes_]
    precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(
        test_result["test_true"],
        test_result["test_pred"],
        labels=[negative_class_index, positive_class_index],
        zero_division=0,
    )

    binary_metrics = {
        "model_name": model_name,
        "accuracy": float(test_result["accuracy"]),
        "auc": None if np.isnan(test_result["auc"]) else float(test_result["auc"]),
        "negative_class_label": str(label_encoder.classes_[negative_class_index]),
        "positive_class_label": str(label_encoder.classes_[positive_class_index]),
        "negative_precision": float(precision_arr[0]),
        "negative_recall": float(recall_arr[0]),
        "positive_precision": float(precision_arr[1]),
        "positive_recall": float(recall_arr[1]),
        "positive_f1": float(f1_arr[1]),
        "threshold": float(best_threshold),
        "retrain_best_epoch": int(retrain_result["best_epoch"]),
        "retrain_best_val_loss": float(retrain_result["best_val_loss"]),
        "retrain_best_val_acc": float(retrain_result["best_val_acc"]),
    }

    print(f"\n===== {model_name}: Final Evaluation on Hold-out Test (Binary) =====")
    print(f"Accuracy: {binary_metrics['accuracy']:.4f}")
    print(f"AUC: {binary_metrics['auc'] if binary_metrics['auc'] is None else format(binary_metrics['auc'], '.4f')}")
    print(f"Negative Precision ({binary_metrics['negative_class_label']}): {binary_metrics['negative_precision']:.4f}")
    print(f"Negative Recall    ({binary_metrics['negative_class_label']}): {binary_metrics['negative_recall']:.4f}")
    print(f"Positive Precision ({binary_metrics['positive_class_label']}): {binary_metrics['positive_precision']:.4f}")
    print(f"Positive Recall    ({binary_metrics['positive_class_label']}): {binary_metrics['positive_recall']:.4f}")
    print(f"Positive F1        ({binary_metrics['positive_class_label']}): {binary_metrics['positive_f1']:.4f}")

    report_text = classification_report(
        test_result["test_true"],
        test_result["test_pred"],
        labels=np.arange(len(target_names)),
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    print("\nClassification Report (Hold-out Test):")
    print(report_text)
    with open(os.path.join(binary_save_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    cm_bin = confusion_matrix(
        test_result["test_true"],
        test_result["test_pred"],
        labels=np.arange(len(target_names)),
    )
    disp_bin = ConfusionMatrixDisplay(confusion_matrix=cm_bin, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp_bin.plot(ax=ax, values_format="d")
    ax.set_title(f"{model_name} Binary Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(binary_save_dir, "confusion_matrix_holdout_test.png"), dpi=300)
    plt.close()

    if len(np.unique(test_result["test_true"])) >= 2:
        y_true_bin = (test_result["test_true"] == positive_class_index).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, test_result["test_prob"], pos_label=1)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {test_result['auc']:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{model_name} Binary ROC Curve")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(binary_save_dir, "roc_curve_holdout_test.png"), dpi=300)
        plt.close()

    torch.save(retrain_result["best_state"], os.path.join(binary_save_dir, "best_model_state_dict.pth"))
    joblib.dump(retrain_result["scaler"], os.path.join(binary_save_dir, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(binary_save_dir, "label_encoder.pkl"))
    with open(os.path.join(binary_save_dir, "binary_final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(binary_metrics, f, ensure_ascii=False, indent=2)

    return binary_metrics


def save_multiclass_outputs(model_name: str, model_dir: str, real_data: pd.DataFrame, pred_labels_8: np.ndarray, class_probs_8: np.ndarray):
    multi8_save_dir = os.path.join(model_dir, "multiclass_8class")
    os.makedirs(multi8_save_dir, exist_ok=True)

    pred_df = real_data.copy()
    pred_df["predict_label"] = pred_labels_8
    for c in range(1, G + 1):
        pred_df[f"prob_class_{c}"] = class_probs_8[:, c - 1]

    pred_csv_path = os.path.join(multi8_save_dir, "Predict_Sensitivity_8d_with_probs.csv")
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"{model_name}: 8-class prediction file saved to: {pred_csv_path}")

    true_col = detect_multiclass_true_label_col(real_data, G)
    if true_col is None:
        print(f"{model_name}: Warning: 未识别到 8 分类真实标签列，只保存预测结果。")
        return None

    print(f"{model_name}: Detected 8-class true label column: {true_col}")
    y_true_8 = normalize_multiclass_true_labels(real_data[true_col].to_numpy(), G)
    y_pred_8 = pred_labels_8

    acc_8 = accuracy_score(y_true_8, y_pred_8)
    macro_f1_8 = f1_score(y_true_8, y_pred_8, labels=np.arange(1, G + 1), average="macro", zero_division=0)
    weighted_f1_8 = f1_score(y_true_8, y_pred_8, labels=np.arange(1, G + 1), average="weighted", zero_division=0)

    try:
        y_true_bin_8 = label_binarize(y_true_8, classes=np.arange(1, G + 1))
        macro_auc_8 = roc_auc_score(
            y_true_bin_8,
            class_probs_8,
            average="macro",
            multi_class="ovr",
        )
    except Exception as e:
        print(f"{model_name}: Warning: 无法计算 8 分类宏 AUC，原因: {e}")
        macro_auc_8 = np.nan

    print(f"\n===== {model_name}: Final Evaluation on 8-Class Data =====")
    print(f"8-class Accuracy: {acc_8:.4f}")
    print(f"8-class Macro AUC: {'nan' if np.isnan(macro_auc_8) else format(macro_auc_8, '.4f')}")
    print(f"8-class Macro F1: {macro_f1_8:.4f}")
    print(f"8-class Weighted F1: {weighted_f1_8:.4f}")

    cm_8 = confusion_matrix(y_true_8, y_pred_8, labels=np.arange(1, G + 1))
    disp_8 = ConfusionMatrixDisplay(confusion_matrix=cm_8, display_labels=[str(i) for i in range(1, G + 1)])
    fig, ax = plt.subplots(figsize=(8, 7))
    disp_8.plot(ax=ax, values_format="d")
    ax.set_title(f"{model_name} 8-Class Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(multi8_save_dir, "confusion_matrix_8class.png"), dpi=300)
    plt.close()

    multiclass_metrics = {
        "model_name": model_name,
        "true_label_column": true_col,
        "accuracy": float(acc_8),
        "macro_auc": None if np.isnan(macro_auc_8) else float(macro_auc_8),
        "macro_f1": float(macro_f1_8),
        "weighted_f1": float(weighted_f1_8),
    }
    with open(os.path.join(multi8_save_dir, "multiclass_8class_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(multiclass_metrics, f, ensure_ascii=False, indent=2)

    return multiclass_metrics


# ============================================================
# 9. 单个模型完整流程
# ============================================================
def run_single_model(
    model_name: str,
    X_trainval: np.ndarray,
    X_holdout: np.ndarray,
    y_trainval: np.ndarray,
    y_holdout: np.ndarray,
    label_encoder,
    input_size: int,
    seq_len: int,
    num_classes: int,
    positive_class_index: int,
    negative_class_index: int,
    real_data_8: Optional[pd.DataFrame],
):
    model_dir = os.path.join(SAVE_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    binary_save_dir = os.path.join(model_dir, "binary_holdout_cv")
    os.makedirs(binary_save_dir, exist_ok=True)

    param_grid = get_model_param_grids()[model_name]
    grid_list = list(ParameterGrid(param_grid))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    cv_results = []
    fold_best_records = {
        fold_idx: {"best_acc": -np.inf, "best_val_loss": np.inf, "best_params": None}
        for fold_idx in range(1, 6)
    }

    print(f"\n{'=' * 20} {model_name}: 5-Fold CV Grid Search {'=' * 20}")
    print(f"Total hyperparameter combinations: {len(grid_list)}")

    for config_idx, params in enumerate(grid_list, start=1):
        fold_accs = []
        fold_losses = []
        fold_epochs = []

        print(f"\n[{model_name} Grid {config_idx}/{len(grid_list)}] Params = {params}")

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_trainval, y_trainval), start=1):
            X_tr = X_trainval[tr_idx]
            y_tr = y_trainval[tr_idx]
            X_va = X_trainval[va_idx]
            y_va = y_trainval[va_idx]

            result = train_one_split(
                model_name=model_name,
                X_train=X_tr,
                y_train=y_tr,
                X_val=X_va,
                y_val=y_va,
                params=params,
                num_classes=num_classes,
                input_size=input_size,
                seq_len=seq_len,
                seed=SEED,
                max_epochs=MAX_EPOCHS,
                patience=PATIENCE,
                device=device,
            )

            fold_acc = result["best_val_acc"]
            fold_loss = result["best_val_loss"]
            fold_epoch = result["best_epoch"]

            fold_accs.append(fold_acc)
            fold_losses.append(fold_loss)
            fold_epochs.append(fold_epoch)

            print(
                f"  Fold {fold_idx}: val_acc={fold_acc:.4f}, "
                f"val_loss={fold_loss:.4f}, best_epoch={fold_epoch}"
            )

            current_best = fold_best_records[fold_idx]
            better = (
                (fold_acc > current_best["best_acc"]) or
                (np.isclose(fold_acc, current_best["best_acc"]) and fold_loss < current_best["best_val_loss"])
            )
            if better:
                fold_best_records[fold_idx] = {
                    "best_acc": float(fold_acc),
                    "best_val_loss": float(fold_loss),
                    "best_params": params.copy(),
                }

        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        mean_loss = float(np.mean(fold_losses))
        mean_epoch = float(np.mean(fold_epochs))

        print(
            f"  => Mean CV Acc = {mean_acc:.4f} ± {std_acc:.4f}, "
            f"Mean CV Loss = {mean_loss:.4f}, Mean Best Epoch = {mean_epoch:.2f}"
        )

        cv_results.append({
            "model_name": model_name,
            "params": params.copy(),
            "fold_accs": fold_accs,
            "fold_losses": fold_losses,
            "fold_epochs": fold_epochs,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "mean_loss": mean_loss,
            "mean_epoch": mean_epoch,
        })

    print(f"\n===== {model_name}: 每一折单独最优超参数（按该折 val accuracy 最大）=====")
    for fold_idx in range(1, 6):
        rec = fold_best_records[fold_idx]
        print(
            f"Fold {fold_idx}: best_acc={rec['best_acc']:.4f}, "
            f"best_val_loss={rec['best_val_loss']:.4f}, best_params={rec['best_params']}"
        )

    best_cv_result = sorted(cv_results, key=lambda x: (-x["mean_acc"], x["mean_loss"]))[0]
    best_params = best_cv_result["params"]

    print(f"\n===== {model_name}: 最终选出的最优超参数（按 5-fold mean accuracy）=====")
    print(best_params)
    print(
        f"Best mean CV accuracy = {best_cv_result['mean_acc']:.4f} ± {best_cv_result['std_acc']:.4f}, "
        f"mean CV loss = {best_cv_result['mean_loss']:.4f}"
    )

    cv_save_rows = []
    for r in cv_results:
        row = {"model_name": model_name, **r["params"]}
        row["mean_acc"] = r["mean_acc"]
        row["std_acc"] = r["std_acc"]
        row["mean_loss"] = r["mean_loss"]
        row["mean_epoch"] = r["mean_epoch"]
        for i, v in enumerate(r["fold_accs"], start=1):
            row[f"fold{i}_acc"] = v
        for i, v in enumerate(r["fold_losses"], start=1):
            row[f"fold{i}_loss"] = v
        cv_save_rows.append(row)

    cv_df = pd.DataFrame(cv_save_rows).sort_values(by=["mean_acc", "mean_loss"], ascending=[False, True])
    cv_df.to_csv(os.path.join(binary_save_dir, "cv_results.csv"), index=False)

    with open(os.path.join(binary_save_dir, "fold_best_params.json"), "w", encoding="utf-8") as f:
        json.dump(fold_best_records, f, ensure_ascii=False, indent=2)
    with open(os.path.join(binary_save_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    X_retrain_train, X_retrain_val, y_retrain_train, y_retrain_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.10,
        random_state=SEED,
        shuffle=True,
        stratify=y_trainval,
    )

    print(f"\n===== {model_name}: Retrain split inside 90% train+val =====")
    print(f"Retrain-Train size: {len(y_retrain_train)}")
    print(f"Retrain-Val size: {len(y_retrain_val)}")

    retrain_result = train_one_split(
        model_name=model_name,
        X_train=X_retrain_train,
        y_train=y_retrain_train,
        X_val=X_retrain_val,
        y_val=y_retrain_val,
        params=best_params,
        num_classes=num_classes,
        input_size=input_size,
        seq_len=seq_len,
        seed=SEED,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        device=device,
    )

    best_model = retrain_result["model"]
    best_scaler = retrain_result["scaler"]

    print(
        f"{model_name} Retrain done: best_epoch={retrain_result['best_epoch']}, "
        f"best_val_loss={retrain_result['best_val_loss']:.4f}, "
        f"best_val_acc={retrain_result['best_val_acc']:.4f}"
    )

    best_threshold, threshold_info = choose_threshold_by_youden_j(
        best_model,
        retrain_result["val_loader"],
        device,
        positive_class_index=positive_class_index,
    )

    print(f"\n===== {model_name}: Threshold selected on retrain-val =====")
    print(f"Best threshold = {best_threshold:.4f}")
    if threshold_info is not None:
        print(
            f"Youden J = {threshold_info['youden_j']:.4f}, "
            f"TPR = {threshold_info['tpr']:.4f}, FPR = {threshold_info['fpr']:.4f}"
        )

    test_result = evaluate_on_holdout(
        model=best_model,
        scaler=best_scaler,
        X_test=X_holdout,
        y_test=y_holdout,
        batch_size=best_params["batch_size"],
        device=device,
        threshold=best_threshold,
        positive_class_index=positive_class_index,
    )

    binary_metrics = save_binary_outputs(
        model_name=model_name,
        model_dir=model_dir,
        label_encoder=label_encoder,
        retrain_result=retrain_result,
        test_result=test_result,
        best_threshold=best_threshold,
        positive_class_index=positive_class_index,
        negative_class_index=negative_class_index,
    )

    multiclass_metrics = None
    if real_data_8 is not None:
        pred_labels_8, class_probs_8 = predict_multiclass_probs_from_binary(
            real_data=real_data_8,
            model=best_model,
            scaler=best_scaler,
            g=G,
            batch_size=best_params["batch_size"],
            device=device,
            right_choice_index=positive_class_index,
        )
        multiclass_metrics = save_multiclass_outputs(
            model_name=model_name,
            model_dir=model_dir,
            real_data=real_data_8,
            pred_labels_8=pred_labels_8,
            class_probs_8=class_probs_8,
        )

    summary = {
        "model_name": model_name,
        "best_params": best_params,
        "best_threshold": best_threshold,
        "best_mean_cv_acc": best_cv_result["mean_acc"],
        "best_mean_cv_loss": best_cv_result["mean_loss"],
        "binary_metrics": binary_metrics,
        "multiclass_metrics": multiclass_metrics,
    }
    with open(os.path.join(model_dir, "model_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


# ============================================================
# 10. 主流程：多模型对比
# ============================================================
def main():
    data, X_all, y_all, label_encoder = load_binary_data(BINARY_DATA_PATH)
    num_classes = len(label_encoder.classes_)
    seq_len = X_all.shape[1]
    input_size = X_all.shape[2]

    if num_classes != 2:
        raise ValueError(f"当前脚本要求二分类，但检测到 num_classes={num_classes}")

    positive_class_index = 1
    negative_class_index = 0

    print(f"Full binary data shape: X={X_all.shape}, y={y_all.shape}")
    print(f"Binary classes (original labels): {list(label_encoder.classes_)}")

    X_trainval, X_holdout, y_trainval, y_holdout = train_test_split(
        X_all,
        y_all,
        test_size=0.10,
        random_state=SEED,
        shuffle=True,
        stratify=y_all,
    )

    print("\n===== Shared Hold-out split for all models =====")
    print(f"Train+Val size: {len(y_trainval)}")
    print(f"Hold-out Test size: {len(y_holdout)}")

    real_data_8 = None
    if os.path.exists(MULTI8_DATA_PATH):
        real_data_8 = pd.read_csv(MULTI8_DATA_PATH)
        print(f"8-class data loaded: shape={real_data_8.shape}")
    else:
        print(f"Warning: 未找到 8 分类数据文件 {MULTI8_DATA_PATH}，将跳过所有模型的 8 分类评估。")

    all_summaries = []
    binary_rows = []
    multi_rows = []

    for model_name in MODEL_NAMES:
        summary = run_single_model(
            model_name=model_name,
            X_trainval=X_trainval,
            X_holdout=X_holdout,
            y_trainval=y_trainval,
            y_holdout=y_holdout,
            label_encoder=label_encoder,
            input_size=input_size,
            seq_len=seq_len,
            num_classes=num_classes,
            positive_class_index=positive_class_index,
            negative_class_index=negative_class_index,
            real_data_8=real_data_8,
        )
        all_summaries.append(summary)

        b = summary["binary_metrics"]
        binary_rows.append({
            "model_name": model_name,
            "best_mean_cv_acc": summary["best_mean_cv_acc"],
            "best_mean_cv_loss": summary["best_mean_cv_loss"],
            "holdout_accuracy": b["accuracy"],
            "holdout_auc": b["auc"],
            "negative_precision": b["negative_precision"],
            "negative_recall": b["negative_recall"],
            "positive_precision": b["positive_precision"],
            "positive_recall": b["positive_recall"],
            "positive_f1": b["positive_f1"],
            "threshold": b["threshold"],
        })

        m = summary["multiclass_metrics"]
        if m is not None:
            multi_rows.append({
                "model_name": model_name,
                "accuracy": m["accuracy"],
                "macro_auc": m["macro_auc"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                "true_label_column": m["true_label_column"],
            })

    binary_compare_df = pd.DataFrame(binary_rows).sort_values(
        by=["holdout_accuracy", "holdout_auc", "positive_f1"],
        ascending=[False, False, False],
    )
    binary_compare_df.to_csv(os.path.join(SAVE_DIR, "binary_model_comparison.csv"), index=False)

    if len(multi_rows) > 0:
        multi_compare_df = pd.DataFrame(multi_rows).sort_values(
            by=["accuracy", "macro_auc", "macro_f1", "weighted_f1"],
            ascending=[False, False, False, False],
        )
        multi_compare_df.to_csv(os.path.join(SAVE_DIR, "multiclass_model_comparison.csv"), index=False)

    with open(os.path.join(SAVE_DIR, "all_model_summaries.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print("\nAll done.")
    print(f"All model results saved in: {SAVE_DIR}")
    print(f"Binary comparison saved in: {os.path.join(SAVE_DIR, 'binary_model_comparison.csv')}")
    if len(multi_rows) > 0:
        print(f"Multiclass comparison saved in: {os.path.join(SAVE_DIR, 'multiclass_model_comparison.csv')}")


if __name__ == "__main__":
    main()
