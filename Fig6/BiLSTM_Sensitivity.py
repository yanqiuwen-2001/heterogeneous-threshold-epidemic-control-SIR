import os
import ast
import json
import random
from typing import Dict, Optional, List

# =========================
# 0. 在 import torch 之前设置环境变量
# =========================
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
    f1_score,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. 全局配置
# ============================================================
SEED = 5
G = 8

# ---------- 批量 n ----------
N_LIST = list(range(100, 701, 100))

# ---------- 输入文件模板 ----------
BINARY_DATA_TEMPLATE = "Traindata_2d_Sensitivity_{n}.csv"
MULTI8_DATA_TEMPLATE = "Traindata_8d_Sensitivity_{n}.csv"

# ---------- 8分类真实标签列 ----------
MULTI_TRUE_LABEL_COL = "alpha*"
MULTI_TRUE_LABEL_COL_CANDIDATES = [
    "label", "true_label", "target", "y", "Y",
    "alpha*", "alpha_star", "best_label", "final_label", "optimal_label"
]

# ---------- 保存目录 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "batch_n_results")
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- 训练参数 ----------
MAX_EPOCHS = 200
PATIENCE = 10

# ---------- 最优超参数 ----------
BEST_PARAMS = {
    "activation": "ReLU",
    "batch_size": 64,
    "dropout1": 0.2,
    "hidden_size": 64,
    "learning_rate": 0.01,
    "num_layers": 2,
}


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
# 4. 8分类辅助函数
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

    raise ValueError(f"8分类真实标签不在 1..{g} 或 0..{g-1} 范围内，实际 unique={sorted(list(uniq))}")


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
# 5. 模型定义
# ============================================================
class BiLSTMClassifier2d(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout1: float,
        activation_name: str,
        num_classes: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        self.dropout1 = nn.Dropout(dropout1)
        self.fc1 = nn.Linear(hidden_size * 2, 32)

        if activation_name == "ReLU":
            self.act = nn.ReLU()
        elif activation_name == "Sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        h_last = self.dropout1(h_last)
        h_last = self.fc1(h_last)
        h_last = self.act(h_last)
        h_last = self.dropout2(h_last)
        logits = self.fc2(h_last)
        return logits


# ============================================================
# 6. 训练 / 验证 / 推断函数
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
    X_train,
    y_train,
    X_val,
    y_val,
    params: Dict,
    num_classes: int,
    input_size: int,
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

    model = BiLSTMClassifier2d(
        input_size=input_size,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout1=params["dropout1"],
        activation_name=params["activation"],
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

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(max_epochs):
        model.train()
        running_train_loss = 0.0
        train_true = []
        train_pred = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            train_true.append(yb.detach().cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        train_loss = running_train_loss / len(train_loader)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = accuracy_score(train_true, train_pred)

        val_loss, val_acc = evaluate_loss_and_accuracy(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch+1:03d}/{max_epochs}] | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            best_epoch = epoch + 1
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
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
# 7. 8分类评估与保存
# ============================================================
def evaluate_multiclass_metrics(
    real_data: pd.DataFrame,
    pred_labels_8: np.ndarray,
    class_probs_8: np.ndarray,
    g: int = 8,
):
    true_col = detect_multiclass_true_label_col(real_data, g)
    if true_col is None:
        print("Warning: 未识别到 8 分类真实标签列，只返回预测结果，不计算 8 分类指标。")
        return None, None

    y_true_8 = normalize_multiclass_true_labels(real_data[true_col].to_numpy(), g)
    y_pred_8 = pred_labels_8

    acc_8 = accuracy_score(y_true_8, y_pred_8)
    macro_f1_8 = f1_score(
        y_true_8,
        y_pred_8,
        labels=np.arange(1, g + 1),
        average="macro",
        zero_division=0,
    )

    # 多分类 AUC
    try:
        y_true_8_zero_based = y_true_8 - 1
        auc_8 = roc_auc_score(
            y_true_8_zero_based,
            class_probs_8,
            multi_class="ovr",
            average="macro"
        )
    except Exception:
        auc_8 = np.nan

    report_8 = classification_report(
        y_true_8,
        y_pred_8,
        labels=np.arange(1, g + 1),
        target_names=[str(i) for i in range(1, g + 1)],
        digits=4,
        zero_division=0,
    )

    metrics = {
        "true_label_column": true_col,
        "accuracy": float(acc_8),
        "macro_f1": float(macro_f1_8),
        "auc": float(auc_8) if not np.isnan(auc_8) else np.nan,
        "classification_report": report_8,
    }

    return metrics, {
        "y_true": y_true_8,
        "y_pred": y_pred_8,
        "class_probs": class_probs_8,
    }


def save_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================================================
# 8. 单个 n 的完整流程
# ============================================================
def run_single_n(n: int):
    print("\n" + "=" * 80)
    print(f"开始运行 n = {n}")
    print("=" * 80)

    binary_path = BINARY_DATA_TEMPLATE.format(n=n)
    multi8_path = MULTI8_DATA_TEMPLATE.format(n=n)

    if not os.path.exists(binary_path):
        print(f"二分类文件不存在，跳过 n={n}: {binary_path}")
        return None

    n_save_dir = os.path.join(SAVE_DIR, f"n_{n}")
    os.makedirs(n_save_dir, exist_ok=True)

    # -------------------------------------------------------
    # A. 读取二分类数据
    # -------------------------------------------------------
    data, X_all, y_all, label_encoder = load_binary_data(binary_path)
    num_classes = len(label_encoder.classes_)
    input_size = X_all.shape[2]

    if num_classes != 2:
        raise ValueError(f"当前脚本要求二分类，但在 n={n} 中检测到 num_classes={num_classes}")

    positive_class_index = 1
    negative_class_index = 0

    print(f"Full data shape: X={X_all.shape}, y={y_all.shape}")
    print(f"Classes (original labels): {list(label_encoder.classes_)}")
    print(f"Using BEST_PARAMS = {BEST_PARAMS}")

    # -------------------------------------------------------
    # B. 先划分 80% Train + 20% Temp
    # -------------------------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all,
        y_all,
        test_size=0.20,
        random_state=SEED,
        shuffle=True,
        stratify=y_all,
    )

    print("\n===== Step B: Train / Temp split =====")
    print(f"Train size: {len(y_train)}")
    print(f"Temp size: {len(y_temp)}")

    # -------------------------------------------------------
    # C. 将剩余 20% 对半分成 10% Validation + 10% Test
    # -------------------------------------------------------
    X_val, X_holdout, y_val, y_holdout = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=SEED,
        shuffle=True,
        stratify=y_temp,
    )

    print("\n===== Step C: Validation / Test split inside 20% =====")
    print(f"Train size: {len(y_train)}")
    print(f"Val size: {len(y_val)}")
    print(f"Test size: {len(y_holdout)}")

    # -------------------------------------------------------
    # D. 训练二分类模型
    # -------------------------------------------------------
    print("\n===== Step D: Train best model =====")
    result = train_one_split(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        params=BEST_PARAMS,
        num_classes=num_classes,
        input_size=input_size,
        seed=SEED,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        device=device,
    )

    best_model = result["model"]
    best_scaler = result["scaler"]

    print(
        f"\nTraining finished: "
        f"best_epoch={result['best_epoch']}, "
        f"best_val_loss={result['best_val_loss']:.4f}, "
        f"best_val_acc={result['best_val_acc']:.4f}"
    )

    # -------------------------------------------------------
    # E. 在验证集上选阈值
    # -------------------------------------------------------
    best_threshold, threshold_info = choose_threshold_by_youden_j(
        best_model,
        result["val_loader"],
        device,
        positive_class_index=positive_class_index,
    )

    print("\n===== Step E: Threshold selected on validation set =====")
    print(f"Best threshold = {best_threshold:.4f}")
    if threshold_info is not None:
        print(
            f"Youden J = {threshold_info['youden_j']:.4f}, "
            f"TPR = {threshold_info['tpr']:.4f}, "
            f"FPR = {threshold_info['fpr']:.4f}"
        )

    # -------------------------------------------------------
    # F. 二分类 hold-out test 评估
    # -------------------------------------------------------
    test_result = evaluate_on_holdout(
        model=best_model,
        scaler=best_scaler,
        X_test=X_holdout,
        y_test=y_holdout,
        batch_size=BEST_PARAMS["batch_size"],
        device=device,
        threshold=best_threshold,
        positive_class_index=positive_class_index,
    )

    target_names = [str(x) for x in label_encoder.classes_]
    precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(
        test_result["test_true"],
        test_result["test_pred"],
        labels=[negative_class_index, positive_class_index],
        zero_division=0,
    )

    binary_report = classification_report(
        test_result["test_true"],
        test_result["test_pred"],
        labels=np.arange(num_classes),
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    binary_metrics = {
        "n": n,
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_holdout)),
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
        "best_epoch": int(result["best_epoch"]),
        "best_val_loss": float(result["best_val_loss"]),
        "best_val_acc": float(result["best_val_acc"]),
    }

    print("\n===== Final Evaluation on Hold-out Test =====")
    print(f"AUC: {binary_metrics['auc'] if binary_metrics['auc'] is None else format(binary_metrics['auc'], '.4f')}")
    print("\nClassification Report (Binary Hold-out Test):")
    print(binary_report)

    # 保存二分类结果
    save_text(os.path.join(n_save_dir, "binary_classification_report.txt"), binary_report)
    save_json(os.path.join(n_save_dir, "binary_metrics.json"), binary_metrics)

    pred_df = pd.DataFrame({
        "y_true": test_result["test_true"],
        "y_pred": test_result["test_pred"],
        "prob_positive": test_result["test_prob"],
    })
    pred_df.to_csv(os.path.join(n_save_dir, "binary_holdout_test_predictions.csv"), index=False)

    torch.save(result["best_state"], os.path.join(n_save_dir, "best_model_state_dict.pth"))
    joblib.dump(best_scaler, os.path.join(n_save_dir, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(n_save_dir, "label_encoder.pkl"))

    # -------------------------------------------------------
    # G_Results. 8分类预测与评估
    # -------------------------------------------------------
    multiclass_metrics = None
    if os.path.exists(multi8_path):
        real_data_8 = pd.read_csv(multi8_path)
        print(f"\n8-class data loaded: shape={real_data_8.shape}")

        pred_labels_8, class_probs_8 = predict_multiclass_probs_from_binary(
            real_data=real_data_8,
            model=best_model,
            scaler=best_scaler,
            g=G,
            batch_size=BEST_PARAMS["batch_size"],
            device=device,
            right_choice_index=positive_class_index,
        )

        pred_out_df = real_data_8.copy()
        pred_out_df["predict_label"] = pred_labels_8
        for c in range(1, G + 1):
            pred_out_df[f"prob_class_{c}"] = class_probs_8[:, c - 1]
        pred_out_df.to_csv(os.path.join(n_save_dir, "predict_8class_with_probs.csv"), index=False)

        multiclass_metrics, multiclass_raw = evaluate_multiclass_metrics(
            real_data=real_data_8,
            pred_labels_8=pred_labels_8,
            class_probs_8=class_probs_8,
            g=G,
        )

        if multiclass_metrics is not None:
            print("\n===== Final Evaluation on 8-Class Data =====")
            print(f"8-class Accuracy: {multiclass_metrics['accuracy']:.4f}")
            print(f"8-class Macro F1: {multiclass_metrics['macro_f1']:.4f}")
            auc8 = multiclass_metrics["auc"]
            print(f"8-class AUC: {auc8 if np.isnan(auc8) else format(auc8, '.4f')}")
            print("\nClassification Report (8-Class):")
            print(multiclass_metrics["classification_report"])

            save_text(
                os.path.join(n_save_dir, "multiclass_8class_classification_report.txt"),
                multiclass_metrics["classification_report"]
            )
            save_json(
                os.path.join(n_save_dir, "multiclass_8class_metrics.json"),
                {
                    "true_label_column": multiclass_metrics["true_label_column"],
                    "accuracy": multiclass_metrics["accuracy"],
                    "macro_f1": multiclass_metrics["macro_f1"],
                    "auc": None if np.isnan(multiclass_metrics["auc"]) else multiclass_metrics["auc"],
                }
            )
        else:
            print(f"Warning: n={n} 的 8分类真实标签未识别，跳过 8分类指标计算。")
    else:
        print(f"Warning: 未找到 8 分类数据文件，跳过 n={n}: {multi8_path}")

    # -------------------------------------------------------
    # H. 汇总单个 n 结果
    # -------------------------------------------------------
    summary_one_n = {
        "n": n,
        "binary_metrics": binary_metrics,
        "multiclass_8class_metrics": None if multiclass_metrics is None else {
            "true_label_column": multiclass_metrics["true_label_column"],
            "accuracy": multiclass_metrics["accuracy"],
            "macro_f1": multiclass_metrics["macro_f1"],
            "auc": None if np.isnan(multiclass_metrics["auc"]) else multiclass_metrics["auc"],
        }
    }

    save_json(os.path.join(n_save_dir, "summary_one_n.json"), summary_one_n)

    return summary_one_n


# ============================================================
# 9. 主流程：批量运行 n=100,200,...,700
# ============================================================
def main():
    all_results: List[Dict] = []

    for n in N_LIST:
        try:
            result = run_single_n(n)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"\n运行 n={n} 时出错: {e}")
            all_results.append({
                "n": n,
                "error": str(e)
            })

    # -------------------------------------------------------
    # 生成总汇总表
    # -------------------------------------------------------
    summary_rows = []
    for item in all_results:
        n = item.get("n")

        if "error" in item:
            summary_rows.append({
                "n": n,
                "status": "failed",
                "error": item["error"],
                "binary_auc": np.nan,
                "binary_accuracy": np.nan,
                "binary_positive_f1": np.nan,
                "multi8_accuracy": np.nan,
                "multi8_macro_f1": np.nan,
                "multi8_auc": np.nan,
            })
            continue

        binary_metrics = item.get("binary_metrics", {})
        multi_metrics = item.get("multiclass_8class_metrics", None)

        summary_rows.append({
            "n": n,
            "status": "success",
            "error": "",
            "binary_auc": binary_metrics.get("auc", np.nan),
            "binary_accuracy": binary_metrics.get("accuracy", np.nan),
            "binary_positive_f1": binary_metrics.get("positive_f1", np.nan),
            "multi8_accuracy": np.nan if multi_metrics is None else multi_metrics.get("accuracy", np.nan),
            "multi8_macro_f1": np.nan if multi_metrics is None else multi_metrics.get("macro_f1", np.nan),
            "multi8_auc": np.nan if multi_metrics is None else multi_metrics.get("auc", np.nan),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(SAVE_DIR, "Sensitivity_results.csv")
    summary_json_path = os.path.join(SAVE_DIR, "Sensitivity_results.json")

    summary_df.to_csv(summary_csv_path, index=False)
    save_json(summary_json_path, all_results)

    print("\n" + "=" * 80)
    print("所有 n 已运行完成")
    print(f"总汇总 CSV: {summary_csv_path}")
    print(f"总汇总 JSON: {summary_json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()