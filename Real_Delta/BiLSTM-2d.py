import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, precision_recall_fscore_support
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

# 1. 加载数据
data = pd.read_csv('Traindata_2d_Sensitivity_300.csv')

# 2. 提取序列特征
infection_data = []
for i in range(14):
    day_col = f'gamI-Day{i+1}'
    parsed = data[day_col].apply(ast.literal_eval)
    infection_data.append(np.stack(parsed.to_numpy()))
X_infected = np.stack(infection_data, axis=1)

# 3. 拼接附加特征
Ts = np.repeat(data['Ts'].values[:, np.newaxis], 14, axis=1)[:, :, np.newaxis]
N1 = np.repeat(data['N1'].values[:, np.newaxis], 14, axis=1)[:, :, np.newaxis]
N2 = np.repeat(data['N2'].values[:, np.newaxis], 14, axis=1)[:, :, np.newaxis]
X_all = np.concatenate([X_infected, Ts, N1, N2], axis=2)

# 5. 标签编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['alpha*'].values)

# 划分训练集、验证集、测试集
X_train, X_temp, y_train, y_temp = train_test_split(X_all, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 归一化
scaler = MinMaxScaler()

def fit_transform_2d(X, scaler):
    X_flat = X.reshape(-1, X.shape[2])
    X_scaled = scaler.fit_transform(X_flat)
    return X_scaled.reshape(X.shape)

def transform_2d(X, scaler):
    X_flat = X.reshape(-1, X.shape[2])
    X_scaled = scaler.transform(X_flat)
    return X_scaled.reshape(X.shape)

X_train = fit_transform_2d(X_train, scaler)
X_val   = transform_2d(X_val, scaler)
X_test  = transform_2d(X_test, scaler)

# 转为 tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32)

# 类权重
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# ============================================================
# 模型定义
# ============================================================
class BiLSTMClassifier2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 64
        self.num_layers = 2

        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.hidden_size * 2, 32)  # ✅ 2H
        self.act = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        # out: (B, T, 2H)
        # h_n: (num_layers * num_directions, B, H)
        out, (h_n, c_n) = self.lstm(x)


        # h_n[-2] = last layer forward, h_n[-1] = last layer backward
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)   # (B, 2H)

        h_last = self.dropout(h_last)
        h_last = self.fc1(h_last)
        h_last = self.act(h_last)
        h_last = self.dropout2(h_last)
        logits = self.fc2(h_last)
        return logits

model = BiLSTMClassifier2d()
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)


# ---- 训练：保存 best model ----
best_loss = float('inf')
patience = 10
wait = 0

for epoch in range(200):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            val_loss += criterion(pred, yb).item()
    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        best_model2d = {k: v.detach().clone() for k, v in model.state_dict().items()}  # ✅ 深拷贝
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

model.load_state_dict(best_model2d)

# 保存模型与scaler
joblib.dump(scaler, '../Real_Delta/scaler.pkl')
joblib.dump(label_encoder, '../Real_Delta/label_encoder.pkl')
torch.save(best_model2d, '../Real_Delta/best_model2d.pth')


# =========================
# 用验证集调阈值，用测试集评估
# =========================
def get_positive_prob_and_true(model, loader):
    model.eval()
    probs_all = []
    y_all = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            probs = nn.functional.softmax(logits, dim=1)[:, 1]  # 正类概率
            probs_all.append(probs.cpu().numpy())
            y_all.append(yb.cpu().numpy())
    return np.concatenate(probs_all), np.concatenate(y_all)

# 1) 用 Val 选阈值：使 J = TPR - FPR 最大
val_prob, val_true = get_positive_prob_and_true(model, val_loader)

val_fpr, val_tpr, val_thresholds = roc_curve(val_true, val_prob)
j_scores = val_tpr - val_fpr

finite_mask = np.isfinite(val_thresholds)
val_fpr = val_fpr[finite_mask]
val_tpr = val_tpr[finite_mask]
val_thresholds = val_thresholds[finite_mask]
j_scores = j_scores[finite_mask]

best_idx = np.argmax(j_scores)
best_threshold = float(val_thresholds[best_idx])
best_j = float(j_scores[best_idx])

print(f"\n✅ Val上选出的最佳阈值: {best_threshold:.4f}, Val-Youden J = {best_j:.4f}")
print(f"   对应 TPR = {val_tpr[best_idx]:.4f}, FPR = {val_fpr[best_idx]:.4f}")

# 2) 用 Test 做最终评估（阈值固定为 Val 选出的）
test_prob, test_true = get_positive_prob_and_true(model, test_loader)
test_pred = (test_prob >= best_threshold).astype(int)

print("\nClassification Report (Test, threshold from Val-Youden J):")
print(classification_report(test_true, test_pred, digits=4))

# 混淆矩阵
cm = confusion_matrix(test_true, test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Test, Threshold from Val)")
plt.show()

# ROC 曲线
fpr, tpr, _ = roc_curve(test_true, test_prob)
auc = roc_auc_score(test_true, test_prob)
plt.plot(fpr, tpr, label=f'ROC (AUC={auc:.4f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC Curve (Test)")
plt.show()
