import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

G = 8
TRUE_LABEL_COL = 'label'

# 定义模型结构
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
        self.fc1 = nn.Linear(self.hidden_size * 2, 32)
        self.act = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)

        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)

        h_last = self.dropout(h_last)
        h_last = self.fc1(h_last)
        h_last = self.act(h_last)
        h_last = self.dropout2(h_last)
        logits = self.fc2(h_last)
        return logits


# 读取真实数据
real_data = pd.read_csv('Realdata.csv')

pred_labels = []
score_list = []

for i in range(len(real_data)):
    D1 = 1
    D2 = 2

    final_score = np.zeros(G)

    for k in range(3, G + 2):
        # 预处理函数
        def preprocess_real_data(df, scaler):
            infection_data = []
            for i in range(14):
                day_col_1 = f'gamI{D1}-Day{i + 1}'
                day_col_2 = f'gamI{D2}-Day{i + 1}'

                parsed_1 = df[day_col_1].values
                parsed_2 = df[day_col_2].values

                combined_data = np.stack([parsed_1, parsed_2], axis=-1)
                infection_data.append(combined_data)

            X_infected = np.stack(infection_data, axis=1)

            Ts = np.repeat(df[f'Ts'].values[:, np.newaxis], 14, axis=1)[:, :, np.newaxis]
            N1 = np.repeat(df[f'N{D1}'].values[:, np.newaxis], 14, axis=1)[:, :, np.newaxis]
            N2 = np.repeat(df[f'N{D2}'].values[:, np.newaxis], 14, axis=1)[:, :, np.newaxis]

            X_all = np.concatenate([X_infected, Ts, N1, N2], axis=2)

            X_flat = X_all.reshape(-1, 5)
            X_scaled = scaler.transform(X_flat).reshape(X_all.shape)

            return torch.tensor(X_scaled, dtype=torch.float32)

        # 加载模型参数
        scaler = joblib.load('../Real_Delta/scaler.pkl')
        label_encoder = joblib.load('../Real_Delta/label_encoder.pkl')

        # 初始化模型列表
        model2d = BiLSTMClassifier2d()
        model2d.load_state_dict(torch.load('../Real_Delta/best_model2d.pth'))
        model2d.eval()

        # 选择一个样本
        sample_data = preprocess_real_data(real_data.iloc[i:i + 1], scaler)

        with torch.no_grad():
            logits = model2d(sample_data)
            probs = nn.functional.softmax(logits, dim=1).numpy()
            pred = np.argmax(probs, axis=1)
            label = label_encoder.inverse_transform(pred)

        if label.item() == 1:
            final_score[D1 - 1] = max(final_score[D1 - 1], probs[0, pred[0]])
            D1 = D1
            D2 = k
        elif label.item() == 2:
            final_score[k - 1] = max(final_score[k - 1], probs[0, pred[0]])
            D1 = k - 1
            D2 = k

    if label.item() == 1:
        final_label = D1
    elif label.item() == 2:
        final_label = G

    pred_labels.append(final_label)

    if final_score.sum() == 0:
        final_score[final_label - 1] = 1.0
    else:
        final_score[final_label - 1] = max(final_score[final_label - 1], final_score.max())

    score_list.append(final_score)


real_data[f'predict_label'] = pred_labels

# 将更新后的 DataFrame 保存到文件
real_data.to_csv('Predict_8d.csv', index=False)


# =========================
# 补充 1：输出分类指标表格
# =========================
if TRUE_LABEL_COL in real_data.columns:
    y_true = real_data[TRUE_LABEL_COL].to_numpy()
    y_pred = np.array(pred_labels)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(1, G + 1)),
        output_dict=True,
        zero_division=0
    )

    rows = []
    for cls in range(1, G + 1):
        rows.append([
            str(cls),
            f"{report_dict[str(cls)]['precision']:.4f}",
            f"{report_dict[str(cls)]['recall']:.4f}",
            f"{report_dict[str(cls)]['f1-score']:.4f}",
            f"{int(report_dict[str(cls)]['support'])}"
        ])

    accuracy = accuracy_score(y_true, y_pred)
    total_support = len(y_true)

    rows.append([
        "Accuracy",
        "",
        f"{accuracy:.4f}",
        "",
        f"{total_support}"
    ])
    rows.append([
        "Macro Avg",
        f"{report_dict['macro avg']['precision']:.4f}",
        f"{report_dict['macro avg']['recall']:.4f}",
        f"{report_dict['macro avg']['f1-score']:.4f}",
        f"{int(report_dict['macro avg']['support'])}"
    ])
    rows.append([
        "Weighted Avg",
        f"{report_dict['weighted avg']['precision']:.4f}",
        f"{report_dict['weighted avg']['recall']:.4f}",
        f"{report_dict['weighted avg']['f1-score']:.4f}",
        f"{int(report_dict['weighted avg']['support'])}"
    ])

    columns = ["Class", "Precision", "Recall", "F1-score", "Support"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis('off')

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc='center',
        colLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1.0, 1.7)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_linewidth(0)
        cell.set_facecolor('white')

    for col in range(len(columns)):
        table[(0, col)].set_text_props(weight='normal')

    ax.plot([0.04, 0.96], [0.90, 0.90], color='black', lw=1.6, transform=ax.transAxes)
    ax.plot([0.04, 0.96], [0.78, 0.78], color='black', lw=1.0, transform=ax.transAxes)
    ax.plot([0.04, 0.96], [0.18, 0.18], color='black', lw=1.0, transform=ax.transAxes)
    ax.plot([0.04, 0.96], [0.08, 0.08], color='black', lw=1.6, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig("Predict_8d_report.eps", format="eps", bbox_inches="tight")
    plt.show()


    # =========================
    # 补充 2：输出 ROC 曲线图
    # =========================
    classes = np.arange(1, G + 1)
    y_true_bin = label_binarize(y_true, classes=classes)
    y_score = np.array(score_list)

    row_sum = y_score.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    y_score = y_score / row_sum

    plt.figure(figsize=(8, 6))

    colors = ['#d73221', '#fcb777', '#fee395', '#acd2e5',
              '#6491c1', '#4573b4', '#7b3294', '#008837']

    for idx, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_score[:, idx])
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            linewidth=2.0,
            color=colors[idx],
            label=f'Class {cls} AUC = {roc_auc:.3f}'
        )

    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)

    plt.plot(
        fpr_micro,
        tpr_micro,
        color='black',
        linestyle='-',
        linewidth=2.6,
        label=f'Micro-average AUC = {auc_micro:.3f}'
    )

    plt.plot([0, 1], [0, 1], linestyle='--', color='#999999', linewidth=1.2)

    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('ROC Curve', fontsize=15, fontweight='semibold')
    plt.tick_params(axis='both', labelsize=11)
    plt.grid(True, linestyle='--', linewidth=0.6, color='#d9d9d9', alpha=0.9)
    plt.legend(loc='lower right', fontsize=9, frameon=True)

    plt.tight_layout()
    plt.savefig("Predict_8d_ROC.eps", format="eps", bbox_inches="tight")
    plt.show()

else:
    print(f"未找到真实标签列：{TRUE_LABEL_COL}")
    print("已完成 Predict_8d.csv 输出；如需分类指标表和 ROC 曲线，请确认 Realdata.csv 中真实标签列名。")