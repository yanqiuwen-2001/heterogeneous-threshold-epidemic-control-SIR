import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


FIG_WIDTH = 174 / 15.4          # 174 mm, full text width
FIG_HEIGHT = 50 / 15.4          # 自定义

FONT_FAMILY = "Arial"
TITLE_SIZE = 14.5
TEXT_SIZE = 14.5
AXIS_LABEL_SIZE = 12
TICK_LABEL_SIZE = 9
LEGEND_SIZE = 10

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT_FAMILY],
    "font.size": TEXT_SIZE,

    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_FAMILY,
    "mathtext.it": f"{FONT_FAMILY}:italic",
    "mathtext.bf": f"{FONT_FAMILY}:bold",
    "mathtext.default": "it",

    "axes.linewidth": 0.6,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.titleweight": "normal",
    "axes.xmargin": 0.1,

    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,

    "legend.loc": "upper right",
    "legend.fontsize": LEGEND_SIZE,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.fancybox": False,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FONT_SIZE = TEXT_SIZE
LABEL_SIZE = AXIS_LABEL_SIZE
TICK_SIZE = TICK_LABEL_SIZE
AXIS_LINEWIDTH = 0.6
TICK_WIDTH = 0.6
YLABEL_PAD = 8


def set_subfigure_title(ax, label, pad=6):
    ax.set_title(
        label,
        fontsize=TITLE_SIZE,
        fontweight="normal",
        loc="center",
        pad=pad
    )


# =========================
# 读取数据
# =========================
df = pd.read_csv("Models_results.csv")

models_original = ["MLP", "CNN", "GRU", "Transformer", "LSTM", "BiLSTM"]

# 按 BiLSTM, GRU, Transformer, LSTM, CNN, MLP 排序
order = [5, 2, 3, 4, 1, 0]
models = [models_original[i] for i in order]

metrics = {
    "Accuracy": [df["2d_accuracy"].tolist()[i] for i in order],
    "F1": [df["2d_f1_pos"].tolist()[i] for i in order],
    "AUC": [df["2d_auc"].tolist()[i] for i in order],
}

# 红蓝渐变色系
cmap = plt.cm.RdBu
colors = cmap(np.linspace(0.95, 0.55, len(models)))

# =========================
# 画图：三个指标，每个指标6个模型
# =========================
fig, axs = plt.subplots(
    1,
    3,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    sharey=True
)

for ax, (metric_name, values) in zip(axs, metrics.items()):
    x = np.arange(len(models))

    bars = ax.bar(
        x,
        values,
        width=0.68,
        color=colors,
        edgecolor="white",
        linewidth=0.6
    )

    # 子图顶部小标题去掉，改为纵轴标题
    ax.set_ylabel(
        metric_name,
        fontsize=LABEL_SIZE,
        rotation=90,
        labelpad=YLABEL_PAD
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        models,
        fontsize=TICK_SIZE,
        rotation=35,
        ha="right"
    )

    ax.set_ylim(0.85, 1.0)

    ax.yaxis.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        color="#d9d9d9"
    )
    ax.set_axisbelow(True)

    ax.tick_params(axis="x", length=0, labelsize=TICK_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE, width=TICK_WIDTH)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(AXIS_LINEWIDTH)
    ax.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.002,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=TICK_SIZE,
            color="#333333",
            rotation=0
        )

yticks = np.linspace(0.85, 1.0, 6)
axs[0].set_yticks(yticks)
axs[0].set_yticklabels(
    [f"{y:.2f}" for y in yticks],
    fontsize=TICK_SIZE
)

# 三个子图合起来正上方写 (a)
fig.text(
    0.5,
    0.96,
    "(a)",
    ha="center",
    va="top",
    fontsize=TITLE_SIZE,
    fontweight="normal"
)

fig.subplots_adjust(
    left=0.07,
    right=1,
    bottom=0.20,
    top=0.85,
    wspace=0.1
)

plt.savefig(
    "Figure_Models_1x3.eps",
    format="eps",
    bbox_inches="tight",
    pad_inches=0.03
)

plt.show()
