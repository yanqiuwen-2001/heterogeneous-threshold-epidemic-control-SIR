import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

FIG_WIDTH = 174 / 15.4
FIG_HEIGHT = 75 / 15.4

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
    "ytick.minor.width": 0.6,
    "ytick.major.width": 0.6,

    "legend.loc": "upper right",
    "legend.fontsize": LEGEND_SIZE,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.fancybox": False,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

AXIS_LINEWIDTH = 0.6
TICK_WIDTH = 0.6
YLABEL_PAD = 6


def set_subfigure_title(ax, label, pad=6):
    ax.set_title(
        label,
        fontsize=TITLE_SIZE,
        fontweight="normal",
        loc="center",
        pad=pad
    )


# ============================================================
# 左图数据：classification report
# ============================================================

columns = ["Class", "Precision", "Recall", "F1-score", "Support"]

rows = [
    ["1", "0.9499", "0.9651", "0.9575", "5013"],
    ["2", "0.9544", "0.9601", "0.9572", "4943"],
    ["3", "0.9521", "0.9610", "0.9565", "5071"],
    ["4", "0.9474", "0.9595", "0.9534", "4938"],
    ["5", "0.9571", "0.9552", "0.9562", "4955"],
    ["6", "0.9603", "0.9492", "0.9547", "4965"],
    ["7", "0.9598", "0.9474", "0.9536", "5042"],
    ["8", "0.9633", "0.9466", "0.9549", "5073"],
    ["Accuracy", "", "0.9555", "", "40000"],
    ["Macro Avg", "0.9555", "0.9555", "0.9555", "40000"],
    ["Weighted Avg", "0.9556", "0.9555", "0.9555", "40000"],
]

# ============================================================
# 右图数据：8-class confusion matrix
# ============================================================

cm = np.array([
    [43, 28, 30, 52, 42, 42, 34, 4802],
    [37, 39, 47, 46, 35, 31, 4777, 30],
    [30, 43, 50, 31, 41, 4713, 27, 30],
    [37, 31, 43, 40, 4733, 24, 24, 23],
    [32, 31, 29, 4738, 21, 27, 37, 23],
    [37, 30, 4873, 28, 26, 26, 21, 30],
    [39, 4746, 24, 29, 23, 27, 32, 23],
    [4838, 25, 22, 37, 24, 18, 25, 24],
])

x_labels = [str(i) for i in range(1, 9)]
y_labels = [str(i) for i in range(8, 0, -1)]

# ============================================================
# 合并作图
# ============================================================

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

gs = fig.add_gridspec(
    nrows=1,
    ncols=2,
    width_ratios=[1.3, 1.0],
    wspace=0.22
)

ax_table = fig.add_subplot(gs[0, 0])
ax_cm = fig.add_subplot(gs[0, 1])

# ============================================================
# (e) 左图：表格
# ============================================================

ax_table.set_xlim(0, 1)
ax_table.set_ylim(0, 1)
ax_table.axis("off")
set_subfigure_title(ax_table, "(e)", pad=8)

left = 0.01
right = 0.99
top = 0.99
bottom = 0.01

n_rows = 1 + len(rows)
row_h = (top - bottom) / n_rows

col_x = np.array([0.13, 0.34, 0.54, 0.74, 0.91])

y_top = top
y_header_bottom = top - row_h
y_after_classes = top - row_h * 9
y_bottom = bottom

ax_table.hlines(y_top, left, right, color="black", linewidth=1.0)
ax_table.hlines(y_header_bottom, left, right, color="black", linewidth=0.6)
ax_table.hlines(y_after_classes, left, right, color="black", linewidth=0.6)
ax_table.hlines(y_bottom, left, right, color="black", linewidth=1.0)

y = top - row_h / 2
for x, text in zip(col_x, columns):
    ax_table.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=TEXT_SIZE,
        fontfamily="Times New Roman"
    )

for r, row in enumerate(rows):
    y = top - row_h * (r + 1) - row_h / 2

    for c, text in enumerate(row):
        ax_table.text(
            col_x[c],
            y,
            text,
            ha="center",
            va="center",
            fontsize=TEXT_SIZE,
            fontfamily="Times New Roman"
        )

# ============================================================
# (f) 右图：混淆矩阵
# ============================================================

set_subfigure_title(ax_cm, "(f)", pad=8)

im = ax_cm.imshow(
    cm,
    cmap=plt.cm.Blues,
    vmin=0,
    vmax=5000,
    interpolation="nearest"
)

ax_cm.set_xlabel(
    "Predicted label",
    fontsize=AXIS_LABEL_SIZE
)

ax_cm.set_ylabel(
    "True label",
    fontsize=AXIS_LABEL_SIZE,
    rotation=90,
    labelpad=YLABEL_PAD
)

ax_cm.set_xticks(np.arange(len(x_labels)))
ax_cm.set_xticklabels(x_labels, fontsize=TICK_LABEL_SIZE)

ax_cm.set_yticks(np.arange(len(y_labels)))
ax_cm.set_yticklabels(y_labels, fontsize=TICK_LABEL_SIZE)

ax_cm.tick_params(
    axis="both",
    labelsize=TICK_LABEL_SIZE,
    width=TICK_WIDTH,
    length=0
)

threshold = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        value = cm[i, j]
        text_color = "white" if value > threshold else "#174a7e"
        ax_cm.text(
            j,
            i,
            f"{value:d}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=TICK_LABEL_SIZE
        )

for spine in ax_cm.spines.values():
    spine.set_visible(False)

cbar = fig.colorbar(
    im,
    ax=ax_cm,
    fraction=0.046,
    pad=0.04
)

cbar.set_ticks([1000, 2000, 3000, 4000])
cbar.ax.tick_params(
    labelsize=TICK_LABEL_SIZE,
    width=TICK_WIDTH,
    length=3
)

cbar.outline.set_visible(False)
for spine in cbar.ax.spines.values():
    spine.set_visible(False)

# ============================================================
# 保存
# ============================================================

fig.subplots_adjust(
    left=0.02,
    right=0.98,
    bottom=0.14,
    top=0.88,
    wspace=0.22
)

# 右图内部整体向左移动
shift = 0.04

cm_pos = ax_cm.get_position()
ax_cm.set_position([
    cm_pos.x0 - shift,
    cm_pos.y0,
    cm_pos.width,
    cm_pos.height
])

cbar_pos = cbar.ax.get_position()
cbar.ax.set_position([
    cbar_pos.x0 - shift,
    cbar_pos.y0,
    cbar_pos.width,
    cbar_pos.height
])

plt.savefig(
    "Figure_ef.eps",
    format="eps",
    bbox_inches="tight",
    pad_inches=0.03
)

plt.show()