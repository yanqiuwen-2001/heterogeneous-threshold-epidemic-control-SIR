# import csv
# import os
# from pathlib import Path
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter, FuncFormatter
#
#
# # ============================================================
# # 0. 全局图形格式
# # ============================================================
# plt.rcParams["font.family"] = "DejaVu Sans"
# plt.rcParams["axes.linewidth"] = 0.9
# plt.rcParams["pdf.fonttype"] = 42
# plt.rcParams["ps.fonttype"] = 42
#
# FONT_SIZE = 9
# LABEL_SIZE = 10
# TICK_SIZE = 9
# AXIS_LINEWIDTH = 0.9
# TICK_WIDTH = 0.9
# YLABEL_PAD = 6
#
#
# # ============================================================
# # 1. 路径设置
# # ============================================================
# SEED_CSV_PATH = Path("all_seed_results_summary.csv")
#
# SENS_BASE_DIR = "../Sensitivity_Ts_data"
# SENS_FILE_NAME = "Sensitivity_results.csv"
# SENS_FILE_PATH = os.path.join(SENS_BASE_DIR, SENS_FILE_NAME)
#
# OUT_PNG = Path("combined_boxplot_sensitivity_v7.png")
# OUT_EPS = Path("combined_boxplot_sensitivity_v7.eps")
#
#
# # ============================================================
# # 2. 箱体图指标设置
# # ============================================================
# binary_metrics = [
#     "binary_accuracy",
#     "binary_auc",
#     "binary_f1",
# ]
#
# binary_labels = [
#     "2D Accuracy",
#     "2D AUC",
#     "2D F1",
# ]
#
# multiclass_metrics = [
#     "multiclass_accuracy",
#     "multiclass_macro_auc",
#     "multiclass_macro_f1",
# ]
#
# multiclass_labels = [
#     "8D Accuracy",
#     "8D Macro-AUC",
#     "8D Macro-F1",
# ]
#
#
# # ============================================================
# # 3. 折线图指标设置
# # ============================================================
# xs = list(range(100, 701, 100))
#
# accuracy_metrics = [
#     "2d_accuracy",
#     "8d_accuracy",
# ]
#
# auc_metrics = [
#     "2d_auc",
#     "8d_macro_auc",
# ]
#
# line_titles = {
#     "2d_accuracy": "2D Accuracy",
#     "2d_auc": "2D AUC",
#     "8d_accuracy": "8D Accuracy",
#     "8d_macro_auc": "8D Macro-AUC",
# }
#
# line_colors = {
#     "2d_accuracy": "#b51519",
#     "2d_auc": "#e9a2a4",
#     "8d_accuracy": "#2354a1",
#     "8d_macro_auc": "#73a4cb",
# }
#
#
# # ============================================================
# # 4. 箱体颜色设置
# # ============================================================
# FILL_COLORS_2D = ["#dce6ee", "#c8d6df", "#b4c9db"]
# EDGE_COLORS_2D = ["#b7cfe4", "#8db1d9", "#6f91c7"]
#
# FILL_COLORS_8D = ["#dce6ee", "#c8d6df", "#b4c9db"]
# EDGE_COLORS_8D = ["#b7cfe4", "#8db1d9", "#6f91c7"]
#
#
# # ============================================================
# # 5. 读取 seed 结果 CSV
# # ============================================================
# rows = []
#
# with SEED_CSV_PATH.open("r", encoding="utf-8", newline="") as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         rows.append({
#             k: float(row[k]) if k != "seed" else int(row[k])
#             for k in row
#         })
#
# if len(rows) == 0:
#     raise ValueError("all_seed_results_summary.csv 文件为空，请检查。")
#
# all_seed_metrics = binary_metrics + multiclass_metrics
# missing_seed_cols = [m for m in all_seed_metrics if m not in rows[0]]
#
# if missing_seed_cols:
#     raise ValueError(
#         f"all_seed_results_summary.csv 中找不到以下列：{missing_seed_cols}\n"
#         f"实际列名为：{list(rows[0].keys())}"
#     )
#
# # 注意：这里不再乘以 100，保持 0-1 尺度
# binary_values = np.array(
#     [[row[m] for m in binary_metrics] for row in rows],
#     dtype=float
# )
#
# multiclass_values = np.array(
#     [[row[m] for m in multiclass_metrics] for row in rows],
#     dtype=float
# )
#
#
# # ============================================================
# # 6. 读取敏感性分析 CSV
# # ============================================================
# df_sens = pd.read_csv(SENS_FILE_PATH)
# df_sens.columns = [c.strip() for c in df_sens.columns]
#
# all_line_metrics = accuracy_metrics + auc_metrics
#
# for m in all_line_metrics:
#     if m not in df_sens.columns:
#         raise ValueError(
#             f"Sensitivity_results.csv 中找不到列 '{m}'，"
#             f"实际列名为：{list(df_sens.columns)}"
#         )
#
# if len(df_sens) != len(xs):
#     raise ValueError(
#         f"横轴 xs 长度为 {len(xs)}，但 Sensitivity_results.csv 行数为 {len(df_sens)}，请检查数据。"
#     )
#
#
# # ============================================================
# # 7. 统一坐标轴格式
# # 只保留左轴和下轴
# # ============================================================
# def format_axis(ax):
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.spines["left"].set_visible(True)
#     ax.spines["bottom"].set_visible(True)
#
#     ax.spines["left"].set_linewidth(AXIS_LINEWIDTH)
#     ax.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)
#     ax.spines["left"].set_color("black")
#     ax.spines["bottom"].set_color("black")
#
#     ax.tick_params(
#         axis="both",
#         labelsize=TICK_SIZE,
#         width=TICK_WIDTH,
#         length=3.5,
#         direction="out"
#     )
#
#     for tick in ax.get_xticklabels() + ax.get_yticklabels():
#         tick.set_fontfamily("DejaVu Sans")
#         tick.set_fontsize(TICK_SIZE)
#         tick.set_fontweight("normal")
#
#
# # ============================================================
# # 8. 竖向纵坐标标题
# # ============================================================
# def set_vertical_ylabel(ax, text):
#     ax.set_ylabel(
#         text,
#         fontsize=LABEL_SIZE,
#         fontweight="normal",
#         rotation=90,
#         labelpad=YLABEL_PAD
#     )
#
#
# # ============================================================
# # 9. 左侧箱形图纵轴刻度格式
# # 让 1.00 显示为 1，其余显示为两位小数，如 0.99
# # ============================================================
# def left_y_formatter(y, pos):
#     if np.isclose(y, 1.0):
#         return "1"
#     return f"{y:.2f}"
#
#
# # ============================================================
# # 10. 箱体图绘制函数
# # ============================================================
# def draw_colored_boxplot(ax, values, labels, fill_colors, edge_colors, ylim, show_ylabel=True):
#     positions = np.arange(1, values.shape[1] + 1)
#
#     bp = ax.boxplot(
#         [values[:, i] for i in range(values.shape[1])],
#         positions=positions,
#         widths=0.40,
#         patch_artist=True,
#         showmeans=False,
#         showfliers=False,
#         whis=1.5,
#         boxprops=dict(linewidth=0.8),
#         medianprops=dict(linewidth=0.9),
#         whiskerprops=dict(linewidth=0.8),
#         capprops=dict(linewidth=0.8),
#     )
#
#     for i in range(values.shape[1]):
#         bp["boxes"][i].set_facecolor(fill_colors[i])
#         bp["boxes"][i].set_edgecolor(edge_colors[i])
#         bp["boxes"][i].set_alpha(0.95)
#         bp["boxes"][i].set_linewidth(0.8)
#
#         bp["medians"][i].set_color(edge_colors[i])
#         bp["medians"][i].set_linewidth(0.9)
#
#         bp["whiskers"][2 * i].set_color(edge_colors[i])
#         bp["whiskers"][2 * i + 1].set_color(edge_colors[i])
#         bp["whiskers"][2 * i].set_linewidth(0.8)
#         bp["whiskers"][2 * i + 1].set_linewidth(0.8)
#
#         bp["caps"][2 * i].set_color(edge_colors[i])
#         bp["caps"][2 * i + 1].set_color(edge_colors[i])
#         bp["caps"][2 * i].set_linewidth(0.8)
#         bp["caps"][2 * i + 1].set_linewidth(0.8)
#
#     # 散点放在箱体右边
#     rng = np.random.default_rng(42)
#     for i in range(values.shape[1]):
#         jitter = rng.normal(0, 0.03, size=values.shape[0])
#         ax.scatter(
#             np.full(values.shape[0], positions[i] + 0.35) + jitter,
#             values[:, i],
#             s=20,
#             facecolors="none",
#             edgecolors=edge_colors[i],
#             linewidths=1.0,
#             alpha=0.95,
#             zorder=3
#         )
#
#     ax.set_xticks(positions)
#     ax.set_xticklabels(labels, fontsize=FONT_SIZE, fontweight="normal")
#     ax.set_ylim(*ylim)
#     ax.set_xlim(0.5, values.shape[1] + 0.75)
#
#     # 左侧两个箱形图纵坐标显示为 1、0.99 这种格式
#     ax.yaxis.set_major_formatter(FuncFormatter(left_y_formatter))
#
#     ax.grid(False)
#
#     if show_ylabel:
#         set_vertical_ylabel(ax, "Performance")
#
#     format_axis(ax)
#
#
# # ============================================================
# # 11. 折线图绘制函数
# # ============================================================
# def draw_lineplot(ax, df, metrics, ylabel, ylim, show_xlabel=False):
#     for m in metrics:
#         y_vals = df[m].to_numpy()
#
#         ax.plot(
#             xs,
#             y_vals,
#             marker="o",
#             markersize=4.2,
#             linewidth=1.8,
#             color=line_colors[m],
#             label=line_titles[m],
#             clip_on=False
#         )
#
#     set_vertical_ylabel(ax, ylabel)
#
#     ax.set_ylim(*ylim)
#     ax.set_xticks(xs)
#
#     # 右侧两个折线子图的纵轴刻度保留两位小数
#     ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
#
#     if show_xlabel:
#         ax.set_xlabel(
#             r"Upper bound of $T_s*\gamma$",
#             fontsize=LABEL_SIZE,
#             fontweight="normal",
#             labelpad=5
#         )
#     else:
#         ax.set_xticklabels([])
#
#     ax.legend(
#         fontsize=8,
#         frameon=False,
#         loc="best"
#     )
#
#     ax.grid(
#         True,
#         linestyle="--",
#         linewidth=0.5,
#         color="#d9d9d9",
#         alpha=0.8
#     )
#
#     format_axis(ax)
#
#
# # ============================================================
# # 12. 组合绘图
# # 左边：1行2列箱形图
# # 右边：2行1列折线图
# # ============================================================
# fig = plt.figure(figsize=(15, 3.5))
#
# outer = fig.add_gridspec(
#     nrows=1,
#     ncols=2,
#     width_ratios=[1, 1],
#     wspace=0.05
# )
#
# left_gs = outer[0].subgridspec(
#     nrows=1,
#     ncols=2,
#     wspace=0.20
# )
#
# right_gs = outer[1].subgridspec(
#     nrows=2,
#     ncols=1,
#     hspace=0.30
# )
#
# ax_box_2d = fig.add_subplot(left_gs[0, 0])
# ax_box_8d = fig.add_subplot(left_gs[0, 1])
#
# ax_line_acc = fig.add_subplot(right_gs[0, 0])
# ax_line_auc = fig.add_subplot(right_gs[1, 0])
#
#
# # -----------------------
# # 左侧：箱体图
# # -----------------------
# draw_colored_boxplot(
#     ax=ax_box_2d,
#     values=binary_values,
#     labels=binary_labels,
#     fill_colors=FILL_COLORS_2D,
#     edge_colors=EDGE_COLORS_2D,
#     ylim=(0.990, 1.0005),
#     show_ylabel=True
# )
#
# draw_colored_boxplot(
#     ax=ax_box_8d,
#     values=multiclass_values,
#     labels=multiclass_labels,
#     fill_colors=FILL_COLORS_8D,
#     edge_colors=EDGE_COLORS_8D,
#     ylim=(0.945, 1.0005),
#     show_ylabel=False
# )
#
#
# # -----------------------
# # 右侧：折线图
# # -----------------------
# draw_lineplot(
#     ax=ax_line_acc,
#     df=df_sens,
#     metrics=accuracy_metrics,
#     ylabel="Accuracy",
#     ylim=(0.8, 1.006),
#     show_xlabel=False
# )
#
# draw_lineplot(
#     ax=ax_line_auc,
#     df=df_sens,
#     metrics=auc_metrics,
#     ylabel="AUC",
#     ylim=(0.98, 1.0015),
#     show_xlabel=True
# )
#
# ax_line_acc.set_box_aspect(0.25)
# ax_line_auc.set_box_aspect(0.25)
#
#
# # ============================================================
# # 13. 加 (a) 和 (b)
# # ============================================================
# ax_box_2d.text(
#     -0.3, 1.0, "(a)",
#     transform=ax_box_2d.transAxes,
#     fontsize=12,
#     fontweight="normal",
#     ha="left",
#     va="bottom"
# )
#
# ax_line_acc.text(
#     -0.2, 1.0, "(b)",
#     transform=ax_line_acc.transAxes,
#     fontsize=12,
#     fontweight="normal",
#     ha="left",
#     va="bottom"
# )
#
#
# # ============================================================
# # 14. 保存与显示
# # ============================================================
# fig.subplots_adjust(
#     left=0.06,
#     right=0.985,
#     bottom=0.18,
#     top=0.94
# )
#
# plt.savefig(OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.03)
# plt.savefig(OUT_EPS, format="eps", bbox_inches="tight", pad_inches=0.03)
#
# plt.show()


import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter, FuncFormatter


# ============================================================
# JMB画图模版
# ============================================================

# FIG_WIDTH = 174 / 25.4          # 174 mm, full text width
# FIG_HEIGHT = 50 / 25.4          # 自定义
#
# FONT_FAMILY = "Arial"
# TITLE_SIZE = 10                 # Subfigure labels, e.g., (a), (b)
# TEXT_SIZE = 10                  # Text inside figures
# AXIS_LABEL_SIZE = 10            # x/y axis labels
# TICK_LABEL_SIZE = 8             # x/y tick labels
# LEGEND_SIZE = 8                 # Legend text
FIG_WIDTH = 174 / 15.4          # 174 mm, full text width
FIG_HEIGHT = 50 / 15.4          # 自定义

FONT_FAMILY = "Arial"
TITLE_SIZE = 14.5
TEXT_SIZE = 14.5
AXIS_LABEL_SIZE = 13
TICK_LABEL_SIZE = 9
LEGEND_SIZE = 10

mpl.rcParams.update({
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": [FONT_FAMILY],
    "font.size": TEXT_SIZE,

    # Math font: italic math symbols, non-italic text via \mathrm{} or plain strings
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_FAMILY,
    "mathtext.it": f"{FONT_FAMILY}:italic",
    "mathtext.bf": f"{FONT_FAMILY}:bold",
    "mathtext.default": "it",

    # Axis
    "axes.linewidth": 0.6,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.titleweight": "normal",
    "axes.xmargin": 0.1,

    # Ticks
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,

    # Legend
    "legend.loc": "upper right",
    "legend.fontsize": LEGEND_SIZE,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.fancybox": False,

    # Output font embedding
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FONT_SIZE = TEXT_SIZE
LABEL_SIZE = AXIS_LABEL_SIZE
TICK_SIZE = TICK_LABEL_SIZE
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
# 1. 路径设置
# ============================================================
SEED_CSV_PATH = Path("all_seed_results_summary.csv")

SENS_BASE_DIR = "Sensitivity_Ts_data"
SENS_FILE_NAME = "Sensitivity_results.csv"
SENS_FILE_PATH = os.path.join(SENS_BASE_DIR, SENS_FILE_NAME)

OUT_PNG = Path("seed_and_Ts.png")
OUT_EPS = Path("seed_and_Ts.eps")


# ============================================================
# 2. 箱体图指标设置
# ============================================================
binary_metrics = [
    "binary_accuracy",
    "binary_auc",
    "binary_f1",
]


multiclass_metrics = [
    "multiclass_accuracy",
    "multiclass_macro_auc",
    "multiclass_macro_f1",
]

binary_labels = [
    "Accuracy\n(2 class)",
    "AUC\n(2 class)",
    "F1\n(2 class)",
]

multiclass_labels = [
    "Accuracy\n(8 class)",
    "Macro-AUC\n(8 class)",
    "Macro-F1\n(8 class)",
]


# ============================================================
# 3. 折线图指标设置
# ============================================================
xs = list(range(100, 701, 100))

accuracy_metrics = [
    "2d_accuracy",
    "8d_accuracy",
]

auc_metrics = [
    "2d_auc",
    "8d_macro_auc",
]

line_titles = {
    "2d_accuracy": "Accuracy (2 class)",
    "2d_auc": "AUC (2 class)",
    "8d_accuracy": "Accuracy (8 class)",
    "8d_macro_auc": "Macro-AUC (8 class)",
}

line_colors = {
    "2d_accuracy": "#b51519",
    "2d_auc": "#e9a2a4",
    "8d_accuracy": "#2354a1",
    "8d_macro_auc": "#73a4cb",
}


# ============================================================
# 4. 箱体颜色设置
# ============================================================
FILL_COLORS_2D = ["#dce6ee", "#c8d6df", "#b4c9db"]
EDGE_COLORS_2D = ["#b7cfe4", "#8db1d9", "#6f91c7"]

FILL_COLORS_8D = ["#dce6ee", "#c8d6df", "#b4c9db"]
EDGE_COLORS_8D = ["#b7cfe4", "#8db1d9", "#6f91c7"]


# ============================================================
# 5. 读取 seed 结果 CSV
# ============================================================
rows = []

with SEED_CSV_PATH.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({
            k: float(row[k]) if k != "seed" else int(row[k])
            for k in row
        })

if len(rows) == 0:
    raise ValueError("all_seed_results_summary.csv 文件为空，请检查。")

all_seed_metrics = binary_metrics + multiclass_metrics
missing_seed_cols = [m for m in all_seed_metrics if m not in rows[0]]

if missing_seed_cols:
    raise ValueError(
        f"all_seed_results_summary.csv 中找不到以下列：{missing_seed_cols}\n"
        f"实际列名为：{list(rows[0].keys())}"
    )

binary_values = np.array(
    [[row[m] for m in binary_metrics] for row in rows],
    dtype=float
)

multiclass_values = np.array(
    [[row[m] for m in multiclass_metrics] for row in rows],
    dtype=float
)


# ============================================================
# 6. 读取敏感性分析 CSV
# ============================================================
df_sens = pd.read_csv(SENS_FILE_PATH)
df_sens.columns = [c.strip() for c in df_sens.columns]

all_line_metrics = accuracy_metrics + auc_metrics

for m in all_line_metrics:
    if m not in df_sens.columns:
        raise ValueError(
            f"Sensitivity_results.csv 中找不到列 '{m}'，"
            f"实际列名为：{list(df_sens.columns)}"
        )

if len(df_sens) != len(xs):
    raise ValueError(
        f"横轴 xs 长度为 {len(xs)}，但 Sensitivity_results.csv 行数为 {len(df_sens)}，请检查数据。"
    )


# ============================================================
# 7. 统一坐标轴格式
# ============================================================
def format_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    ax.spines["left"].set_linewidth(AXIS_LINEWIDTH)
    ax.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    ax.tick_params(
        axis="both",
        labelsize=TICK_SIZE,
        width=TICK_WIDTH,
        length=3.0,
        direction="out"
    )

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily(FONT_FAMILY)
        tick.set_fontsize(TICK_SIZE)
        tick.set_fontweight("normal")


# ============================================================
# 8. 竖向纵坐标标题
# ============================================================
def set_vertical_ylabel(ax, text):
    ax.set_ylabel(
        text,
        fontsize=LABEL_SIZE,
        fontweight="normal",
        rotation=90,
        labelpad=YLABEL_PAD
    )


# ============================================================
# 9. 左侧箱形图纵轴刻度格式
# ============================================================
def left_y_formatter(y, pos):
    if np.isclose(y, 1.0):
        return "1"
    return f"{y:.2f}"


# ============================================================
# 10. 箱体图绘制函数
# ============================================================
def draw_colored_boxplot(ax, values, labels, fill_colors, edge_colors, ylim, show_ylabel=True):
    positions = np.arange(1, values.shape[1] + 1)

    bp = ax.boxplot(
        [values[:, i] for i in range(values.shape[1])],
        positions=positions,
        widths=0.40,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        whis=1.5,
        boxprops=dict(linewidth=0.6),
        medianprops=dict(linewidth=0.7),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
    )

    for i in range(values.shape[1]):
        bp["boxes"][i].set_facecolor(fill_colors[i])
        bp["boxes"][i].set_edgecolor(edge_colors[i])
        bp["boxes"][i].set_linewidth(0.6)

        bp["medians"][i].set_color(edge_colors[i])
        bp["medians"][i].set_linewidth(0.7)

        bp["whiskers"][2 * i].set_color(edge_colors[i])
        bp["whiskers"][2 * i + 1].set_color(edge_colors[i])
        bp["whiskers"][2 * i].set_linewidth(0.6)
        bp["whiskers"][2 * i + 1].set_linewidth(0.6)

        bp["caps"][2 * i].set_color(edge_colors[i])
        bp["caps"][2 * i + 1].set_color(edge_colors[i])
        bp["caps"][2 * i].set_linewidth(0.6)
        bp["caps"][2 * i + 1].set_linewidth(0.6)

    rng = np.random.default_rng(42)
    for i in range(values.shape[1]):
        jitter = rng.normal(0, 0.03, size=values.shape[0])
        ax.scatter(
            np.full(values.shape[0], positions[i] + 0.35) + jitter,
            values[:, i],
            s=12,
            facecolors="none",
            edgecolors=edge_colors[i],
            linewidths=0.6,
            zorder=3
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE, fontweight="normal")
    ax.set_ylim(*ylim)
    ax.set_xlim(0.5, values.shape[1] + 0.75)

    ax.yaxis.set_major_formatter(FuncFormatter(left_y_formatter))
    ax.grid(False)

    if show_ylabel:
        set_vertical_ylabel(ax, "Performance")

    format_axis(ax)


# ============================================================
# 11. 折线图绘制函数
# ============================================================
def draw_lineplot(ax, df, metrics, ylabel, ylim, show_xlabel=False):
    for m in metrics:
        y_vals = df[m].to_numpy()

        ax.plot(
            xs,
            y_vals,
            marker="o",
            markersize=3.0,
            linewidth=1.0,
            color=line_colors[m],
            label=line_titles[m],
            clip_on=False
        )

    set_vertical_ylabel(ax, ylabel)

    ax.set_ylim(*ylim)
    ax.set_xticks(xs)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    if show_xlabel:
        ax.set_xlabel(
            r"$U_{\max}\gamma$",
            fontsize=LABEL_SIZE,
            fontweight="normal",
            labelpad=4
        )
    else:
        ax.set_xticklabels([])

    ax.legend(
        fontsize=LEGEND_SIZE,
        frameon=True,
        loc="best"
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.4,
        color="#d9d9d9"
    )

    format_axis(ax)


# ============================================================
# 12. 组合绘图
# ============================================================
fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

outer = fig.add_gridspec(
    nrows=1,
    ncols=2,
    width_ratios=[1, 1],
    wspace=0.08
)

left_gs = outer[0].subgridspec(
    nrows=1,
    ncols=2,
    wspace=0.22
)

right_gs = outer[1].subgridspec(
    nrows=2,
    ncols=1,
    hspace=0.32
)

ax_box_2d = fig.add_subplot(left_gs[0, 0])
ax_box_8d = fig.add_subplot(left_gs[0, 1])

ax_line_acc = fig.add_subplot(right_gs[0, 0])
ax_line_auc = fig.add_subplot(right_gs[1, 0])


# -----------------------
# 左侧：箱体图
# -----------------------
draw_colored_boxplot(
    ax=ax_box_2d,
    values=binary_values,
    labels=binary_labels,
    fill_colors=FILL_COLORS_2D,
    edge_colors=EDGE_COLORS_2D,
    ylim=(0.990, 1.0005),
    show_ylabel=True
)

draw_colored_boxplot(
    ax=ax_box_8d,
    values=multiclass_values,
    labels=multiclass_labels,
    fill_colors=FILL_COLORS_8D,
    edge_colors=EDGE_COLORS_8D,
    ylim=(0.945, 1.0005),
    show_ylabel=False
)


# -----------------------
# 右侧：折线图
# -----------------------
draw_lineplot(
    ax=ax_line_acc,
    df=df_sens,
    metrics=accuracy_metrics,
    ylabel="Accuracy",
    ylim=(0.8, 1.006),
    show_xlabel=False
)

draw_lineplot(
    ax=ax_line_auc,
    df=df_sens,
    metrics=auc_metrics,
    ylabel="AUC",
    ylim=(0.98, 1.0015),
    show_xlabel=True
)

ax_line_acc.set_box_aspect(0.25)
ax_line_auc.set_box_aspect(0.25)


# ============================================================
# 13. 加 (a) 和 (b)，正上方居中
# ============================================================
title_left_ax = fig.add_subplot(
    outer[0],
    frameon=False
)
title_left_ax.set_title(
    "(a)",
    fontsize=TITLE_SIZE,
    fontweight="normal",
    loc="center",
    pad=6
)
title_left_ax.axis("off")

title_right_ax = fig.add_subplot(
    outer[1],
    frameon=False
)
title_right_ax.set_title(
    "(b)",
    fontsize=TITLE_SIZE,
    fontweight="normal",
    loc="center",
    pad=6
)
title_right_ax.axis("off")


# ============================================================
# 14. 保存与显示
# ============================================================
fig.subplots_adjust(
    left=0.07,
    right=1,
    bottom=0.20,
    top=0.90
)


plt.savefig(OUT_EPS, format="eps", bbox_inches="tight", pad_inches=0.03)

plt.show()