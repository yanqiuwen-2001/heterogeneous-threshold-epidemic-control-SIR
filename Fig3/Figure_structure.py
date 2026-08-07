import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle, FancyArrowPatch

FIG_WIDTH = 174 / 15.4          # 174 mm, full text width
FIG_HEIGHT = 90 / 15.4          # 自定义

FONT_FAMILY = "Arial"
TITLE_SIZE = 14.5
TEXT_SIZE = 14.5
AXIS_LABEL_SIZE = 13
TICK_LABEL_SIZE = 9
LEGEND_SIZE = 13

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
    "axes.xmargin": 0.06,

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

# ---------------- 参数设置 ---------------- #
G = 8

node_radius = 0.2
circle_radius = 1.14
line_width = 1.0
arrow_size = 8

node_edge_color = "black"
default_node_color = "white"
highlight_red = "red"
highlight_yellow = "yellow"

blue_stem_color = "#377eb8"
red_point_color = "#e41a1c"

purple_stem_color = "#984ea3"
yellow_point_color = "#ffcc00"


# ============================================================
# 网络结构图函数
# ============================================================
def circular_positions(n, radius=1.0, center=(0, 0), start_angle=np.pi / 2):
    """在圆周上均匀生成节点坐标。"""
    angles = start_angle + np.arange(n) * 2 * np.pi / n

    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)

    return np.column_stack((x, y))


def draw_node(ax, position, face_color=default_node_color):
    """绘制单个节点。"""
    node = Circle(
        position,
        radius=node_radius,
        facecolor=face_color,
        edgecolor=node_edge_color,
        linewidth=line_width,
        zorder=3
    )

    ax.add_patch(node)


def shorten_line(start, end, offset=node_radius):
    """缩短线段，使箭头位于节点边缘。"""
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)

    direction = end - start
    distance = np.linalg.norm(direction)

    if distance == 0:
        raise ValueError("两个节点的位置不能相同。")

    unit_direction = direction / distance

    new_start = start + offset * unit_direction
    new_end = end - offset * unit_direction

    return new_start, new_end


def draw_edge(ax, start, end, bidirectional=False):
    """绘制单向或双向连接。"""
    new_start, new_end = shorten_line(start, end)

    arrow_style = "<->" if bidirectional else "<-"

    edge = FancyArrowPatch(
        new_start,
        new_end,
        arrowstyle=arrow_style,
        mutation_scale=arrow_size,
        linewidth=line_width,
        color="black",
        shrinkA=0,
        shrinkB=0,
        zorder=2
    )

    ax.add_patch(edge)


def format_network_axis(ax):
    """统一设置网络结构图。"""
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")


def draw_star(
    ax,
    bidirectional=False,
    outward=False,
    highlight_outer_node=False,
    center_color=highlight_red,
    outer_highlight_color=highlight_yellow
):
    """绘制 Star 图。"""
    center = np.array([0.0, 0.0])

    outer_positions = circular_positions(
        G - 1,
        radius=circle_radius
    )

    for outer_node in outer_positions:
        if outward and not bidirectional:
            draw_edge(
                ax,
                outer_node,
                center,
                bidirectional=False
            )
        else:
            draw_edge(
                ax,
                center,
                outer_node,
                bidirectional=bidirectional
            )

    draw_node(
        ax,
        center,
        face_color=center_color
    )

    for index, outer_node in enumerate(outer_positions):
        is_highlighted_node = (
            highlight_outer_node
            and index == len(outer_positions) - 1
        )

        if is_highlighted_node:
            draw_node(
                ax,
                outer_node,
                face_color=outer_highlight_color
            )
        else:
            draw_node(
                ax,
                outer_node
            )


def draw_cycle(ax, bidirectional=False):
    """绘制循环图。"""
    positions = circular_positions(
        G,
        radius=circle_radius
    )

    for index in range(G):
        next_index = (index + 1) % G

        draw_edge(
            ax,
            positions[index],
            positions[next_index],
            bidirectional=bidirectional
        )

    for index, position in enumerate(positions):
        if index == 0:
            draw_node(
                ax,
                position,
                face_color=highlight_red
            )
        else:
            draw_node(
                ax,
                position
            )


# ============================================================
# CSV 数据处理函数
# ============================================================
def ratio_to_float(series):
    series = series.astype(str).str.strip()

    if series.str.contains("%").any():
        values = (
            series
            .str.rstrip("%")
            .astype(float)
            .to_numpy()
            / 100
        )
    else:
        values = pd.to_numeric(
            series,
            errors="raise"
        ).to_numpy()

        if np.nanmax(values) > 1:
            values = values / 100

    return values


def read_summary_data(file_name, x_column, y_column):
    df = pd.read_csv(file_name)

    for column in [x_column, y_column]:
        if column not in df.columns:
            raise ValueError(
                f"{file_name} 中缺少列：{column}"
            )

    x = pd.to_numeric(
        df[x_column],
        errors="raise"
    ).to_numpy()

    y = ratio_to_float(
        df[y_column]
    )

    order = np.argsort(x)

    return x[order], y[order]


# ============================================================
# 竖线图函数
# ============================================================
def draw_stem_plot(
    ax,
    file_name,
    x_column="center_diagonal_value",
    y_column="yes_ratio",
    stem_color=blue_stem_color,
    point_color=red_point_color
):
    """绘制竖线和顶端圆点。"""
    x, y = read_summary_data(
        file_name,
        x_column,
        y_column
    )

    ax.vlines(
        x,
        ymin=0,
        ymax=y,
        color=stem_color,
        linewidth=1.0
    )

    ax.scatter(
        x,
        y,
        color=point_color,
        edgecolor="black",
        linewidth=0.45,
        s=24,
        zorder=3
    )

    ax.set_xlabel(
        r"$c_{ii}$",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=4
    )

    ax.set_ylabel(
        "Ratio",
        fontsize=AXIS_LABEL_SIZE,
        rotation=0,
        labelpad=18
    )
    ax.yaxis.set_label_coords(-0.3, 0.47)

    ax.set_ylim(
        0,
        1.05
    )

    ax.set_xticks(
        x
    )

    ax.tick_params(
        axis="both",
        labelsize=TICK_LABEL_SIZE,
        width=0.6
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


# ============================================================
# 标准组合图：左侧网络，右侧一个竖线图
# ============================================================
def add_standard_panel(
    fig,
    grid_position,
    title,
    network_type,
    bidirectional,
    csv_file,
    outward=False
):
    inner_grid = grid_position.subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=[0.9, 0.9],
        wspace=0.52
    )

    network_ax = fig.add_subplot(
        inner_grid[0, 0]
    )

    stem_ax = fig.add_subplot(
        inner_grid[0, 1]
    )

    if network_type == "star":
        draw_star(
            network_ax,
            bidirectional=bidirectional,
            outward=outward,
            highlight_outer_node=False
        )

    elif network_type == "cycle":
        draw_cycle(
            network_ax,
            bidirectional=bidirectional
        )

    else:
        raise ValueError(
            f"未知结构：{network_type}"
        )

    format_network_axis(
        network_ax
    )

    draw_stem_plot(
        stem_ax,
        file_name=csv_file
    )

    title_ax = fig.add_subplot(
        grid_position,
        frameon=False
    )

    title_ax.set_title(
        title,
        fontsize=TITLE_SIZE,
        fontweight="normal",
        pad=10,
        loc="center"
    )

    title_ax.axis("off")


# ============================================================
# 第五个组合图：左侧网络，右侧两个竖线图
# ============================================================
def add_distinct_centers_panel(
    fig,
    grid_position,
    title
):
    inner_grid = grid_position.subgridspec(
        nrows=1,
        ncols=3,
        width_ratios=[1.10, 0.90, 0.90],
        wspace=0.55
    )

    network_ax = fig.add_subplot(
        inner_grid[0, 0]
    )

    transmission_center_ax = fig.add_subplot(
        inner_grid[0, 1]
    )

    structural_center_ax = fig.add_subplot(
        inner_grid[0, 2]
    )

    draw_star(
        network_ax,
        bidirectional=True,
        highlight_outer_node=True,
        center_color=highlight_yellow,
        outer_highlight_color=highlight_red
    )

    format_network_axis(
        network_ax
    )

    draw_stem_plot(
        transmission_center_ax,
        file_name="Center_or_beta_summary.csv",
        x_column="outer_center_value",
        y_column="alpha*=center Yes ratio",
        stem_color=blue_stem_color,
        point_color=red_point_color
    )

    draw_stem_plot(
        structural_center_ax,
        file_name="Center_or_beta_summary.csv",
        x_column="outer_center_value",
        y_column="alpha*=structure Yes ratio",
        stem_color=purple_stem_color,
        point_color=yellow_point_color
    )

    title_ax = fig.add_subplot(
        grid_position,
        frameon=False
    )

    title_ax.set_title(
        title,
        fontsize=TITLE_SIZE,
        fontweight="normal",
        pad=10,
        loc="center"
    )

    title_ax.axis("off")


# ============================================================
# 绘图：两行，共五个组合图
# ============================================================
fig = plt.figure(
    figsize=(FIG_WIDTH, FIG_HEIGHT)
)

outer_grid = fig.add_gridspec(
    nrows=2,
    ncols=6,
    wspace=0.20,
    hspace=0.30
)

# ---------------- 第一行 ---------------- #
add_standard_panel(
    fig,
    outer_grid[0, 0:2],
    title="(a)",
    network_type="star",
    bidirectional=True,
    csv_file="Bi-Star_summary.csv"
)

add_standard_panel(
    fig,
    outer_grid[0, 2:4],
    title="(b)",
    network_type="star",
    bidirectional=False,
    csv_file="Uni-Star_summary.csv",
    outward=True
)

add_standard_panel(
    fig,
    outer_grid[0, 4:6],
    title="(c)",
    network_type="cycle",
    bidirectional=False,
    csv_file="Uni-Cycle_summary.csv"
)

# ---------------- 第二行 ---------------- #
add_standard_panel(
    fig,
    outer_grid[1, 0:2],
    title="(d)",
    network_type="cycle",
    bidirectional=True,
    csv_file="Bi-Cycle_summary.csv"
)

add_distinct_centers_panel(
    fig,
    outer_grid[1, 2:6],
    title="(e)"
)

# ---------------- 调整布局并保存 ---------------- #
fig.subplots_adjust(
    left=0.05,
    right=0.98,
    bottom=0.08,
    top=0.94,
    wspace=0.20,
    hspace=0.30
)

plt.savefig(
    "Figure_structure.eps",
    format="eps",
    bbox_inches="tight"
)

plt.show()
