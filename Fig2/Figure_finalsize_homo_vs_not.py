import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl

FIG_WIDTH = 174 / 25.4          # 174 mm, full text width
FIG_HEIGHT = 80 / 25.4          # 自定义

FONT_FAMILY = "Arial"
TITLE_SIZE = 10                 # Subfigure labels, e.g., (a), (b)
TEXT_SIZE = 10                  # Text inside figures
AXIS_LABEL_SIZE = 10            # x/y axis labels
TICK_LABEL_SIZE = 8             # x/y tick labels
LEGEND_SIZE = 8                 # Legend text

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
    "axes.xmargin": 0.05,

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

# =========================
# 读取参数
# =========================
G = 8
df = pd.read_csv("data_homo_vs_not.csv")
row = df.iloc[1]

beta0 = float(row["beta0"])
gam = float(row["gam"])
Ts = float(row["Ts"])

N = np.array([row[f"N{i}"] for i in range(1, G + 1)], dtype=float)
S0 = np.array([row[f"S14_{i}"] for i in range(1, G + 1)], dtype=float)
I0 = np.array([row[f"I14_{i}"] for i in range(1, G + 1)], dtype=float)
R0 = np.array([row[f"R14_{i}"] for i in range(1, G + 1)], dtype=float)

C = np.array(
    [[row[f"C{i}{j}"] for j in range(1, G + 1)] for i in range(1, G + 1)],
    dtype=float
)

E = np.array(
    [[row[f"E{i}{j}"] for j in range(1, G + 1)] for i in range(1, G + 1)],
    dtype=float
)

# =========================
# 时间设置
# =========================
t_end = 1000
t_eval = np.arange(0, t_end, 0.1)

initial = np.zeros(3 * G)
for i in range(G):
    initial[3 * i:3 * i + 3] = [S0[i], I0[i], R0[i]]


# =========================
# 系统定义
# =========================
def system_free(t, y):
    dydt = []
    for i in range(G):
        Si = y[3 * i]
        Ii = y[3 * i + 1]

        infection_sum = 0.0
        for j in range(G):
            Ij = y[3 * j + 1]
            infection_sum += C[i, j] * Ij

        dS_dt = -beta0 * infection_sum * Si / N[i]
        dI_dt = beta0 * infection_sum * Si / N[i] - gam * Ii
        dR_dt = gam * Ii
        dydt.extend([dS_dt, dI_dt, dR_dt])

    return dydt


def system_control(t, y):
    dydt = []
    for i in range(G):
        Si = y[3 * i]
        Ii = y[3 * i + 1]

        control_sum = 0.0
        for j in range(G):
            Ij = y[3 * j + 1]
            control_sum += C[i, j] * E[i, j] * Ij

        dS_dt = -beta0 * control_sum * Si / N[i]
        dI_dt = beta0 * control_sum * Si / N[i] - gam * Ii
        dR_dt = gam * Ii
        dydt.extend([dS_dt, dI_dt, dR_dt])

    return dydt


# =========================
# alpha 情况
# =========================
alpha_equal = np.ones(G) / G

alpha_star = np.array([row[f"alpha{i}*"] for i in range(1, G + 1)], dtype=float)
optimal_idx = int(np.argmax(alpha_star))

if not np.isclose(alpha_star[optimal_idx], 1.0):
    raise ValueError("alpha1* 到 alpha8* 中没有识别到等于 1 的最优 alpha。")

if not np.isclose(alpha_star.sum(), 1.0):
    raise ValueError("alpha1* 到 alpha8* 的和不是 1，请检查数据。")

print(f"识别到最优 alpha*: alpha{optimal_idx + 1} = 1")

rng = np.random.default_rng(128)

random_alphas = []
while len(random_alphas) < 3:
    alpha_random = rng.dirichlet(np.ones(G))
    is_equal = np.allclose(alpha_random, alpha_equal, atol=1e-6)
    is_unit = np.isclose(alpha_random.max(), 1.0, atol=1e-6)
    if (not is_equal) and (not is_unit):
        random_alphas.append(alpha_random)

for idx, alpha_random in enumerate(random_alphas, start=1):
    print(f"随机 alpha {idx} =", alpha_random)


# =========================
# 单个 alpha 情况模拟
# =========================
def get_I_vector(y):
    return np.array([y[3 * i + 1] for i in range(G)])


def simulate_alpha(alpha):
    def delta_y(y):
        return float(alpha @ get_I_vector(y))

    def event_reach_Ts(t, y):
        return delta_y(y) - Ts

    event_reach_Ts.terminal = True
    event_reach_Ts.direction = 1

    delta0 = float(alpha @ I0)

    if delta0 < Ts:
        sol1 = solve_ivp(
            system_free,
            [0, t_end],
            initial,
            events=event_reach_Ts,
            t_eval=t_eval,
            dense_output=True,
            rtol=1e-8,
            atol=1e-10
        )

        if sol1.t_events[0].size > 0:
            T_switch = float(sol1.t_events[0][0])
            y_switch = sol1.sol(T_switch)

            t_eval2 = t_eval[t_eval > T_switch]
            sol2 = solve_ivp(
                system_control,
                [T_switch, t_end],
                y_switch,
                t_eval=t_eval2,
                dense_output=True,
                rtol=1e-8,
                atol=1e-10
            )

            y_combined = np.hstack((sol1.y[:, sol1.t <= T_switch], sol2.y))
            switch_type = "free_to_control"

        else:
            y_combined = sol1.y
            switch_type = "free_only"

    else:
        sol = solve_ivp(
            system_control,
            [0, t_end],
            initial,
            t_eval=t_eval,
            dense_output=True,
            rtol=1e-8,
            atol=1e-10
        )

        y_combined = sol.y
        switch_type = "control_only"

    R_final = np.array([y_combined[3 * i + 2, -1] for i in range(G)])
    final_size = float(np.sum(R_final - R0))

    return final_size, switch_type


# =========================
# 构建四类结果
# =========================
groups = []

final_size, switch_type = simulate_alpha(alpha_star)
groups.append([
    {
        "label": rf"$\substack{{\alpha^{{\ast}}\\(\alpha_{{{optimal_idx + 1}}}=1)}}$",
        "alpha": alpha_star,
        "final_size": final_size,
        "switch_type": switch_type,
        "group": "optimal"
    }
])

final_size, switch_type = simulate_alpha(alpha_equal)
groups.append([
    {
        "label": r"$\substack{\mathrm{Homogeneous}\\(\alpha_i=1/8)}$",
        "alpha": alpha_equal,
        "final_size": final_size,
        "switch_type": switch_type,
        "group": "homogeneous"
    }
])

random_group = []
for idx, alpha_random in enumerate(random_alphas, start=1):
    final_size, switch_type = simulate_alpha(alpha_random)
    random_group.append({
        "label": rf"$\substack{{\mathrm{{Random}}\\{idx}}}$",
        "alpha": alpha_random,
        "final_size": final_size,
        "switch_type": switch_type,
        "group": "random"
    })
groups.append(random_group)

unit_group = []
for i in range(G):
    if i != optimal_idx:
        a = np.zeros(G)
        a[i] = 1.0
        final_size, switch_type = simulate_alpha(a)
        unit_group.append({
            "label": rf"$\alpha_{{{i + 1}}}=1$",
            "alpha": a,
            "final_size": final_size,
            "switch_type": switch_type,
            "group": "unit"
        })
groups.append(unit_group)



# =========================
# 生成带间隔的横坐标
# =========================
x_positions = []
plot_items = []
current_x = 0.0

GROUP_GAP = 1.2        # 不同组之间的间隔
NORMAL_STEP = 1.0      # 第一组、第二组内部间隔
WIDE_STEP = 1.5     # 第三组、第四组内部每个柱子的间隔

for group_idx, group in enumerate(groups):
    if group_idx in [2, 3]:
        step = WIDE_STEP
    else:
        step = NORMAL_STEP

    for item in group:
        x_positions.append(current_x)
        plot_items.append(item)
        current_x += step

    current_x += GROUP_GAP

x_positions = np.array(x_positions)
final_sizes = np.array([item["final_size"] for item in plot_items])
bar_labels = [item["label"] for item in plot_items]

cmap = plt.cm.Blues
group_colors = {
    "optimal": cmap(0.88),
    "homogeneous": cmap(0.70),
    "random": cmap(0.52),
    "unit": cmap(0.34),
}

bar_colors = [group_colors[item["group"]] for item in plot_items]

summary = pd.DataFrame({
    "case": [item["label"] for item in plot_items],
    "final_size": [item["final_size"] for item in plot_items],
    "switch_type": [item["switch_type"] for item in plot_items],
    "group": [item["group"] for item in plot_items]
})
# summary.to_csv("Alpha_cases_finalsize_summary.csv", index=False)
# print(summary)


# =========================
# 画柱状图
# =========================
fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

bars = ax.bar(
    x_positions,
    final_sizes,
    color=bar_colors,
    edgecolor="white",
    linewidth=0.9,
    width=0.8
)

ax.set_xticks(x_positions)
ax.set_xticklabels(bar_labels, fontsize=TICK_LABEL_SIZE, rotation=0)

ax.set_ylabel(
    "Final Size",
    fontsize=AXIS_LABEL_SIZE,
    rotation=0,
    labelpad=42
)
ax.yaxis.set_label_coords(-0.10, 0.5)

ax.set_ylim(1e5, 6e5)
yticks = np.arange(1, 7) * 1e5
ax.set_yticks(yticks)
ax.set_yticklabels(
    [rf"${i}\times 10^5$" for i in range(1, 7)],
    fontsize=TICK_LABEL_SIZE
)

ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE, width=0.6)
ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, width=0.6)

ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#d9d9d9", alpha=0.9)
ax.set_axisbelow(True)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.015e5,
        f"{height:.0f}",
        ha="center",
        va="bottom",
        fontsize=TICK_LABEL_SIZE,
        rotation=0,
        color="#333333"
    )

for spine in ax.spines.values():
    spine.set_linewidth(0.6)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("Alpha_cases_finalsize_bar.eps", format="eps", bbox_inches="tight")

plt.show()