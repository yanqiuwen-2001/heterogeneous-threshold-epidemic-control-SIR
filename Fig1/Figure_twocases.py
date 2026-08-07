import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl


FIG_WIDTH = 174 / 25.4          # 174 mm, full text width
FIG_HEIGHT = 75 / 25.4         # 自定义

FONT_FAMILY = "Arial"
TITLE_SIZE = 10                 # Subfigure labels, e.g., (a), (b)
TEXT_SIZE = 10                  # Text inside figures
AXIS_LABEL_SIZE = 10            # x/y axis labels
TICK_LABEL_SIZE = 8             # x/y tick labels
LEGEND_SIZE = 8                 # Legend text

mpl.rcParams.update({
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": [FONT_FAMILY],
    "font.size": TEXT_SIZE,

    # Math font
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
    "axes.xmargin": 0,

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

# ------------------ 参数设置 ------------------ #
beta0, gam, c11, c21, c12, c22, e11, e21, e12, e22, N1, N2 = (
    0.05, 1 / 7, 5, 7, 7, 8, 0.6, 0.6, 0.6, 0.6, 1000, 1500
)

R10, R20, I10, I20 = 300, 400, 40, 60

S10 = N1 - R10 - I10
S20 = N2 - R20 - I20
t_end = 80
t_eval = np.arange(0, t_end, 0.001)

v1 = I10 + S10
v2 = I20 + S20
a11 = beta0 * c11 / N1
a21 = beta0 * c21 / N1
a12 = beta0 * c12 / N2
a22 = beta0 * c22 / N2
b11 = beta0 * e11 * c11 / N1
b21 = beta0 * e21 * c21 / N1
b12 = beta0 * e12 * c12 / N2
b22 = beta0 * e22 * c22 / N2
B1 = np.array([[a11, a21], [a12, a22]])
B2 = np.array([[b11, b21], [b12, b22]])
E0 = np.array([[1, 0], [0, 1]])


def system1(t, y):
    S1, I1, R1, S2, I2, R2 = y
    dS1_dt = -beta0 * (c11 * I1 + c21 * I2) * S1 / N1
    dI1_dt = beta0 * (c11 * I1 + c21 * I2) * S1 / N1 - gam * I1
    dR1_dt = gam * I1
    dS2_dt = -beta0 * (c12 * I1 + c22 * I2) * S2 / N2
    dI2_dt = beta0 * (c12 * I1 + c22 * I2) * S2 / N2 - gam * I2
    dR2_dt = gam * I2
    return [dS1_dt, dI1_dt, dR1_dt, dS2_dt, dI2_dt, dR2_dt]


def system2(t, y):
    S1, I1, R1, S2, I2, R2 = y
    dS1_dt = -beta0 * (e11 * c11 * I1 + e21 * c21 * I2) * S1 / N1
    dI1_dt = beta0 * (e11 * c11 * I1 + e21 * c21 * I2) * S1 / N1 - gam * I1
    dR1_dt = gam * I1
    dS2_dt = -beta0 * (e12 * c12 * I1 + e22 * c22 * I2) * S2 / N2
    dI2_dt = beta0 * (e12 * c12 * I1 + e22 * c22 * I2) * S2 / N2 - gam * I2
    dR2_dt = gam * I2
    return [dS1_dt, dI1_dt, dR1_dt, dS2_dt, dI2_dt, dR2_dt]


alpha1 = 0.5
alpha2 = 1 - alpha1


def run_case(Ts):
    y0 = [S10, I10, R10, S20, I20, R20]

    def delta(I1, I2):
        return alpha1 * I1 + alpha2 * I2

    def delta_reach_Ts(t, y):
        return delta(y[1], y[4]) - Ts

    delta_reach_Ts.terminal = True
    delta_reach_Ts.direction = 0

    T1 = None
    T2 = None
    Delta_T1 = None
    Delta_T2 = None
    delta0 = delta(I10, I20)

    if delta0 <= Ts:
        sol1 = solve_ivp(
            system1, [0, t_end], y0,
            events=delta_reach_Ts,
            t_eval=t_eval,
            dense_output=True,
            rtol=1e-8,
            atol=1e-10
        )

        if sol1.t_events[0].size > 0:
            T1 = sol1.t_events[0][0]
            y_switch = sol1.sol(T1)
            Delta_T1 = alpha1 * y_switch[1] + alpha2 * y_switch[4]

            t_eval2 = t_eval[t_eval > T1]
            sol2 = solve_ivp(
                system2, [T1, t_end], y_switch,
                t_eval=t_eval2,
                dense_output=True,
                rtol=1e-8,
                atol=1e-10
            )

            y_combined = np.hstack((sol1.y[:, sol1.t <= T1], sol2.y))
            t_combined = np.concatenate((sol1.t[sol1.t <= T1], sol2.t))
        else:
            y_combined = sol1.y
            t_combined = sol1.t

    else:
        sol1 = solve_ivp(
            system2, [0, t_end], y0,
            events=delta_reach_Ts,
            t_eval=t_eval,
            dense_output=True,
            rtol=1e-8,
            atol=1e-10
        )

        if sol1.t_events[0].size > 0:
            T2 = sol1.t_events[0][0]
            y_switch = sol1.sol(T2)
            Delta_T2 = alpha1 * y_switch[1] + alpha2 * y_switch[4]

            t_eval2 = t_eval[t_eval > T2]
            sol2 = solve_ivp(
                system1, [T2, t_end], y_switch,
                t_eval=t_eval2,
                dense_output=True,
                rtol=1e-8,
                atol=1e-10
            )

            y_combined = np.hstack((sol1.y[:, sol1.t <= T2], sol2.y))
            t_combined = np.concatenate((sol1.t[sol1.t <= T2], sol2.t))
        else:
            print("出错了，该情况不可能存在")
            y_combined = sol1.y
            t_combined = sol1.t

    I1_full = y_combined[1]
    I2_full = y_combined[4]
    delta_combined = alpha1 * I1_full + alpha2 * I2_full

    return t_combined, delta_combined, T1, T2, Delta_T1, Delta_T2


fig, axs = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=True)

# ------------------ 左图：Case2 ------------------ #
Ts = 200
t_combined, delta_combined, T1, T2, Delta_T1, Delta_T2 = run_case(Ts)

mask_before_T1 = t_combined <= T1
mask_after_T1 = t_combined > T1

axs[0].plot(
    t_combined[mask_before_T1],
    delta_combined[mask_before_T1],
    color="#377eb8",
        label=r"$\Delta(t): t  < T_1$",
    linewidth=1.5
)
axs[0].plot(
    t_combined[mask_after_T1],
    delta_combined[mask_after_T1],
    color="#e41a1c",
    label=r"$\Delta(t): t > T_1$",
    linewidth=1.5
)
axs[0].axhline(
    Ts,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    label=r"$y = T_s$"
)
axs[0].text(3, 100, "free system", fontsize=TEXT_SIZE)
axs[0].text(35, 50, "control system", fontsize=TEXT_SIZE)
axs[0].plot(T1, Delta_T1, "o", color="black", markersize=4)
axs[0].text(T1, Delta_T1 - 20, r"$\Delta(T_1) = T_s$", fontsize=TEXT_SIZE)
axs[0].vlines(T1, 0, Ts, color="gray", linestyle="--", linewidth=1)
axs[0].annotate(
    r"$t = T_1$",
    xy=(T1 + 4, 0),
    xycoords=("data", "axes fraction"),
    xytext=(0, -6),
    textcoords="offset points",
    ha="center",
    va="top",
    fontsize=TEXT_SIZE
)
axs[0].set_title("(a)", fontsize=TITLE_SIZE, fontweight="normal", pad=4)
axs[0].set_xlabel(r"$t$", fontsize=AXIS_LABEL_SIZE)
axs[0].set_ylabel(r"$\Delta(t)$", rotation=0, fontsize=AXIS_LABEL_SIZE)
axs[0].yaxis.set_label_coords(-0.14, 0.5)
axs[0].legend(loc="upper right", frameon=True)
axs[0].grid(False)
axs[0].set_xlim(0, float(t_combined[-1]))

# ------------------ 右图：Case1 ------------------ #
Ts = 360
t_combined, delta_combined, T1, T2, Delta_T1, Delta_T2 = run_case(Ts)

axs[1].plot(
    t_combined,
    delta_combined,
    color="#377eb8",
    label=r"$\Delta(t)$",
    linewidth=1.5
)
axs[1].axhline(
    Ts,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    label=r"$y = T_s$"
)
axs[1].text(23, 150, "free system", fontsize=TEXT_SIZE)
axs[1].set_ylabel(r"$\Delta(t)$", rotation=0, fontsize=AXIS_LABEL_SIZE)
axs[1].yaxis.set_label_coords(-0.13, 0.5)
axs[1].set_xlabel(r"$t$", fontsize=AXIS_LABEL_SIZE)
axs[1].set_title("(b)", fontsize=TITLE_SIZE, fontweight="normal", pad=4)
axs[1].legend(loc="upper right", frameon=True)
axs[1].grid(False)
axs[1].set_xlim(0, float(t_combined[-1]))

for ax in axs.ravel():
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, width=0.6)
    ax.yaxis.set_label_coords(-0.2, 0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

plt.tight_layout()
plt.subplots_adjust(wspace=0.40)
plt.savefig("twocases.eps", format="eps", bbox_inches="tight")
plt.show()
