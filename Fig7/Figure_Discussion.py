import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap



FIG_WIDTH = 174 / 15.4          # 174 mm, full text width
FIG_HEIGHT = 95 / 15.4          # 自定义

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
YLABEL_PAD = 8


def set_subfigure_title(ax, label, pad=6):
    ax.set_title(
        label,
        fontsize=TITLE_SIZE,
        fontweight="normal",
        loc="center",
        pad=pad
    )


# ============================================================
# 1. 两组参数
# ============================================================
PARAMS_FIXED_14 = {
    "name": "fixed_14_days",
    "e": (0.85929499, 0.41595848, 0.34196803, 0.54249403),
    "beta0": 0.029184736943291597,
    "gam": (1 / 7, 1 / 7),
    "c": (4.0, 3.0, 2.0, 10.0),
    "N": (1551475.0, 637580.0),
    "R0": (0.0, 0.0),
    "I0": (150.0, 200.0),
    "Ts": 105592.90065792958,
    "control_duration": 14.0,
    "t_plot_end": 129.83905956462215,
}

PARAMS_THRESHOLD = {
    "name": "threshold_down",
    "e": (0.94698653, 0.45346735, 0.45849559, 0.92936917),
    "beta0": 0.03477090364534523,
    "gam": (1 / 7, 1 / 7),
    "c": (10.0, 3.0, 2.0, 4.0),
    "N": (1072108.0, 1238980.0),
    "R0": (0.0, 0.0),
    "I0": (150.0, 200.0),
    "Ts": 45050.289414917075,
    "t_plot_end": 103.27458915238516,
}

t_final_max = 2000
infection_tol = 1e-5
r_change_tol = 1e-4
final_check_window = 100


# ============================================================
# 2. 建立模型
# ============================================================
def make_model(params):
    e11, e21, e12, e22 = params["e"]
    beta0 = params["beta0"]
    gam1, gam2 = params["gam"]
    c11, c21, c12, c22 = params["c"]
    N1, N2 = params["N"]
    R10, R20 = params["R0"]
    I10, I20 = params["I0"]
    S10 = N1 - R10 - I10
    S20 = N2 - R20 - I20

    def system_free(t, y):
        S1, I1, R1, S2, I2, R2 = y

        dS1_dt = -beta0 * (c11 * I1 + c21 * I2) * S1 / N1
        dI1_dt = beta0 * (c11 * I1 + c21 * I2) * S1 / N1 - gam1 * I1
        dR1_dt = gam1 * I1

        dS2_dt = -beta0 * (c12 * I1 + c22 * I2) * S2 / N2
        dI2_dt = beta0 * (c12 * I1 + c22 * I2) * S2 / N2 - gam2 * I2
        dR2_dt = gam2 * I2

        return [dS1_dt, dI1_dt, dR1_dt, dS2_dt, dI2_dt, dR2_dt]

    def system_control(t, y):
        S1, I1, R1, S2, I2, R2 = y

        dS1_dt = -beta0 * (e11 * c11 * I1 + e21 * c21 * I2) * S1 / N1
        dI1_dt = beta0 * (e11 * c11 * I1 + e21 * c21 * I2) * S1 / N1 - gam1 * I1
        dR1_dt = gam1 * I1

        dS2_dt = -beta0 * (e12 * c12 * I1 + e22 * c22 * I2) * S2 / N2
        dI2_dt = beta0 * (e12 * c12 * I1 + e22 * c22 * I2) * S2 / N2 - gam2 * I2
        dR2_dt = gam2 * I2

        return [dS1_dt, dI1_dt, dR1_dt, dS2_dt, dI2_dt, dR2_dt]

    initial_condition1f = beta0 * (c11 * I10 + c21 * I20) * S10 / N1 - gam1 * I10
    initial_condition2f = beta0 * (c12 * I10 + c22 * I20) * S20 / N2 - gam2 * I20
    initial_condition1c = beta0 * (e11 * c11 * I10 + e21 * c21 * I20) * S10 / N1 - gam1 * I10
    initial_condition2c = beta0 * (e12 * c12 * I10 + e22 * c22 * I20) * S20 / N2 - gam2 * I20

    print(f"\nInitial growth conditions: {params['name']}")
    print(f"initial_condition1f = {initial_condition1f:.6f}")
    print(f"initial_condition2f = {initial_condition2f:.6f}")
    print(f"initial_condition1c = {initial_condition1c:.6f}")
    print(f"initial_condition2c = {initial_condition2c:.6f}")

    if not (
        initial_condition1f > 0
        and initial_condition2f > 0
        and initial_condition1c > 0
        and initial_condition2c > 0
    ):
        raise ValueError(f"{params['name']} 不满足四个初始增长条件都 > 0。")

    return {
        "system_free": system_free,
        "system_control": system_control,
        "y0": np.array([S10, I10, R10, S20, I20, R20], dtype=float),
        "I10": I10,
        "I20": I20,
    }


# ============================================================
# 3. 工具函数
# ============================================================
def append_solution(t_list, y_list, phase_list, sol, phase_name, t_plot_end=None):
    start_idx = 0 if len(t_list) == 0 else 1
    t_part = sol.t[start_idx:]
    y_part = sol.y.T[start_idx:]

    if t_plot_end is not None:
        keep = t_part <= t_plot_end
        t_part = t_part[keep]
        y_part = y_part[keep]

    t_list.extend(t_part.tolist())
    y_list.extend(y_part.tolist())
    phase_list.extend([phase_name] * len(t_part))


def final_size_after_free(system_free, t_start, y_start):
    t0 = float(t_start)
    y0 = np.array(y_start, dtype=float)

    while t0 < t_final_max:
        t1 = min(t0 + final_check_window, t_final_max)
        sol = solve_ivp(
            system_free,
            [t0, t1],
            y0,
            rtol=1e-8,
            atol=1e-10,
            max_step=0.2,
        )

        y1 = sol.y[:, -1]
        i_total = y1[1] + y1[4]
        r_change = abs((y1[2] + y1[5]) - (y0[2] + y0[5]))

        if i_total < infection_tol and r_change < r_change_tol:
            return y1[2] + y1[5], t1, i_total, r_change

        t0 = t1
        y0 = y1

    return y0[2] + y0[5], t_final_max, y0[1] + y0[4], np.nan


# ============================================================
# 4. 策略一：free -> controlled system (14 days) -> free
# ============================================================
def simulate_fixed_14(params, alpha1, return_trajectory=False):
    model = make_model(params) if "_model" not in params else params["_model"]
    system_free = model["system_free"]
    system_control = model["system_control"]
    y0 = model["y0"]
    I10 = model["I10"]
    I20 = model["I20"]

    alpha2 = 1.0 - alpha1
    Ts = params["Ts"]
    t_plot_end = params["t_plot_end"]
    control_duration = params["control_duration"]

    def delta_from_y(y):
        return alpha1 * y[1] + alpha2 * y[4]

    def event_reach_Ts(t, y):
        return delta_from_y(y) - Ts

    event_reach_Ts.terminal = True
    event_reach_Ts.direction = 1

    t_all = []
    y_all = []
    phase_all = []

    if alpha1 * I10 + alpha2 * I20 < Ts:
        sol_free_1 = solve_ivp(
            system_free,
            [0, t_plot_end],
            y0,
            events=event_reach_Ts,
            rtol=1e-8,
            atol=1e-10,
            max_step=0.05,
        )

        if return_trajectory:
            append_solution(t_all, y_all, phase_all, sol_free_1, "free_1")

        if sol_free_1.t_events[0].size == 0:
            y_policy_end = sol_free_1.y[:, -1]
            t_policy_end = sol_free_1.t[-1]
            final_size, t_final, i_final, r_change_final = final_size_after_free(
                system_free,
                t_policy_end,
                y_policy_end,
            )
            result = {
                "alpha1": alpha1,
                "alpha2": alpha2,
                "T_switch": np.nan,
                "T_control_end": np.nan,
                "final_size": final_size,
                "t_final": t_final,
                "I_total_final": i_final,
                "R_change_final_window": r_change_final,
                "status": "free_only",
            }
            return add_trajectory(result, alpha1, alpha2, t_all, y_all, phase_all, return_trajectory)

        T_switch = sol_free_1.t_events[0][0]
        y_switch = sol_free_1.y_events[0][0]
    else:
        T_switch = 0.0
        y_switch = y0.copy()
        if return_trajectory:
            t_all.append(0.0)
            y_all.append(y0.tolist())
            phase_all.append("control")

    T_control_end = T_switch + control_duration
    sol_control = solve_ivp(
        system_control,
        [T_switch, T_control_end],
        y_switch,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.05,
    )

    if return_trajectory:
        append_solution(t_all, y_all, phase_all, sol_control, "control")

    y_after_control = sol_control.y[:, -1]

    if T_control_end < t_plot_end:
        sol_free_2_plot = solve_ivp(
            system_free,
            [T_control_end, t_plot_end],
            y_after_control,
            rtol=1e-8,
            atol=1e-10,
            max_step=0.05,
        )

        if return_trajectory:
            append_solution(t_all, y_all, phase_all, sol_free_2_plot, "free_2")

    final_size, t_final, i_final, r_change_final = final_size_after_free(
        system_free,
        T_control_end,
        y_after_control,
    )

    result = {
        "alpha1": alpha1,
        "alpha2": alpha2,
        "T_switch": T_switch,
        "T_control_end": T_control_end,
        "final_size": final_size,
        "t_final": t_final,
        "I_total_final": i_final,
        "R_change_final_window": r_change_final,
        "status": "free_control_free",
    }
    return add_trajectory(result, alpha1, alpha2, t_all, y_all, phase_all, return_trajectory)


# ============================================================
# 5. 策略二：free -> control，到 delta 下降到 Ts -> free
# ============================================================
def simulate_threshold_down(params, alpha1, return_trajectory=False):
    model = make_model(params) if "_model" not in params else params["_model"]
    system_free = model["system_free"]
    system_control = model["system_control"]
    y0 = model["y0"]
    I10 = model["I10"]
    I20 = model["I20"]

    alpha2 = 1.0 - alpha1
    Ts = params["Ts"]
    t_plot_end = params["t_plot_end"]

    def delta_from_y(y):
        return alpha1 * y[1] + alpha2 * y[4]

    def event_up_to_Ts(t, y):
        return delta_from_y(y) - Ts

    event_up_to_Ts.terminal = True
    event_up_to_Ts.direction = 1

    def event_down_to_Ts(t, y):
        return delta_from_y(y) - Ts

    event_down_to_Ts.terminal = True
    event_down_to_Ts.direction = -1

    t_all = []
    y_all = []
    phase_all = []

    if alpha1 * I10 + alpha2 * I20 < Ts:
        sol_free_1 = solve_ivp(
            system_free,
            [0, t_final_max],
            y0,
            events=event_up_to_Ts,
            rtol=1e-8,
            atol=1e-10,
            max_step=0.05,
        )

        if return_trajectory:
            append_solution(t_all, y_all, phase_all, sol_free_1, "free_1", t_plot_end=t_plot_end)

        if sol_free_1.t_events[0].size == 0:
            final_size, t_final, i_final, r_change_final = final_size_after_free(
                system_free,
                sol_free_1.t[-1],
                sol_free_1.y[:, -1],
            )
            result = {
                "alpha1": alpha1,
                "alpha2": alpha2,
                "T_switch_up": np.nan,
                "T_switch_down": np.nan,
                "final_size": final_size,
                "t_final": t_final,
                "I_total_final": i_final,
                "R_change_final_window": r_change_final,
                "status": "free_only",
            }
            return add_trajectory(result, alpha1, alpha2, t_all, y_all, phase_all, return_trajectory)

        T_switch_up = sol_free_1.t_events[0][0]
        y_switch_up = sol_free_1.y_events[0][0]
    else:
        T_switch_up = 0.0
        y_switch_up = y0.copy()
        if return_trajectory and T_switch_up <= t_plot_end:
            t_all.append(0.0)
            y_all.append(y0.tolist())
            phase_all.append("control")

    eps = 1e-4
    t_control_start = T_switch_up
    y_control_start = y_switch_up.copy()

    if T_switch_up + eps < t_final_max:
        sol_eps = solve_ivp(
            system_control,
            [T_switch_up, T_switch_up + eps],
            y_switch_up,
            rtol=1e-8,
            atol=1e-10,
            max_step=eps,
        )

        if return_trajectory:
            append_solution(t_all, y_all, phase_all, sol_eps, "control", t_plot_end=t_plot_end)

        t_control_start = T_switch_up + eps
        y_control_start = sol_eps.y[:, -1]

    sol_control = solve_ivp(
        system_control,
        [t_control_start, t_final_max],
        y_control_start,
        events=event_down_to_Ts,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.05,
    )

    if return_trajectory:
        append_solution(t_all, y_all, phase_all, sol_control, "control", t_plot_end=t_plot_end)

    if sol_control.t_events[0].size == 0:
        y_policy_end = sol_control.y[:, -1]
        final_size, t_final, i_final, r_change_final = final_size_after_free(
            system_free,
            sol_control.t[-1],
            y_policy_end,
        )
        result = {
            "alpha1": alpha1,
            "alpha2": alpha2,
            "T_switch_up": T_switch_up,
            "T_switch_down": np.nan,
            "final_size": final_size,
            "t_final": t_final,
            "I_total_final": i_final,
            "R_change_final_window": r_change_final,
            "status": "free_control_to_end",
        }
        return add_trajectory(result, alpha1, alpha2, t_all, y_all, phase_all, return_trajectory)

    T_switch_down = sol_control.t_events[0][0]
    y_switch_down = sol_control.y_events[0][0]

    if T_switch_down < t_plot_end:
        sol_free_2_plot = solve_ivp(
            system_free,
            [T_switch_down, t_plot_end],
            y_switch_down,
            rtol=1e-8,
            atol=1e-10,
            max_step=0.05,
        )

        if return_trajectory:
            append_solution(t_all, y_all, phase_all, sol_free_2_plot, "free_2", t_plot_end=t_plot_end)

    final_size, t_final, i_final, r_change_final = final_size_after_free(
        system_free,
        T_switch_down,
        y_switch_down,
    )

    result = {
        "alpha1": alpha1,
        "alpha2": alpha2,
        "T_switch_up": T_switch_up,
        "T_switch_down": T_switch_down,
        "final_size": final_size,
        "t_final": t_final,
        "I_total_final": i_final,
        "R_change_final_window": r_change_final,
        "status": "free_control_free",
    }
    return add_trajectory(result, alpha1, alpha2, t_all, y_all, phase_all, return_trajectory)


def add_trajectory(result, alpha1, alpha2, t_all, y_all, phase_all, return_trajectory):
    if return_trajectory:
        y_arr = np.array(y_all)
        result["t"] = np.array(t_all)
        result["y"] = y_arr
        result["phase"] = np.array(phase_all)
        if len(y_arr) == 0:
            result["delta"] = np.array([])
        else:
            result["delta"] = alpha1 * y_arr[:, 1] + alpha2 * y_arr[:, 4]
    return result


# ============================================================
# 6. 计算数据
# ============================================================
def compute_dataset(params, simulate_func):
    params["_model"] = make_model(params)

    alpha_curve_values = np.round(np.arange(0, 1.01, 0.1), 1)
    curve_results = [
        simulate_func(params, alpha1, return_trajectory=True)
        for alpha1 in alpha_curve_values
    ]

    alpha1_range = np.linspace(0, 1, 51)
    results = []
    finalsize_list = []

    for alpha1 in alpha1_range:
        res = simulate_func(params, alpha1, return_trajectory=False)
        results.append(res)
        finalsize_list.append(res["final_size"])

    finalsize_list = np.array(finalsize_list, dtype=float)
    min_idx = int(np.argmin(finalsize_list))

    print(f"\nMinimum final size: {params['name']}")
    print(f"alpha1 = {alpha1_range[min_idx]:.2f}")
    print(f"final_size = {finalsize_list[min_idx]:.6f}")

    return {
        "params": params,
        "curve_results": curve_results,
        "alpha1_range": alpha1_range,
        "results": results,
        "finalsize_list": finalsize_list,
        "min_idx": min_idx,
    }


data_fixed_14 = compute_dataset(PARAMS_FIXED_14, simulate_fixed_14)
data_threshold = compute_dataset(PARAMS_THRESHOLD, simulate_threshold_down)


# ============================================================
# 7. 作图函数
# ============================================================
norm = Normalize(vmin=0, vmax=1)
blue_cmap = LinearSegmentedColormap.from_list(
    "custom_blues",
    plt.cm.Blues(np.linspace(0.1, 0.9, 256)),
)
red_cmap = LinearSegmentedColormap.from_list(
    "custom_reds",
    plt.cm.Reds(np.linspace(0.1, 0.9, 256)),
)


def format_axis(ax):
    ax.tick_params(
        axis="both",
        labelsize=TICK_SIZE,
        width=TICK_WIDTH,
        length=3,
        direction="out"
    )

    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINEWIDTH)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_delta_panel(ax, cax_free, cax_control, data, text_config, ts_xytext=(-5, 0), ts_ha="right"):
    params = data["params"]
    Ts = params["Ts"]

    for res in data["curve_results"]:
        alpha1 = res["alpha1"]
        t = res["t"]
        delta = res["delta"]
        phase = res["phase"]

        for phase_name in ["free_1", "control", "free_2"]:
            mask = phase == phase_name
            if not np.any(mask):
                continue

            if phase_name == "control":
                color = red_cmap(norm(alpha1))
                linewidth = 1.0
                zorder = 1
            else:
                color = blue_cmap(norm(alpha1))
                linewidth = 1.0
                zorder = 2

            ax.plot(
                t[mask],
                delta[mask],
                linewidth=linewidth,
                color=color,
                zorder=zorder,
            )

    ax.axhline(Ts, color="black", linestyle="--", linewidth=0.8, zorder=0)
    ax.annotate(
        r"$T_s$",
        xy=(0, Ts),
        xycoords=("axes fraction", "data"),
        xytext=ts_xytext,
        textcoords="offset points",
        ha=ts_ha,
        va="center",
        fontsize=TEXT_SIZE,
        annotation_clip=False,
    )

    label_res = min(data["curve_results"], key=lambda item: abs(item["alpha1"] - 0.5))
    t_label = label_res["t"]
    delta_label = label_res["delta"]
    phase_label = label_res["phase"]

    def phase_midpoint_text(phase_name, text, color, x_shift=0, y_shift=0):
        mask = phase_label == phase_name
        if not np.any(mask):
            return

        tt = t_label[mask]
        yy = delta_label[mask]
        idx = len(tt) // 2

        ax.text(
            tt[idx] + x_shift,
            yy[idx] + y_shift,
            text,
            ha="center",
            va="center",
            fontsize=TEXT_SIZE,
            color=color,
            zorder=5,
        )

    for item in text_config:
        phase_midpoint_text(**item)

    ax.set_xlim(0, params["t_plot_end"])
    ax.set_xlabel("Time", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$\alpha_1 I_1 + \alpha_2 I_2$", fontsize=LABEL_SIZE,labelpad=YLABEL_PAD)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=False)
    ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)

    format_axis(ax)

    sm_free = ScalarMappable(norm=norm, cmap=blue_cmap)
    sm_free.set_array([])
    cbar_free = fig.colorbar(sm_free, cax=cax_free)
    cbar_free.ax.tick_params(labelsize=TICK_SIZE, width=TICK_WIDTH)
    cbar_free.ax.set_xlabel(r"$\alpha_1$", fontsize=LEGEND_SIZE, labelpad=4)
    for spine in cbar_free.ax.spines.values():
        spine.set_linewidth(AXIS_LINEWIDTH)

    sm_control = ScalarMappable(norm=norm, cmap=red_cmap)
    sm_control.set_array([])
    cbar_control = fig.colorbar(sm_control, cax=cax_control)
    cbar_control.ax.tick_params(labelsize=TICK_SIZE, width=TICK_WIDTH)
    cbar_control.ax.set_xlabel(r"$\alpha_1$", fontsize=LEGEND_SIZE, labelpad=4)
    for spine in cbar_control.ax.spines.values():
        spine.set_linewidth(AXIS_LINEWIDTH)


def plot_finalsize_panel(ax, data):
    alpha1_range = data["alpha1_range"]
    finalsize_list = data["finalsize_list"]
    min_idx = data["min_idx"]

    ax.scatter(
        alpha1_range,
        finalsize_list,
        s=22,
        facecolors="none",
        edgecolors="#2354a1",
        linewidths=0.7,
        zorder=3,
    )
    ax.plot(
        alpha1_range,
        finalsize_list,
        color="#2354a1",
        linewidth=0.8,
        zorder=2,
    )
    ax.scatter(
        alpha1_range[min_idx],
        finalsize_list[min_idx],
        s=28,
        color="#d62728",
        zorder=4,
    )

    ax.set_xlabel(r"$\alpha_1$", fontsize=LABEL_SIZE)
    ax.set_ylabel("Final Size", fontsize=LABEL_SIZE,labelpad=YLABEL_PAD)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=False)
    ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)

    format_axis(ax)


# ============================================================
# 8. 四个子图：第一行 ab，第二行 cd
# ============================================================
fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

gs = fig.add_gridspec(
    nrows=2,
    ncols=5,
    width_ratios=[1.30, 0.035, 0.035, 0.12, 1.00],
    height_ratios=[1, 1],
    wspace=0.28,
    hspace=0.45,
)

ax_a = fig.add_subplot(gs[0, 0])
cax_a_free = fig.add_subplot(gs[0, 1])
cax_a_control = fig.add_subplot(gs[0, 2])
ax_b = fig.add_subplot(gs[0, 4])

ax_c = fig.add_subplot(gs[1, 0])
cax_c_free = fig.add_subplot(gs[1, 1])
cax_c_control = fig.add_subplot(gs[1, 2])
ax_d = fig.add_subplot(gs[1, 4])

plot_delta_panel(
    ax_a,
    cax_a_free,
    cax_a_control,
    data_fixed_14,
    [
        {
            "phase_name": "free_1",
            "text": "free system",
            "color": "0.20",
            "x_shift": 25,
            "y_shift": 0.04 * PARAMS_FIXED_14["Ts"],
        },
        {
            "phase_name": "control",
            "text": "control system\n(14 days)",
            "color": "0.20",
            "x_shift": 24,
            "y_shift": 0.035 * PARAMS_FIXED_14["Ts"],
        },
        {
            "phase_name": "free_2",
            "text": "free system",
            "color": "0.20",
            "x_shift": 26,
            "y_shift": 0.08 * PARAMS_FIXED_14["Ts"],
        },
    ],
    ts_xytext=(-15, 0),
    ts_ha="left"
)


plot_finalsize_panel(ax_b, data_fixed_14)

plot_delta_panel(
    ax_c,
    cax_c_free,
    cax_c_control,
    data_threshold,
    [
        {
            "phase_name": "free_1",
            "text": "free system",
            "color": "0.20",
            "x_shift": 25,
            "y_shift": 0.3 * PARAMS_THRESHOLD["Ts"],
        },
        {
            "phase_name": "control",
            "text": "control system",
            "color": "0.20",
            "x_shift": 22,
            "y_shift": 0.035 * PARAMS_THRESHOLD["Ts"],
        },
        {
            "phase_name": "free_2",
            "text": "free system",
            "color": "0.20",
            "x_shift": 10,
            "y_shift": 0.25 * PARAMS_THRESHOLD["Ts"],
        },
    ],
    ts_xytext=(-15, 0),
        ts_ha="left"
)


plot_finalsize_panel(ax_d, data_threshold)

set_subfigure_title(ax_a, "(a)")
set_subfigure_title(ax_b, "(b)")
set_subfigure_title(ax_c, "(c)")
set_subfigure_title(ax_d, "(d)")


fig.subplots_adjust(
    left=0.07,
    right=0.985,
    bottom=0.10,
    top=0.90
)

OUT_EPS = "Discussion_abcd.eps"
plt.savefig(OUT_EPS, format="eps", bbox_inches="tight", pad_inches=0.03)

plt.show()
