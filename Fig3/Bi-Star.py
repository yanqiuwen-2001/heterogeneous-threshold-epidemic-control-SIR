import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from itertools import product

time_range = np.arange(0, 14, 1)

G = 8

columns = [f'N{i}' for i in range(1, G + 1)]
columns += ['Ts']
columns += [f'alpha{i}*' for i in range(1, G + 1)]
columns += ['alpha*', 'center', 'accuracy']
columns += [
    f'gamI{j + 1}-Day{i + 1}'
    for i in time_range
    for j in range(G)
]

accuracy_summary = []


# ============================================================
# center 主对角线元素依次取 6, 7, ..., 15
# ============================================================
for center_diagonal_value in range(6, 16):

    # 每个主对角线取值单独生成一个 CSV 文件
    data = []

    # 记录总尝试次数
    attempt_count = 0

    # 只有成功放入 CSV 的数据达到 1000 条时才停止
    while len(data) < 1000:

        attempt_count += 1
        attention = ''

        # ------------------ 参数设置 ------------------ #
        t_end = 1000
        t_eval = np.arange(0, t_end, 0.01)

        gam = 1 / 7
        beta0 = np.random.uniform(0.02, 0.035)

        # ------------------ 单向 Star ------------------ #
        C = np.zeros((G, G), dtype=float)

        diagonal_value = np.random.randint(1, 6)
        np.fill_diagonal(C, diagonal_value)

        center = np.random.randint(G)

        # 依次取 6, 7, ..., 15
        C[center, center] = center_diagonal_value

        for j in range(G):
            if j != center:
                C[center, j] = np.random.randint(1, 4)
                C[j, center] = np.random.randint(1, 4)

        # ------------------ 人口与初始感染人数 ------------------ #
        N_value = np.random.randint(600000, 1800000)
        N = np.full(G, N_value, dtype=float)

        I0_value = np.random.randint(1, 60)
        I0_old = np.full(G, I0_value, dtype=float)

        R0_old = np.zeros(G)
        S0_old = N - I0_old - R0_old

        # ------------------ 控制参数 ------------------ #
        E_value = np.random.uniform(0.1, 0.5)
        E = np.full((G, G), E_value, dtype=float)
        diag_value_E = np.random.uniform(0.1, 0.99)
        np.fill_diagonal(E, diag_value_E)

        # ------------------ 系统定义 ------------------ #
        def system01(t, y, beta0, gam, C, N):
            dydt = []

            for i in range(G):
                Si = y[i * 3]
                Ii = y[i * 3 + 1]
                Ri = y[i * 3 + 2]

                infection_sum = 0

                for j in range(G):
                    Ij = y[j * 3 + 1]
                    infection_sum += C[i, j] * Ij

                dSi_dt = (
                    -beta0
                    * infection_sum
                    * Si
                    / N[i]
                )

                dIi_dt = (
                    beta0
                    * infection_sum
                    * Si
                    / N[i]
                    - gam * Ii
                )

                dRi_dt = gam * Ii

                dydt.extend([
                    dSi_dt,
                    dIi_dt,
                    dRi_dt
                ])

            return dydt


        def system02(t, y, beta0, gam, C, E, N):
            dydt = []

            for i in range(G):
                Si = y[i * 3]
                Ii = y[i * 3 + 1]
                Ri = y[i * 3 + 2]

                control_sum = 0

                for j in range(G):
                    Ij = y[j * 3 + 1]
                    control_sum += (
                        C[i, j]
                        * E[i, j]
                        * Ij
                    )

                dSi_dt = (
                    -beta0
                    * control_sum
                    * Si
                    / N[i]
                )

                dIi_dt = (
                    beta0
                    * control_sum
                    * Si
                    / N[i]
                    - gam * Ii
                )

                dRi_dt = gam * Ii

                dydt.extend([
                    dSi_dt,
                    dIi_dt,
                    dRi_dt
                ])

            return dydt

        # ------------------ 前 14 个观测点 free ------------------ #
        t_span_old = (0, 14)

        y0_old = np.zeros(3 * G)

        for i in range(G):
            y0_old[3 * i:3 * i + 3] = [
                S0_old[i],
                I0_old[i],
                R0_old[i]
            ]

        sol_old = solve_ivp(
            system01,
            t_span_old,
            y0_old,
            args=(beta0, gam, C, N),
            t_eval=time_range,
            rtol=1e-8,
            atol=1e-10
        )

        if not sol_old.success:
            print('前 14 个观测点求解失败')
            continue

        I_values = []
        I_gam = []
        S_values = []

        for k in range(len(time_range)):

            Ik = np.array([
                sol_old.y[3 * j + 1, k]
                for j in range(G)
            ])

            Sk = np.array([
                sol_old.y[3 * j, k]
                for j in range(G)
            ])

            I_values.append(tuple(Ik))
            I_gam.append(tuple(gam * Ik))
            S_values.append(tuple(Sk))

        # 最后一个观测点 t = 13 的状态作为后续系统初值
        S0 = np.array([
            sol_old.y[3 * i, -1]
            for i in range(G)
        ])

        I0 = np.array([
            sol_old.y[3 * i + 1, -1]
            for i in range(G)
        ])

        R0 = np.array([
            sol_old.y[3 * i + 2, -1]
            for i in range(G)
        ])

        # ------------------ 检查初始条件：都上升 ------------------ #
        inc_free = np.zeros(G)
        inc_ctrl = np.zeros(G)

        for i in range(G):

            inf_sum = 0.0
            ctrl_sum = 0.0

            for j in range(G):

                inf_sum += (
                    C[i, j]
                    * I0[j]
                )

                ctrl_sum += (
                    C[i, j]
                    * E[i, j]
                    * I0[j]
                )

            inc_free[i] = (
                beta0
                * inf_sum
                * S0[i]
                / N[i]
                - gam * I0[i]
            )

            inc_ctrl[i] = (
                beta0
                * ctrl_sum
                * S0[i]
                / N[i]
                - gam * I0[i]
            )

        if np.any(inc_free <= 0):
            print('不满足初始时刻都上升')
            continue

        # ------------------ 求解后续 free 系统 ------------------ #
        initial = np.zeros(3 * G)

        for i in range(G):
            initial[3 * i:3 * i + 3] = [
                S0[i],
                I0[i],
                R0[i]
            ]

        solution = solve_ivp(
            system01,
            [0, t_end],
            initial,
            t_eval=t_eval,
            args=(beta0, gam, C, N),
            rtol=1e-8,
            atol=1e-10
        )

        if not solution.success:
            print('后续系统求解失败')
            continue

        # ------------------ 截取有效时间区间 ------------------ #
        eps = 0.1
        t_max_idx = len(t_eval) - 1

        for idx in range(len(t_eval)):

            I_all = np.array([
                solution.y[3 * j + 1, idx]
                for j in range(G)
            ])

            if np.all(I_all < eps):
                t_max_idx = idx
                break

        t_eval_f = t_eval[:t_max_idx + 1]

        I_traj = np.vstack([
            solution.y[
                3 * j + 1,
                :t_max_idx + 1
            ]
            for j in range(G)
        ])

        # ------------------ 求各群体首个峰值 ------------------ #
        dI_dt = np.gradient(
            I_traj,
            t_eval_f,
            axis=1
        )

        peaks = []

        for i in range(G):

            di = dI_dt[i]

            idxs = np.where(
                (di[:-1] > 0)
                & (di[1:] < 0)
            )[0]

            if len(idxs) > 0:
                peaks.append(
                    I_traj[i, idxs[0]]
                )
            else:
                peaks.append(np.nan)

        peaks = np.array(peaks)

        if np.all(np.isnan(peaks)):
            print('没有检测到峰值')
            continue

        min_I0 = float(I0.min())
        max_I0 = float(I0.max())
        max_peaks = float(np.nanmax(peaks))

        Tsmax = min(
            2100,
            max_peaks
        )

        if max_I0 >= Tsmax:
            print('max_I0 >= Tsmax')
            continue

        # ------------------ 生成 Ts ------------------ #
        Ts = np.random.uniform(
            max_I0,
            Tsmax
        )

        # ------------------ alpha 扫描 ------------------ #
        # 仅扫描单位向量
        alphas = np.eye(G)

        def delta_of_y(y):
            Ivec = np.array([
                y[3 * j + 1]
                for j in range(G)
            ])

            return Ivec

        T1_list = []
        T2_list = []
        alpha_rows = []

        for a in alphas:

            def event_reach_Ts(t, y, *args):

                Ivec = delta_of_y(y)

                return float(
                    a @ Ivec
                    - Ts
                )

            event_reach_Ts.terminal = True
            event_reach_Ts.direction = 0

            delta0 = float(
                a @ I0
            )

            T1 = None
            T2 = None

            if delta0 <= Ts:

                sol1 = solve_ivp(
                    system01,
                    [0, t_end],
                    initial,
                    events=event_reach_Ts,
                    t_eval=t_eval,
                    dense_output=True,
                    args=(beta0, gam, C, N),
                    rtol=1e-8,
                    atol=1e-10
                )

                if sol1.t_events[0].size > 0:
                    T1 = float(
                        sol1.t_events[0][0]
                    )

                else:
                    attention = 'Case1'
                    T1 = 1e10

            else:

                sol2 = solve_ivp(
                    system02,
                    [0, t_end],
                    initial,
                    events=event_reach_Ts,
                    t_eval=t_eval,
                    dense_output=True,
                    args=(beta0, gam, C, E, N),
                    rtol=1e-8,
                    atol=1e-10
                )

                if sol2.t_events[0].size > 0:
                    T2 = float(
                        sol2.t_events[0][0]
                    )

                else:
                    T2 = -1
                    attention = '删掉该条'

            T1_list.append(T1)
            T2_list.append(T2)
            alpha_rows.append(a)

        # ------------------ 取最小的 T1 ------------------ #
        T1_arr = np.array(
            T1_list,
            dtype=float
        )

        min_idx = int(
            np.nanargmin(T1_arr)
        )

        alpha_star = alpha_rows[min_idx]

        # ------------------ 写入一条有效数据 ------------------ #
        row = list(N)

        row += [Ts]

        row += list(
            alpha_star
        )

        alpha_star_index = (
            int(np.argmax(alpha_star))
            + 1
        )

        row += [
            alpha_star_index
        ]

        row += [
            center + 1
        ]

        accuracy = (
            'Yes'
            if alpha_star_index == center + 1
            else 'No'
        )

        row += [
            accuracy
        ]

        I_gam_np = np.array(
            I_gam
        )

        row += (
            I_gam_np
            .flatten()
            .tolist()
        )

        data.append(row)

    df = pd.DataFrame(
        data,
        columns=columns
    )

    file_name = (
        f'Bi-Star_{center_diagonal_value}.csv'
    )

    df.to_csv(
        file_name,
        index=False
    )

    # ------------------ 计算 accuracy == Yes 的比例 ------------------ #
    yes_count = int(
        (df['accuracy'] == 'Yes')
        .sum()
    )

    yes_ratio = (
        yes_count
        / len(df)
    )

    accuracy_summary.append({
        'center_diagonal_value': center_diagonal_value,
        'rows': len(df),
        'attempt_count': attempt_count,
        'yes_count': yes_count,
        'yes_ratio': yes_ratio
    })


# ============================================================
# 保存并输出汇总结果
# ============================================================
summary_df = pd.DataFrame(
    accuracy_summary
)

summary_df.to_csv(
    'Bi-Star_summary.csv',
    index=False
)
