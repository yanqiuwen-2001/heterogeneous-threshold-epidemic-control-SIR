import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from itertools import product

time_range = np.arange(0, 14, 1)
data = []
# 设置纬度
G =2

columns = ['beta0', 'gam']

columns += [f'N{i}' for i in range(1, G + 1)]

columns += [f'S0_old_{i}' for i in range(1, G + 1)]
columns += [f'I0_old_{i}' for i in range(1, G + 1)]
columns += [f'R0_old_{i}' for i in range(1, G + 1)]

columns += [f'S14_{i}' for i in range(1, G + 1)]
columns += [f'I14_{i}' for i in range(1, G + 1)]
columns += [f'R14_{i}' for i in range(1, G + 1)]

columns += [f'C{i+1}{j+1}' for i in range(G) for j in range(G)]
columns += [f'E{i+1}{j+1}' for i in range(G) for j in range(G)]

columns += ['Ts', 'Tsmax', 'max_I0', 'max_peaks']

columns += [f'peak{i}' for i in range(1, G + 1)]
columns += [f'alpha{i}*' for i in range(1, G + 1)]

columns += ['出错']

columns += [f'gamI{j+1}-Day{i+1}' for i in time_range for j in range(G)]


for _ in range(60):
    attention = ''
    # ------------------ 参数设置 ------------------ #
    t_end = 1000
    t_eval = np.arange(0, t_end, 0.01)
    gam = 1/7
    beta0 = np.random.uniform(0.02,0.035)
    C = np.zeros((G, G), dtype=float)
    np.fill_diagonal(C, np.random.randint(1, 11, size=G)) ##主对角线1～10
    for j in range(G):
        col_sum = np.random.randint(1, 4)
        rows = [i for i in range(G) if i != j]  # 该列的非对角线行索引（长度 G-1）
        vals = np.random.multinomial(col_sum, [1.0 / (G - 1)] * (G - 1))
        C[rows, j] = vals
    E = np.random.uniform(0.1, 0.5, size=(G, G)).astype(float)
    diag_vals_E = np.random.uniform(0.1, 0.99, size=G)
    np.fill_diagonal(E, diag_vals_E)
    N = np.random.randint(600000, 1800000, size=G).astype(float)
    I0_old = np.random.randint(1, 60, size=G).astype(float)
    R0_old = np.zeros(G)
    S0_old = N - I0_old - R0_old


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
                infection_sum += C[i,j] * Ij
            dSi_dt = -beta0 * infection_sum * Si / N[i]
            dIi_dt = beta0 * infection_sum * Si / N[i] - gam * Ii
            dRi_dt = gam * Ii
            dydt.extend([dSi_dt, dIi_dt, dRi_dt])
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
                control_sum += C[i, j] * E[i, j] * Ij
            dSi_dt = -beta0 * control_sum * Si / N[i]
            dIi_dt = beta0 * control_sum * Si / N[i] - gam * Ii
            dRi_dt = gam * Ii
            dydt.extend([dSi_dt, dIi_dt, dRi_dt])
        return dydt

    def system1(t, y):
        dydt = []
        for i in range(G):
            Si = y[i * 3]
            Ii = y[i * 3 + 1]
            Ri = y[i * 3 + 2]
            infection_sum = 0
            for j in range(G):
                Ij = y[j * 3 + 1]
                infection_sum += C[i, j] * Ij
            dSi_dt = -beta0 * infection_sum * Si / N[i]
            dIi_dt = beta0 * infection_sum * Si / N[i] - gam * Ii
            dRi_dt = gam * Ii
            dydt.extend([dSi_dt, dIi_dt, dRi_dt])
        return dydt

    def system2(t, y):
        dydt = []
        for i in range(G):
            Si = y[i * 3]
            Ii = y[i * 3 + 1]
            Ri = y[i * 3 + 2]
            control_sum = 0
            for j in range(G):
                Ij = y[j * 3 + 1]
                control_sum += C[i, j] * E[i, j] * Ij
            dSi_dt = -beta0 * control_sum * Si / N[i]
            dIi_dt = beta0 * control_sum * Si / N[i] - gam * Ii
            dRi_dt = gam * Ii
            dydt.extend([dSi_dt, dIi_dt, dRi_dt])
        return dydt

    ######前14天free
    t_span_old = (0, 14)
    y0_old = np.zeros(3 * G)
    for i in range(G):
        y0_old[3 * i:3 * i + 3] = [S0_old[i], I0_old[i], R0_old[i]]
    sol_old = solve_ivp(system01,t_span_old,y0_old,args=(beta0, gam, C, N),t_eval=time_range,rtol=1e-8, atol=1e-10)

    I_values = []
    I_gam = []
    S_values = []
    R0_values = []
    EN_values = []
    for k in range(len(time_range)):
        Ik = np.array([sol_old.y[3 * j + 1, k] for j in range(G)])
        Sk = np.array([sol_old.y[3 * j, k] for j in range(G)])

        I_values.append(tuple(Ik))
        I_gam.append(tuple(gam*Ik))
        S_values.append(tuple(Sk))


    #14天末状态作为后续系统初值
    S0 = np.array([sol_old.y[3 * i, -1] for i in range(G)])
    I0 = np.array([sol_old.y[3 * i + 1, -1] for i in range(G)])
    R0 = np.array([sol_old.y[3 * i + 2, -1] for i in range(G)])



    #####--------初始条件：都增------------------------
    inc_free = np.zeros(G)
    inc_ctrl = np.zeros(G)
    for i in range(G):
        inf_sum = 0.0
        ctrl_sum = 0.0
        for j in range(G):
            inf_sum += C[i, j] * I0[j]
            ctrl_sum += C[i, j] * E[i, j] * I0[j]
        inc_free[i] = beta0 * inf_sum * S0[i] / N[i] - gam * I0[i]
        inc_ctrl[i] = beta0 * ctrl_sum * S0[i] / N[i] - gam * I0[i]

    if np.any(inc_free <= 0):
        print('不满足初始时刻都上升')
        continue

    else:
        initial = np.zeros(3 * G)
        for i in range(G):
            initial[3 * i:3 * i + 3] = [S0[i], I0[i], R0[i]]
        solution = solve_ivp(system01, [0, t_end], initial, t_eval=t_eval,
                             args=(beta0, gam, C, N), rtol=1e-8, atol=1e-10)

        eps = 0.1
        t_max_idx = len(t_eval) - 1
        for idx in range(len(t_eval)):
            I_all = np.array([solution.y[3 * j + 1, idx] for j in range(G)])
            if np.all(I_all < eps):
                t_max_idx = idx
                break

        t_eval_f = t_eval[:t_max_idx + 1]
        I_traj = np.vstack([solution.y[3 * j + 1, :t_max_idx + 1] for j in range(G)])  
        dI_dt = np.gradient(I_traj, t_eval_f, axis=1)
        peaks = []  # 每群体的首个峰值
        for i in range(G):
            di = dI_dt[i]
            idxs = np.where((di[:-1] > 0) & (di[1:] < 0))[0]
            if len(idxs) > 0:
                peaks.append(I_traj[i, idxs[0]])
            else:
                peaks.append(np.nan)
        peaks = np.array(peaks)



        min_I0 = float(I0.min())
        max_I0 = float(I0.max())
        max_peaks = float(np.nanmax(peaks))

        Tsmax=min(2100,max_peaks)
        if max_I0>=Tsmax:
            print('max_I0>=Tsmax')
            continue
        else:
            Ts = np.random.uniform(max_I0,Tsmax)

            # ===== alpha 扫描 =====
            alphas = np.eye(G)

            # 事件函数：delta(t) - Ts
            def delta_of_y(y):
                # y: 长度 3G
                Ivec = np.array([y[3*j+1] for j in range(G)])
                return Ivec

            T1_list, T2_list = [], []
            alpha_rows = []

            for a in alphas:  # a: 长度G 的 one-hot
                # print(a)
                def event_reach_Ts(t, y,*args):
                    Ivec = delta_of_y(y)
                    return float(a @ Ivec - Ts)
                event_reach_Ts.terminal = True
                event_reach_Ts.direction = 0

                delta0 = float(a @ I0)

                T1 = None
                T2 = None

                if delta0 <= Ts:
                    R_initial = np.array([beta0 * C[i, i] * S0[i] / (N[i] * gam) for i in range(G)])
                    R_initial_new = np.array([C[i, i] * S0[i] / N[i] for i in range(G)])

                    sol1 = solve_ivp(
                        system01, [0, t_end], initial,
                        events=event_reach_Ts, t_eval=t_eval, dense_output=True,
                        args=(beta0, gam, C, N), rtol=1e-8, atol=1e-10
                    )
                    if sol1.t_events[0].size > 0:
                        T1 = float(sol1.t_events[0][0])
                    else:
                        attention = 'Case1'
                        T1 = 1e10
                    # print('T1=',T1)
                else:
                    R_initial = np.array([beta0 * C[i, i] * S0[i] / (N[i] * gam) for i in range(G)])
                    R_initial_new = np.array([C[i, i] * S0[i] / N[i] for i in range(G)])

                    sol2 = solve_ivp(
                        system02, [0, t_end], initial,
                        events=event_reach_Ts, t_eval=t_eval, dense_output=True,
                        args=(beta0, gam, C, E, N), rtol=1e-8, atol=1e-10
                    )
                    if sol2.t_events[0].size > 0:
                        T2 = float(sol2.t_events[0][0])
                    else:
                        T2 = -1
                        attention = '删掉该条'
                    # print('T2=',T2)

                T1_list.append(T1)
                T2_list.append(T2)
                alpha_rows.append(a)

            # 取最小的T1
            T1_arr = np.array(T1_list)
            min_idx = int(np.nanargmin(T1_arr))
            alpha_star = alpha_rows[min_idx]


        row = [beta0, gam]

        row += list(N)

        row += list(S0_old)
        row += list(I0_old)
        row += list(R0_old)

        row += list(S0)
        row += list(I0)
        row += list(R0)

        row += C.flatten().tolist()
        row += E.flatten().tolist()

        row += [Ts, Tsmax, max_I0, max_peaks]

        row += peaks.tolist()
        row += list(alpha_star)

        row += [attention]

        I_gam_np = np.array(I_gam)
        row += I_gam_np.flatten().tolist()




    data.append(row)


df = pd.DataFrame(data, columns=columns)
df.to_csv('.csv', index=False)
