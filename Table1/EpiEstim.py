import numpy as np
import pandas as pd
from scipy.stats import gamma
from epyestim.estimate_r import estimate_r, gamma_quantiles


def make_discrete_si(mean_si: float, std_si: float, max_days: int = 30) -> np.ndarray:
    shape = (mean_si / std_si) ** 2
    scale = (std_si ** 2) / mean_si
    days = np.arange(1, max_days + 1)
    cdf_upper = gamma.cdf(days + 0.5, a=shape, scale=scale)
    cdf_lower = gamma.cdf(days - 0.5, a=shape, scale=scale)
    si = cdf_upper - cdf_lower
    return si / si.sum()


def calc_R_last_from_14days(infections_14: np.ndarray, gt_distribution: np.ndarray,
                            a_prior: float, b_prior: float) -> float:
    """任何异常都返回 NaN"""
    try:
        infections = np.asarray(infections_14, dtype=float)
        if infections.shape[0] != 14 or (not np.isfinite(infections).all()):
            return np.nan

        infections_ts = pd.Series(
            infections,
            index=pd.RangeIndex(start=1, stop=15),
            name="infection"
        )

        r_post = estimate_r(
            infections_ts=infections_ts,
            gt_distribution=gt_distribution,
            a_prior=a_prior,
            b_prior=b_prior,
            window_size=14,
        )

        r_mean = r_post["a_posterior"] * r_post["b_posterior"]
        return float(r_mean.iloc[-1])
    except Exception:
        return np.nan


def main():
    df = pd.read_csv("Traindata_8d.csv")

    mean_si, std_si = 3.9,2.9
    gt_distribution = make_discrete_si(mean_si, std_si, max_days=30)

    a_prior, b_prior = 1.0, 0.2

    # 结果先放这里，最后一次性 concat（避免碎片化）
    results = {}

    for k in range(1, 9):
        infection_cols = [f"gamI{k}-Day{i}" for i in range(1, 15)]

        # 这里用 reindex：缺列会自动补 NaN，但不会往 df 里“插列”
        block = df.reindex(columns=infection_cols).apply(pd.to_numeric, errors="coerce")

        out_col = f"R{k}-Day14"
        results[out_col] = block.apply(
            lambda s: calc_R_last_from_14days(
                s.to_numpy(), gt_distribution, a_prior, b_prior
            ),
            axis=1
        )

    # 一次性合并 + copy 去碎片
    df_out = pd.concat([df, pd.DataFrame(results)], axis=1).copy()
    df_out.to_csv("EpiEstim_Rt.csv", index=False)


if __name__ == "__main__":
    main()
