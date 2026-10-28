#!/usr/bin/env python3
"""
btc-price-milestones: Econometric Model Selection & Extrapolation
Fits Power Law, Exponential Delta, and Logistic Growth models on sustained milestones.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from config import PROCESSED_DIR, OUTPUT_DIR


def compute_aic_bic(n: int, rss: float, k_params: int) -> tuple[float, float]:
    """Computes Akaike and Bayesian Information Criteria."""
    if rss <= 0 or n <= k_params:
        return np.nan, np.nan
    aic = n * np.log(rss / n) + 2 * k_params
    bic = n * np.log(rss / n) + k_params * np.log(n)
    return float(aic), float(bic)


# Functional forms
def power_law(ln_price, a, b):
    """ln(days) = a + b * ln(price)"""
    return a + b * ln_price


def logistic_func(t, L, k, t_mid):
    """P(t) = L / (1 + exp(-k * (t - t_mid)))"""
    return L / (1.0 + np.exp(-np.clip(k * (t - t_mid), -500, 500)))


def fit_models_for_progression(prog_name: str) -> dict:
    csv_file = PROCESSED_DIR / f"btc_milestones_{prog_name}.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Missing {csv_file}. Run milestones.py first.")

    df = pd.read_csv(csv_file)
    # Filter valid sustained passages
    valid = df.dropna(subset=["sustained_touch", "cumulative_days_sustained"]).copy()
    valid = valid[valid["cumulative_days_sustained"] > 0].reset_index(drop=True)

    prices = valid["milestone"].values
    cum_days = valid["cumulative_days_sustained"].values
    ln_prices = np.log(prices)
    ln_days = np.log(cum_days)
    n = len(valid)

    results = {"progression": prog_name, "sample_size": n}

    # 1. Power Law Fit: ln(t) vs ln(P)
    popt_pl, _ = curve_fit(power_law, ln_prices, ln_days)
    pred_ln_days = power_law(ln_prices, *popt_pl)
    rss_pl = np.sum((ln_days - pred_ln_days) ** 2)
    ss_tot_pl = np.sum((ln_days - np.mean(ln_days)) ** 2)
    r2_pl = 1.0 - (rss_pl / ss_tot_pl)
    aic_pl, bic_pl = compute_aic_bic(n, rss_pl, k_params=2)

    # Project $1,000,000 under Power Law
    ln_target = np.log(1_000_000.0)
    proj_ln_days_1m = power_law(ln_target, *popt_pl)
    proj_days_1m_pl = np.exp(proj_ln_days_1m)

    results["power_law"] = {
        "params": {"a": popt_pl[0], "b": popt_pl[1]},
        "scaling_exponent_beta": 1.0 / popt_pl[1],
        "R2": r2_pl,
        "AIC": aic_pl,
        "BIC": bic_pl,
        "projected_days_to_1m": proj_days_1m_pl,
    }

    # 2. Logistic Fit (TAM Saturation with Gold-parity Carrying Capacity L = $1,000,000)
    # Target capacity L = $1.5M (~$30T market cap)
    L_target = 1_500_000.0
    try:
        p0 = [L_target, 0.001, 4000.0]
        bounds = ([L_target * 0.99, 1e-5, 1000.0], [L_target * 1.01, 0.1, 15000.0])
        popt_log, _ = curve_fit(logistic_func, cum_days, prices, p0=p0, bounds=bounds, maxfev=10000)
        pred_prices = logistic_func(cum_days, *popt_log)
        rss_log = np.sum((prices - pred_prices) ** 2)
        ss_tot_log = np.sum((prices - np.mean(prices)) ** 2)
        r2_log = 1.0 - (rss_log / ss_tot_log)
        aic_log, bic_log = compute_aic_bic(n, rss_log, k_params=3)

        # Solve for t where P(t) = $1,000,000:
        # 1M = L / (1 + exp(-k(t - t_mid))) => t = t_mid - (1/k) * ln(L/1M - 1)
        k_val = popt_log[1]
        t_mid_val = popt_log[2]
        ratio = (popt_log[0] / 1_000_000.0) - 1.0
        proj_days_1m_log = t_mid_val - (1.0 / k_val) * np.log(ratio) if ratio > 0 else np.nan

        results["logistic"] = {
            "params": {"L": popt_log[0], "k": popt_log[1], "t_mid": popt_log[2]},
            "R2": r2_log,
            "AIC": aic_log,
            "BIC": bic_log,
            "projected_days_to_1m": proj_days_1m_log,
        }
    except Exception as e:
        results["logistic"] = {"error": str(e)}

    return results


def main():
    print("[*] Running econometric model fitting across progressions...")
    progressions = ["decadic", "euler", "binary"]
    t0 = pd.Timestamp("2009-10-05")

    summary_rows = []

    for prog in progressions:
        res = fit_models_for_progression(prog)
        pl = res["power_law"]

        proj_date_pl = t0 + pd.Timedelta(days=int(pl["projected_days_to_1m"]))

        row = {
            "Progression": prog.capitalize(),
            "Sample (N)": res["sample_size"],
            "PowerLaw Scaling Exponent (beta)": round(pl["scaling_exponent_beta"], 3),
            "PowerLaw R2": round(pl["R2"], 4),
            "PowerLaw AIC": round(pl["AIC"], 2),
            "PowerLaw Proj Date ($1M)": proj_date_pl.strftime("%Y-%m-%d"),
        }

        if "logistic" in res and "R2" in res["logistic"]:
            log_res = res["logistic"]
            proj_days_log = log_res["projected_days_to_1m"]
            if pd.notnull(proj_days_log):
                proj_date_log = t0 + pd.Timedelta(days=int(proj_days_log))
                row["Logistic Proj Date ($1M)"] = proj_date_log.strftime("%Y-%m-%d")
                row["Logistic R2"] = round(log_res["R2"], 4)
            else:
                row["Logistic Proj Date ($1M)"] = "Unreachable (< Carrying Capacity)"
                row["Logistic R2"] = round(log_res["R2"], 4)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print("\n--- Model Sensitivity & Projection Comparison ---")
    print(summary_df.to_string(index=False))

    out_file = OUTPUT_DIR / "model_selection_summary.csv"
    summary_df.to_csv(out_file, index=False)
    print(f"\n[+] Results written to: {out_file}")


if __name__ == "__main__":
    main()