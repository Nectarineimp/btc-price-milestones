#!/usr/bin/env python3
"""
btc-price-milestones: Beta Evolution Across Market Eras
Evaluates how the temporal elasticity of Bitcoin price growth (beta) 
changes across distinct structural market regimes.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from config import PROCESSED_DIR, OUTPUT_DIR, MARKET_ERAS


def fit_power_law(ln_price, a, b):
    """ln(days) = a + b * ln_price, where beta = 1 / b"""
    return a + b * ln_price


def evaluate_beta_evolution():
    csv_file = PROCESSED_DIR / "btc_milestones_independent_eras.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Missing {csv_file}. Run milestones.py first.")

    df = pd.read_csv(csv_file)
    results = []

    print("\n[*] Evaluating Beta (Temporal Elasticity) Evolution Across Eras...")

    for era in MARKET_ERAS:
        era_subset = df[df["era_id"] == era.id].copy()

        # Prioritize sustained P95 data, fallback to raw if N is small
        valid = era_subset.dropna(subset=["sustained_date", "days_from_era_t0_sustained"])
        valid = valid[valid["days_from_era_t0_sustained"] > 0].reset_index(drop=True)

        is_fallback = False
        if len(valid) < 3:
            raw_subset = era_subset.dropna(subset=["raw_date", "days_from_era_t0_raw"])
            raw_subset = raw_subset[raw_subset["days_from_era_t0_raw"] > 0].reset_index(drop=True)
            if len(raw_subset) < 3:
                continue
            valid = raw_subset
            prices = valid["milestone"].values
            cum_days = valid["days_from_era_t0_raw"].values
            is_fallback = True
        else:
            prices = valid["milestone"].values
            cum_days = valid["days_from_era_t0_sustained"].values

        ln_p = np.log(prices)
        ln_d = np.log(cum_days)
        n = len(valid)

        popt, pcov = curve_fit(fit_power_law, ln_p, ln_d)
        b_slope = popt[1]
        b_stderr = np.sqrt(np.diag(pcov))[1]

        # beta = 1 / b
        beta = 1.0 / b_slope
        # Delta method for standard error of 1/b: SE(1/b) = SE(b) / b^2
        beta_stderr = b_stderr / (b_slope ** 2)
        ci_lower = max(0.0, beta - 1.96 * beta_stderr)
        ci_upper = beta + 1.96 * beta_stderr

        # Goodness of fit
        pred_ln_d = fit_power_law(ln_p, *popt)
        rss = np.sum((ln_d - pred_ln_d) ** 2)
        ss_tot = np.sum((ln_d - np.mean(ln_d)) ** 2)
        r2 = 1.0 - (rss / ss_tot) if ss_tot > 0 else 1.0

        results.append({
            "Era ID": era.id,
            "Market Era": era.name,
            "Sample (N)": n,
            "Data Type": "Raw Fallback" if is_fallback else "Sustained P95",
            "Beta (Elasticity)": round(beta, 3),
            "Beta 95% CI": f"[{ci_lower:.2f}, {ci_upper:.2f}]",
            "Slope (b)": round(b_slope, 3),
            "R2": round(r2, 4),
            "Price Span": f"${prices.min():,.2f} -> ${prices.max():,.2f}"
        })

    res_df = pd.DataFrame(results)

    # Compute delta beta between consecutive eras
    res_df["Delta Beta (%)"] = (res_df["Beta (Elasticity)"].pct_change() * 100).round(1)
    res_df["Regime Dynamics"] = np.where(
        res_df["Delta Beta (%)"].isna(), "Baseline Regime",
        np.where(res_df["Delta Beta (%)"] > 0, "Leverage Expansion / Acceleration", "Deceleration / Liquidity Friction")
    )

    print("\n--- Summary: Beta Trajectory Across Market Eras ---")
    print(res_df[["Market Era", "Sample (N)", "Beta (Elasticity)", "Beta 95% CI", "Delta Beta (%)", "Regime Dynamics", "R2"]].to_string(index=False))

    out_file = OUTPUT_DIR / "beta_evolution_summary.csv"
    res_df.to_csv(out_file, index=False)
    print(f"\n[✓] Results saved to: {out_file}")


if __name__ == "__main__":
    evaluate_beta_evolution()