#!/usr/bin/env python3
"""
btc-price-milestones: Independent Era Analysis & Modeling
Fits isolated regression models for each Market Era to determine era-specific scaling rates.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from config import PROCESSED_DIR, OUTPUT_DIR, MARKET_ERAS


def fit_era_power_law(ln_price, a, b):
    """ln(days) = a + b * ln_price"""
    return a + b * ln_price


def compute_aic_bic(n: int, rss: float, k_params: int) -> tuple[float, float]:
    """Computes Akaike and Bayesian Information Criteria."""
    if rss <= 0 or n <= k_params:
        return np.nan, np.nan
    aic = n * np.log(rss / n) + 2 * k_params
    bic = n * np.log(rss / n) + k_params * np.log(n)
    return float(aic), float(bic)


def analyze_independent_eras():
    csv_file = PROCESSED_DIR / "btc_milestones_independent_eras.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Missing {csv_file}. Run milestones.py first.")

    df = pd.read_csv(csv_file)
    summary_rows = []

    print("\n[*] Econometric Regression for Independent Market Eras:")

    for era in MARKET_ERAS:
        era_subset = df[df["era_id"] == era.id].copy()
        
        # We need sustained passage times strictly within the era's timeline
        valid = era_subset.dropna(subset=["sustained_date", "days_from_era_t0_sustained"])
        valid = valid[valid["days_from_era_t0_sustained"] > 0].reset_index(drop=True)

        n = len(valid)
        if n < 3:
            # We can run the raw dates instead as backup if sustained points are too sparse
            raw_subset = era_subset.dropna(subset=["raw_date", "days_from_era_t0_raw"])
            raw_subset = raw_subset[raw_subset["days_from_era_t0_raw"] > 0].reset_index(drop=True)
            if len(raw_subset) < 3:
                print(f"    [!] {era.name}: Insufficient data points (N={n} sustained, N={len(raw_subset)} raw) to fit separate era model.")
                summary_rows.append({
                    "Era": era.name,
                    "N (Sustained)": n,
                    "Scaling (beta)": "N/A",
                    "R2": "N/A",
                    "Est. Days to 10x Price Expansion": "N/A",
                    "Status": "Insufficient points (Early Stage / Incomplete Era)"
                })
                continue
            # Fallback to raw touch only for estimation
            n = len(raw_subset)
            prices = raw_subset["milestone"].values
            cum_days = raw_subset["days_from_era_t0_raw"].values
            is_fallback = True
        else:
            prices = valid["milestone"].values
            cum_days = valid["days_from_era_t0_sustained"].values
            is_fallback = False

        ln_prices = np.log(prices)
        ln_days = np.log(cum_days)

        try:
            popt, _ = curve_fit(fit_era_power_law, ln_prices, ln_days)
            pred_ln_days = fit_era_power_law(ln_prices, *popt)
            rss = np.sum((ln_days - pred_ln_days) ** 2)
            ss_tot = np.sum((ln_days - np.mean(ln_days)) ** 2)
            r2 = 1.0 - (rss / ss_tot) if ss_tot > 0 else 1.0
            
            # scaling exponent (beta) 1/slope
            beta = 1.0 / popt[1]

            # Let's project how many days from the era's t0 to hit 10x multiple from the baseline anchor
            target_10x_price = era.anchor_price * 10.0
            ln_10x = np.log(target_10x_price)
            proj_ln_days_10x = fit_era_power_law(ln_10x, *popt)
            proj_days_10x = np.exp(proj_ln_days_10x)

            # Estimate targeted date
            t0_era = pd.Timestamp(era_subset["t0_era"].iloc[0])
            target_date_10x = t0_era + pd.Timedelta(days=int(proj_days_10x))

            # Estimate $1,000,000 target projection within this specific Era's trajectory
            ln_1m = np.log(1_000_000.0)
            proj_ln_days_1m = fit_era_power_law(ln_1m, *popt)
            proj_days_1m = np.exp(proj_ln_days_1m)
            target_date_1m = t0_era + pd.Timedelta(days=int(proj_days_1m)) if proj_days_1m < 100000 else None

            summary_rows.append({
                "Era": era.name,
                "Sample Size (N)": n,
                "Exponents (beta)": round(beta, 3),
                "Model Fit (R2)": round(r2, 4),
                "Base (Anchor Price)": f"${era.anchor_price:,.2f}",
                "10x Journey Duration (days)": f"{int(proj_days_10x):,} days",
                "10x Passage Date": target_date_10x.strftime("%Y-%m-%d"),
                "Isolated Proj Date ($1M)": target_date_1m.strftime("%Y-%m-%d") if target_date_1m else "Extrapolates Out Of Century",
                "Data Used": "Raw fallback" if is_fallback else "Sustained P95"
            })
            print(f"    [✓] {era.name}: Done (R2={r2:.4f}, Beta={beta:.3f})")
        except Exception as e:
            print(f"    [X] {era.name}: Regression failed ({str(e)})")

    summary_df = pd.DataFrame(summary_rows)
    print("\n--- Summary: Independent Era Price Discovery ---")
    print(summary_df.to_string(index=False))

    out_file = OUTPUT_DIR / "independent_era_selection_summary.csv"
    summary_df.to_csv(out_file, index=False)
    print(f"\n[✓] Results successfully written to: {out_file}")


if __name__ == "__main__":
    analyze_independent_eras()