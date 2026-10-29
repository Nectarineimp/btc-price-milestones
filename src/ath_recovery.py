#!/usr/bin/env python3
"""
btc-price-milestones: ATH Recovery and Underwater Duration Analysis
Quantifies drawdown depth, recovery duration, and recovery velocity
across major historical market cycles.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from config import PROCESSED_DIR, OUTPUT_DIR, get_market_era


def ensure_output_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_clean_data() -> pd.DataFrame:
    clean_path = PROCESSED_DIR / "btc_daily_clean.csv"
    if not clean_path.exists():
        raise FileNotFoundError(f"Missing {clean_path}. Run milestones.py first.")
    df = pd.read_csv(clean_path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def identify_ath_drawdown_cycles(
    df: pd.DataFrame,
    min_drawdown_pct: float = 0.20,
    min_duration_days: int = 30,
) -> pd.DataFrame:
    """
    Identifies macro ATH drawdown and recovery cycles using a continuous
    running maximum filter.
    """
    df = df.copy()
    df["running_max"] = df["price"].cummax()
    df["drawdown"] = (df["price"] - df["running_max"]) / df["running_max"]

    at_ath = df["drawdown"] == 0
    cycles = []
    in_drawdown = False
    ath_idx = 0

    for i in range(len(df)):
        if at_ath.iloc[i]:
            if in_drawdown:
                # Cycle complete: slice from previous ATH to current recovery index
                cycle_slice = df.iloc[ath_idx : i + 1]
                trough_idx = cycle_slice["price"].idxmin()

                max_dd = df.loc[trough_idx, "drawdown"]
                total_days = (df.loc[i, "date"] - df.loc[ath_idx, "date"]).days

                # Filter out minor short-term intraday dips
                if abs(max_dd) >= min_drawdown_pct and total_days >= min_duration_days:
                    days_bleed = (df.loc[trough_idx, "date"] - df.loc[ath_idx, "date"]).days
                    days_climb = (df.loc[i, "date"] - df.loc[trough_idx, "date"]).days
                    multiple_recovery = df.loc[ath_idx, "price"] / df.loc[trough_idx, "price"]

                    # Annualized recovery velocity
                    recovery_cagr = (multiple_recovery ** (365.25 / max(1, days_climb))) - 1.0

                    ath_date = df.loc[ath_idx, "date"]
                    recovery_date = df.loc[i, "date"]

                    cycles.append({
                        "cycle_id": len(cycles) + 1,
                        "origin_era": get_market_era(ath_date).name,
                        "exit_era": get_market_era(recovery_date).name,
                        "ath_date": ath_date,
                        "ath_price": round(df.loc[ath_idx, "price"], 2),
                        "trough_date": df.loc[trough_idx, "date"],
                        "trough_price": round(df.loc[trough_idx, "price"], 2),
                        "recovery_date": recovery_date,
                        "max_drawdown_pct": round(max_dd * 100, 2),
                        "days_to_trough": days_bleed,
                        "days_trough_to_recovery": days_climb,
                        "total_underwater_days": total_days,
                        "recovery_multiple": round(multiple_recovery, 2),
                        "annualized_recovery_velocity_pct": round(recovery_cagr * 100, 1),
                        "status": "Recovered",
                    })
                in_drawdown = False
            ath_idx = i
        else:
            in_drawdown = True

    # Account for any active ongoing drawdown at the tail of the data
    if in_drawdown:
        cycle_slice = df.iloc[ath_idx:]
        trough_idx = cycle_slice["price"].idxmin()
        max_dd = df.loc[trough_idx, "drawdown"]
        total_days = (df.iloc[-1]["date"] - df.loc[ath_idx, "date"]).days

        if abs(max_dd) >= min_drawdown_pct:
            days_bleed = (df.loc[trough_idx, "date"] - df.loc[ath_idx, "date"]).days
            ath_date = df.loc[ath_idx, "date"]

            cycles.append({
                "cycle_id": len(cycles) + 1,
                "origin_era": get_market_era(ath_date).name,
                "exit_era": "Ongoing",
                "ath_date": ath_date,
                "ath_price": round(df.loc[ath_idx, "price"], 2),
                "trough_date": df.loc[trough_idx, "date"],
                "trough_price": round(df.loc[trough_idx, "price"], 2),
                "recovery_date": pd.NaT,
                "max_drawdown_pct": round(max_dd * 100, 2),
                "days_to_trough": days_bleed,
                "days_trough_to_recovery": np.nan,
                "total_underwater_days": total_days,
                "recovery_multiple": round(df.loc[ath_idx, "price"] / df.loc[trough_idx, "price"], 2),
                "annualized_recovery_velocity_pct": np.nan,
                "status": "Ongoing",
            })

    return pd.DataFrame(cycles)


def main():
    ensure_output_directories()
    print("[*] Loading continuous price data...")
    df = load_clean_data()

    print("[*] Identifying macro ATH drawdown and recovery cycles (min DD >= 20%)...")
    cycles_df = identify_ath_drawdown_cycles(df, min_drawdown_pct=0.20, min_duration_days=30)

    # Save outputs
    out_csv = OUTPUT_DIR / "ath_recovery_cycles.csv"
    cycles_df.to_csv(out_csv, index=False)
    print(f"[✓] Full cycle dataset written to: {out_csv}\n")

    # Formatted terminal report
    display_cols = [
        "cycle_id",
        "origin_era",
        "ath_date",
        "ath_price",
        "max_drawdown_pct",
        "days_to_trough",
        "days_trough_to_recovery",
        "total_underwater_days",
        "annualized_recovery_velocity_pct",
        "status",
    ]
    report_df = cycles_df[display_cols].copy()
    report_df["ath_date"] = report_df["ath_date"].dt.strftime("%Y-%m-%d")

    print("=================== ALL-TIME HIGH RECOVERY TRAJECTORIES ===================")
    print(report_df.to_string(index=False))

    # Cross-Era Synthesis
    print("\n================ CROSS-ERA UNDERWATER SUMMARY ================")
    summary = cycles_df.groupby("origin_era").agg(
        num_cycles=("cycle_id", "count"),
        avg_drawdown_pct=("max_drawdown_pct", "mean"),
        avg_days_to_trough=("days_to_trough", "mean"),
        avg_days_to_recover=("days_trough_to_recovery", "mean"),
        avg_total_underwater=("total_underwater_days", "mean"),
        avg_recovery_velocity_cagr=("annualized_recovery_velocity_pct", "mean"),
    ).reset_index()
    print(summary.round(1).to_string(index=False))


if __name__ == "__main__":
    main()