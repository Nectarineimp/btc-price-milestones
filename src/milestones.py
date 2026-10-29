#!/usr/bin/env python3
"""
btc-price-milestones: Independent Era Passage Extraction
Extracts and isolates historical and sustained passage times within each distinct Market Era.
"""

import io
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from config import (
    RAW_DIR,
    PROCESSED_DIR,
    OUTPUT_DIR,
    MARKET_ERAS,
    IndependentMarketEra,
)


def ensure_directories():
    """Ensure data and output directories exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_coinmetrics_btc_history(force_refresh: bool = False) -> pd.DataFrame:
    """Fetches daily BTC history from CoinMetrics community open archive and caches."""
    raw_cache_path = RAW_DIR / "btc_coinmetrics_raw.csv"

    if raw_cache_path.exists() and not force_refresh:
        df = pd.read_csv(raw_cache_path)
        df["time"] = pd.to_datetime(df["time"]).dt.normalize()
        df = df.rename(columns={"time": "date", "PriceUSD": "price"})
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        return df[["date", "price"]].dropna().sort_values("date").reset_index(drop=True)

    url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
    print("[*] Downloading historical BTC daily data from CoinMetrics GitHub...")

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    with open(raw_cache_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    df = pd.read_csv(io.StringIO(response.text))
    df["time"] = pd.to_datetime(df["time"]).dt.normalize()
    df = df.rename(columns={"time": "date", "PriceUSD": "price"})
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df[["date", "price"]].dropna().sort_values("date").reset_index(drop=True)


def append_nls_history(df: pd.DataFrame) -> pd.DataFrame:
    """Stitches early New Liberty Standard data & interpolates daily series."""
    early_records = [
        {"date": pd.Timestamp("2009-10-05"), "price": 0.00099},
        {"date": pd.Timestamp("2009-10-12"), "price": 0.00100},
        {"date": pd.Timestamp("2010-02-20"), "price": 0.01000},
        {"date": pd.Timestamp("2010-07-17"), "price": 0.08000},
    ]
    early_df = pd.DataFrame(early_records)
    min_cm_date = df["date"].min()
    early_df = early_df[early_df["date"] < min_cm_date]

    combined = pd.concat([early_df, df]).drop_duplicates(subset=["date"]).sort_values("date")
    combined = combined.set_index("date").resample("D").interpolate(method="time").reset_index()
    return combined


def compute_era_independent_passages(df: pd.DataFrame, era: IndependentMarketEra, window: int = 30, p_threshold: float = 0.05) -> pd.DataFrame:
    """
    Isolates price action to a single Era and tracks milestones starting
    from that specific Era's t0, discarding historical memory of prior eras.
    """
    # Slice to this era's timeline
    era_df = df[(df["date"] >= era.start_date) & (df["date"] <= era.end_date)].copy().reset_index(drop=True)
    if era_df.empty:
        return pd.DataFrame()

    # Create rolling Support Indicator (quantiles) just for this era's timeline
    support_col = f"rolling_p{int((1 - p_threshold) * 100)}_support"
    era_df[support_col] = era_df["price"].rolling(window=window, min_periods=1).quantile(p_threshold)

    results = []
    t0_era = era.start_date

    for milestone in era.target_milestones:
        # 1. Earliest touch strictly within this era
        raw_hits = era_df[era_df["price"] >= milestone]
        raw_date = raw_hits["date"].iloc[0] if not raw_hits.empty else None

        # 2. Earliest P95 Support within this era
        sustained_hits = era_df[era_df[support_col] >= milestone]
        sustained_date = sustained_hits["date"].iloc[0] if not sustained_hits.empty else None

        # Calculate time elapsed *from* the start of the era (t0_era)
        days_to_raw = (raw_date - t0_era).days if pd.notnull(raw_date) else None
        days_to_sustained = (sustained_date - t0_era).days if pd.notnull(sustained_date) else None

        results.append({
            "era_id": era.id,
            "era_name": era.name,
            "t0_era": t0_era.strftime("%Y-%m-%d"),
            "anchor_price": era.anchor_price,
            "milestone": milestone,
            "multiple_from_anchor": milestone / era.anchor_price,
            "raw_date": raw_date,
            "days_from_era_t0_raw": days_to_raw,
            "sustained_date": sustained_date,
            "days_from_era_t0_sustained": days_to_sustained,
        })

    return pd.DataFrame(results)


def main():
    ensure_directories()

    # Load Clean Series
    raw_df = fetch_coinmetrics_btc_history(force_refresh=False)
    clean_df = append_nls_history(raw_df)
    clean_df.to_csv(PROCESSED_DIR / "btc_daily_clean.csv", index=False)

    print("\n[*] Running Independent Era Passage Calculations...")
    all_era_results = []

    for era in MARKET_ERAS:
        print(f"    - Processing {era.name}...")
        era_res = compute_era_independent_passages(clean_df, era, window=30, p_threshold=0.05)
        if not era_res.empty:
            all_era_results.append(era_res)

    results_df = pd.concat(all_era_results, ignore_index=True)
    results_df.to_csv(PROCESSED_DIR / "btc_milestones_independent_eras.csv", index=False)
    print(f"\n[✓] Isolated milestone data saved to data/processed/btc_milestones_independent_eras.csv")

    # Grouped Quick Summary Output
    pd.set_option("display.width", 1000)
    for era in MARKET_ERAS:
        subset = results_df[results_df["era_id"] == era.id].copy()
        if subset.empty:
            continue
        print(f"\n=== {era.name.upper()} (Started: {era.start_date.strftime('%Y-%m-%d')}, Base: ${era.anchor_price:,.2f}) ===")
        print(subset[["milestone", "multiple_from_anchor", "raw_date", "days_from_era_t0_raw", "sustained_date", "days_from_era_t0_sustained"]].to_string(index=False))


if __name__ == "__main__":
    main()