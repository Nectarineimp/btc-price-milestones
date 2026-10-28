#!/usr/bin/env python3
"""
btc-price-milestones: Ingestion and Multi-Base Passage Extraction
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
    generate_milestone_grid,
    get_market_era,
)


def ensure_directories():
    """Ensure data/raw, data/processed, and output directories exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_coinmetrics_btc_history(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetches daily BTC history from CoinMetrics community archive.
    Caches raw CSV to data/raw/btc_coinmetrics_raw.csv.
    """
    raw_cache_path = RAW_DIR / "btc_coinmetrics_raw.csv"

    if raw_cache_path.exists() and not force_refresh:
        print(f"[*] Loading raw CoinMetrics dataset from cache: {raw_cache_path}")
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
    print(f"[+] Raw upstream cache saved to: {raw_cache_path}")

    df = pd.read_csv(io.StringIO(response.text))
    df["time"] = pd.to_datetime(df["time"]).dt.normalize()
    df = df.rename(columns={"time": "date", "PriceUSD": "price"})
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df[["date", "price"]].dropna().sort_values("date").reset_index(drop=True)


def append_nls_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepends early New Liberty Standard (2009-2010) records prior to
    order-book exchanges and interpolates missing calendar days.
    """
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

    combined["log_price"] = np.log(combined["price"])
    combined["log_return"] = combined["log_price"].diff()
    return combined


def compute_milestone_passages(
    df: pd.DataFrame,
    milestones: list[float],
    window: int = 30,
    p_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Calculates instantaneous vs. sustained (rolling 5th percentile floor) first passage.
    """
    col_name = f"rolling_p{int((1 - p_threshold) * 100)}_support"
    if col_name not in df.columns:
        df[col_name] = df["price"].rolling(window=window).quantile(p_threshold)

    results = []
    t0 = df["date"].iloc[0]

    for m in milestones:
        raw_hits = df[df["price"] >= m]
        raw_date = raw_hits["date"].iloc[0] if not raw_hits.empty else None

        sustained_hits = df[df[col_name] >= m]
        sustained_date = sustained_hits["date"].iloc[0] if not sustained_hits.empty else None

        results.append({
            "milestone": m,
            "first_touch_raw": raw_date,
            "sustained_touch": sustained_date,
            "era_raw": get_market_era(raw_date) if pd.notnull(raw_date) else None,
            "era_sustained": get_market_era(sustained_date) if pd.notnull(sustained_date) else None,
        })

    res_df = pd.DataFrame(results)

    # Calculate calendar deltas and cumulative days from t0
    res_df["cumulative_days_raw"] = (res_df["first_touch_raw"] - t0).dt.days
    res_df["cumulative_days_sustained"] = (res_df["sustained_touch"] - t0).dt.days

    res_df["delta_days_raw"] = res_df["cumulative_days_raw"].diff()
    res_df["delta_days_sustained"] = res_df["cumulative_days_sustained"].diff()

    return res_df


def main():
    ensure_directories()

    # 1. Ingest & Prepare Clean Time Series
    raw_df = fetch_coinmetrics_btc_history(force_refresh=False)
    clean_df = append_nls_history(raw_df)

    clean_path = PROCESSED_DIR / "btc_daily_clean.csv"
    clean_df.to_csv(clean_path, index=False)
    print(f"[+] Clean continuous dataset saved: {clean_path}")

    # 2. Extract Milestones Across Progressions
    progressions = ["decadic", "euler", "binary"]

    for prog in progressions:
        print(f"[*] Processing {prog} milestone grid...")
        grid = generate_milestone_grid(prog, start_price=0.001, end_price=100000.0)
        res_df = compute_milestone_passages(clean_df, grid, window=30, p_threshold=0.05)

        out_path = PROCESSED_DIR / f"btc_milestones_{prog}.csv"
        res_df.to_csv(out_path, index=False)
        print(f"[+] Saved {len(res_df)} levels to: {out_path}")

    # Display Decadic Baseline Summary
    print("\n--- Decadic (Base-10) Passage Summary ---")
    dec_df = pd.read_csv(PROCESSED_DIR / "btc_milestones_decadic.csv")
    cols = ["milestone", "first_touch_raw", "sustained_touch", "delta_days_raw", "delta_days_sustained", "era_sustained"]
    print(dec_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()