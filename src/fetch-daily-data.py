#!/usr/bin/env python3
"""
btc-price-milestones:
Ingests, caches, and analyzes instantaneous vs. sustained (P95 support) 10x milestones.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import requests

# Base directory relative to this script (assumes script is in src/)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def ensure_directories():
    """Ensure data/raw and data/processed directories exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fetch_coingecko_daily(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetches full history of BTC/USD daily close from CoinGecko API.
    Uses cached CSV if present unless force_refresh=True.
    """
    raw_cache_path = RAW_DIR / "btc_daily_raw.csv"

    if raw_cache_path.exists() and not force_refresh:
        print(f"[*] Loading raw dataset from cache: {raw_cache_path}")
        df = pd.read_csv(raw_cache_path, parse_dates=["date"])
        return df

    print("[*] Fetching raw daily price history from CoinGecko API...")
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "max", "interval": "daily"}

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    prices = data["prices"]  # [[timestamp_ms, price], ...]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    df = df[["date", "price"]]

    # Save raw snapshot
    df.to_csv(raw_cache_path, index=False)
    print(f"[+] Raw data saved to: {raw_cache_path}")
    return df


def append_early_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepends early 2009-2010 NLS benchmarks and interpolates missing days
    to produce a continuous daily series.
    """
    early_records = [
        {"date": pd.Timestamp("2009-10-05"), "price": 0.00099},
        {"date": pd.Timestamp("2009-10-12"), "price": 0.00100},
        {"date": pd.Timestamp("2010-02-20"), "price": 0.01000},
        {"date": pd.Timestamp("2010-07-17"), "price": 0.08000},
        {"date": pd.Timestamp("2010-07-18"), "price": 0.10000},
    ]
    early_df = pd.DataFrame(early_records)
    combined = pd.concat([early_df, df]).drop_duplicates(subset=["date"]).sort_values("date")
    combined = combined.set_index("date").resample("D").interpolate(method="time").reset_index()
    return combined


def compute_milestone_passages(
    df: pd.DataFrame,
    milestones: list[float],
    window: int = 30,
    p_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Computes:
      1. First raw touch: min date where price >= M
      2. Sustained touch: min date where trailing window (1 - p_threshold) quantile >= M
         (e.g., 5th percentile >= M means 95% of days in window were >= M).
    """
    df["rolling_p95_support"] = df["price"].rolling(window=window).quantile(p_threshold)

    results = []
    for m in milestones:
        raw_hits = df[df["price"] >= m]
        raw_date = raw_hits["date"].iloc[0] if not raw_hits.empty else None

        sustained_hits = df[df["rolling_p95_support"] >= m]
        sustained_date = sustained_hits["date"].iloc[0] if not sustained_hits.empty else None

        results.append({
            "milestone": m,
            "log10_level": int(np.log10(m)) if m >= 1 else int(np.floor(np.log10(m))),
            "first_touch_raw": raw_date,
            "sustained_touch": sustained_date,
        })

    res_df = pd.DataFrame(results)

    # Calculate step deltas
    res_df["delta_days_raw"] = (res_df["first_touch_raw"] - res_df["first_touch_raw"].shift(1)).dt.days
    res_df["delta_days_sustained"] = (res_df["sustained_touch"] - res_df["sustained_touch"].shift(1)).dt.days

    return res_df


def main():
    ensure_directories()

    # 1. Fetch / Load Cached Raw Data
    raw_df = fetch_coingecko_daily(force_refresh=False)

    # 2. Build Continuous Cleaned Series
    print("[*] Processing series (prepending early records, interpolating continuous daily series)...")
    clean_df = append_early_history(raw_df)

    clean_path = PROCESSED_DIR / "btc_daily_clean.csv"
    clean_df.to_csv(clean_path, index=False)
    print(f"[+] Clean continuous dataset saved to: {clean_path}")

    # 3. Compute Milestones
    milestones = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    print("[*] Evaluating passage milestones (30-day window, 95% support threshold)...")
    results_df = compute_milestone_passages(clean_df, milestones, window=30, p_threshold=0.05)

    milestones_path = PROCESSED_DIR / "btc_10x_milestones.csv"
    results_df.to_csv(milestones_path, index=False)
    print(f"[+] Milestone results saved to: {milestones_path}\n")

    # 4. Display Formatted Output
    print("--- Summary: Raw vs. Sustained First Passage ---")
    display_df = results_df.copy()
    display_df["first_touch_raw"] = display_df["first_touch_raw"].dt.strftime("%Y-%m-%d")
    display_df["sustained_touch"] = display_df["sustained_touch"].dt.strftime("%Y-%m-%d")
    print(display_df.to_string(index=False))


if __name__ == "__main__":
    main()
