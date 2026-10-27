#!/usr/bin/env python3
"""
btc-price-milestones:
Ingests historical daily data from CoinMetrics community archive,
prepends early 2009-2010 NLS benchmarks, and computes instantaneous vs.
sustained (P95 support) 10x milestones.
"""

import io
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


def fetch_coinmetrics_btc_history(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetches daily BTC history (July 2010 to present) from CoinMetrics.
    Caches raw CSV to data/raw/btc_coinmetrics_raw.csv.
    """
    raw_cache_path = RAW_DIR / "btc_coinmetrics_raw.csv"

    if raw_cache_path.exists() and not force_refresh:
        print(f"[*] Loading raw CoinMetrics dataset from cache: {raw_cache_path}")
        return pd.read_csv(raw_cache_path, parse_dates=["date"])

    url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
    print(f"[*] Downloading historical BTC data from CoinMetrics GitHub...")

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    # Raw cache write
    with open(raw_cache_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"[+] Raw upstream cache saved to: {raw_cache_path}")

    df = pd.read_csv(io.StringIO(response.text))
    df["time"] = pd.to_datetime(df["time"]).dt.normalize()
    df = df.rename(columns={"time": "date", "PriceUSD": "price"})

    # Filter to valid numeric prices
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[["date", "price"]].dropna(subset=["price"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


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

    # Feature engineering
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
    Calculates:
      1. First raw touch: earliest date where price >= M
      2. Sustained touch: earliest date where the trailing window (1 - p_threshold)
         quantile >= M (5th percentile >= M => 95% of days in window were >= M).
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

    # Step deltas in calendar days
    res_df["delta_days_raw"] = (res_df["first_touch_raw"] - res_df["first_touch_raw"].shift(1)).dt.days
    res_df["delta_days_sustained"] = (res_df["sustained_touch"] - res_df["sustained_touch"].shift(1)).dt.days

    return res_df


def main():
    ensure_directories()

    # 1. Fetch / Ingest from CoinMetrics
    cm_df = fetch_coinmetrics_btc_history(force_refresh=False)

    # 2. Build Continuous Series (with NLS Prepend)
    print("[*] Processing series (stitching 2009-2010 NLS, generating daily interpolation)...")
    clean_df = append_nls_history(cm_df)

    clean_path = PROCESSED_DIR / "btc_daily_clean.csv"
    clean_df.to_csv(clean_path, index=False)
    print(f"[+] Clean continuous dataset saved to: {clean_path}")
    print(f"    Date Range:     {clean_df['date'].min().strftime('%Y-%m-%d')} to {clean_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"    Starting Price: ${clean_df['price'].iloc[0]:.5f}")
    print(f"    Latest Price:   ${clean_df['price'].iloc[-1]:,.2f}")

    # 3. Calculate 10x Milestones
    milestones = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    print("\n[*] Evaluating passage milestones (30-day window, 95% support threshold)...")
    results_df = compute_milestone_passages(clean_df, milestones, window=30, p_threshold=0.05)

    milestones_path = PROCESSED_DIR / "btc_10x_milestones.csv"
    results_df.to_csv(milestones_path, index=False)
    print(f"[+] Milestone results saved to: {milestones_path}\n")

    # 4. Display Results
    display_df = results_df.copy()
    display_df["first_touch_raw"] = display_df["first_touch_raw"].dt.strftime("%Y-%m-%d")
    display_df["sustained_touch"] = display_df["sustained_touch"].dt.strftime("%Y-%m-%d")
    print(display_df.to_string(index=False))


if __name__ == "__main__":
    main()