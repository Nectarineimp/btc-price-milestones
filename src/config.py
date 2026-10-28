#!/usr/bin/env python3
"""
btc-price-milestones: Configuration and Regime Metadata
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "output"


@dataclass(frozen=True)
class MarketEra:
    name: str
    description: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp


MARKET_ERAS = [
    MarketEra(
        name="Era 1: Spot & Hobbyist",
        description="Early mining, peer-to-peer transfers, Mt. Gox, zero institutional rails.",
        start_date=pd.Timestamp("2009-10-05"),
        end_date=pd.Timestamp("2015-12-31"),
    ),
    MarketEra(
        name="Era 2: Offshore Derivatives",
        description="BitMEX perpetual swaps, CME futures launch, initial prop trading entry.",
        start_date=pd.Timestamp("2016-01-01"),
        end_date=pd.Timestamp("2019-12-31"),
    ),
    MarketEra(
        name="Era 3: Corporate & Macro Balance Sheets",
        description="MicroStrategy, Tesla, sovereign legal tender, central bank stimulus.",
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2023-12-31"),
    ),
    MarketEra(
        name="Era 4: Regulated Spot ETFs & Sovereign Macro",
        description="US spot ETFs (IBIT, FBTC), options clearing, national reserve allocations.",
        start_date=pd.Timestamp("2024-01-01"),
        end_date=pd.Timestamp("2035-12-31"),
    ),
]

STRUCTURAL_SHOCKS = {
    "Mt. Gox Insolvency": pd.Timestamp("2014-02-07"),
    "Blocksize Civil War / BCH Fork": pd.Timestamp("2017-08-01"),
    "COVID Liquidity Shock": pd.Timestamp("2020-03-12"),
    "FTX Collapse & Credit Contagion": pd.Timestamp("2022-11-11"),
    "US Spot ETF Inception": pd.Timestamp("2024-01-11"),
}


def get_market_era(dt: pd.Timestamp) -> str:
    """Classifies a given timestamp into one of the 4 defined structural eras."""
    for era in MARKET_ERAS:
        if era.start_date <= dt <= era.end_date:
            return era.name
    return "Era 4: Regulated Spot ETFs & Sovereign Macro"


def generate_milestone_grid(progression: str, start_price: float = 0.001, end_price: float = 100000.0) -> list[float]:
    """
    Generates target price thresholds for sensitivity analysis:
      - 'decadic': Base-10 orders of magnitude (10^k)
      - 'euler': e-folds (~2.718x per step, continuously compounded 1 nat return)
      - 'binary': Power-of-2 doublings (2x per step)
    """
    if progression == "decadic":
        powers = np.arange(np.log10(start_price), np.log10(end_price) + 0.001, 1.0)
        return list(np.round(10.0 ** powers, 4))
    elif progression == "euler":
        powers = np.arange(np.log(start_price), np.log(end_price) + 0.001, 1.0)
        return list(np.round(np.exp(powers), 4))
    elif progression == "binary":
        powers = np.arange(np.log2(start_price), np.log2(end_price) + 0.001, 1.0)
        return list(np.round(2.0 ** powers, 4))
    else:
        raise ValueError(f"Unknown progression: {progression}. Choose 'decadic', 'euler', or 'binary'.")