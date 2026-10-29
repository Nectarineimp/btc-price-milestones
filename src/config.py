#!/usr/bin/env python3
"""
btc-price-milestones: Independent Multi-Era Configuration
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
class IndependentMarketEra:
    id: int
    name: str
    description: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    anchor_price: float      # The pricing baseline at t0 for this era's model
    target_milestones: list[float]


MARKET_ERAS = [
    IndependentMarketEra(
        id=1,
        name="Era 1: Spot & Hobbyist",
        description="P2P, early mining, Mt. Gox, zero institutional rails.",
        start_date=pd.Timestamp("2009-10-05"),
        end_date=pd.Timestamp("2015-12-31"),
        anchor_price=0.00099,
        target_milestones=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    ),
    IndependentMarketEra(
        id=2,
        name="Era 2: Offshore Derivatives",
        description="Synthetic leverage, perp swaps, CME futures launch, fork resolution.",
        start_date=pd.Timestamp("2016-01-01"),
        end_date=pd.Timestamp("2019-12-31"),
        anchor_price=430.0,  # ~Start of 2016 close
        target_milestones=[1000.0, 2000.0, 4000.0, 8000.0, 10000.0]  # Focus on key inflection triggers
    ),
    IndependentMarketEra(
        id=3,
        name="Era 3: Corporate & Macro",
        description="Balance sheets, stimulus, retail prime brokerage, FTX/3AC credit cycles.",
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2023-12-31"),
        anchor_price=7200.0,  # ~Start of 2020 close
        target_milestones=[10000.0, 20000.0, 30000.0, 40000.0, 50000.0, 60000.0]
    ),
    IndependentMarketEra(
        id=4,
        name="Era 4: ETFs & Sovereign Macro",
        description="Spot ETFs, RIA allocations, options clearing, geopolitical hedging.",
        start_date=pd.Timestamp("2024-01-01"),
        end_date=pd.Timestamp("2035-12-31"),  # Track current regime forward
        anchor_price=44000.0,  # ETF launch era baseline
        target_milestones=[50000.0, 80000.0, 100000.0, 250000.0, 500000.0, 1000000.0]
    )
]


def get_market_era(dt: pd.Timestamp) -> IndependentMarketEra:
    """Classifies a given timestamp into its corresponding Independent Market Era."""
    for era in MARKET_ERAS:
        if era.start_date <= dt <= era.end_date:
            return era
    return MARKET_ERAS[-1]  # Defaults to latest if out of range