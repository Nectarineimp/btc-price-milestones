# btc-price-milestones

Quantitative empirical evaluation of Bitcoin's order-of-magnitude ($10^k$) passage times, contrasting transient liquidity wicks with structural price consolidation.

---

## 1. Executive Overview & Problem Statement

A persistent narrative across retail and speculative crypto commentary suggests that Bitcoin advances along an orderly "ten-times ladder" ($10^{-3} \to 10^5$), with each progressive order of magnitude requiring systematically more calendar time due to asset maturation.

However, standard retail framing suffers from two severe analytical flaws:
1. **Survivorship & Framing Bias:** Base-10 scaling is an anthropogenic artifact of decimal accounting. It treats psychological round numbers as mechanical network properties without accounting for the non-linear growth in capital depth required to move market capitalization across orders of magnitude.
2. **First-Passage Distortion:** Standard analyses evaluate "first touch" on raw daily or intraday highs. In hyper-volatile, margin-leveraged regimes, these prints often represent transient blow-off wicks (e.g., the November 2013 spike to $1,000 on Mt. Gox) rather than sustained market acceptance. Structural trading above that level did not re-emerge until early 2017.

This project establishes a reproducible pipeline to calculate and contrast:
- **Instantaneous First-Passage Time ($\tau_{\text{raw}}$):** The earliest timestamp where market price breaches $10^k$.
- **Structural Support Passage Time ($\tau_{\text{struct}}$):** The earliest timestamp where the rolling 95th-percentile floor ($P_{95}$) establishes sustained consolidation above $10^k$.

Using these series, we test whether the time-delta progression ($\Delta t_k$) exhibits power-law deceleration, cyclic halving artifacts, or purely path-dependent noise.

---

## 2. Mathematical & Methodological Definitions

Let $P(t)$ denote the continuous or daily closing price series of Bitcoin in USD, defined over $t \in [t_0, T]$, where $t_0 = \text{2009-10-05}$ (initial New Liberty Standard clearing benchmark).

### 2.1 Decade Milestones
Milestones are indexed by integer order of magnitude $k$:
$$M_k = 10^k \quad \text{for } k \in \{-3, -2, -1, 0, 1, 2, 3, 4, 5, 6\}$$

### 2.2 Instantaneous First Passage ($\tau_{\text{raw}}$)
$$\tau_{\text{raw}}(M_k) = \min \{ t \ge t_0 \mid P(t) \ge M_k \}$$

### 2.3 Structural Support First Passage ($\tau_{\text{struct}}$)
To eliminate brief speculative anomalies and thin-liquidity squeeze events, we define structural support using a trailing rolling window $W$ (default $W = 30$ days) at a support quantile $(1 - \alpha)$ where $\alpha = 0.05$ (the empirical 5th percentile floor):

$$Q_{\alpha}(t, W) = \inf \left\{ p \in \mathbb{R} \mid \frac{1}{W} \sum_{\tau = t - W + 1}^{t} \mathbf{1}_{\{P(\tau) \le p\}} \ge \alpha \right\}$$

Structural passage requires the 5th percentile of the trailing lookback window to trade at or above the milestone:
$$\tau_{\text{struct}}(M_k, W, \alpha) = \min \{ t \ge t_0 \mid Q_{\alpha}(t, W) \ge M_k \}$$

*Interpretation:* At least $(1 - \alpha) = 95\%$ of all observations within the trailing $W$-day window must clear $M_k$.

### 2.4 Passage Duration Deltas
$$\Delta t_{k, \text{raw}} = \tau_{\text{raw}}(M_k) - \tau_{\text{raw}}(M_{k-1})$$
$$\Delta t_{k, \text{struct}} = \tau_{\text{struct}}(M_k) - \tau_{\text{struct}}(M_{k-1})$$

---

## 3. Repository Structure

```text
btc-price-milestones/
├── data/
│   ├── raw/                  # Immutable upstream captures (API snapshots, early records)
│   │   └── btc_daily_raw.csv
│   └── processed/            # Sanitized continuous time series & derived milestone tables
│       ├── btc_daily_clean.csv
│       └── btc_10x_milestones.csv
├── notebooks/                # Exploratory econometric visualization and curve fitting
├── src/
│   ├── __init__.py
│   ├── ingest.py             # Data retrieval and caching routines
│   ├── transform.py          # Early history stitching and daily interpolation
│   └── milestones.py         # Rolling quantile calculations and passage evaluators
├── tests/                    # Deterministic validation of quantile & edge conditions
├── pyproject.toml            # Poetry environment and dependency specification
├── README.md
└── LICENSE

---

## 4. Data Provenance & Pipeline Architecture

1. **Early Price Imputation (2009–2010):**
Prior to public exchange order books (pre-Mt. Gox July 2010), benchmark transactions are derived from New Liberty Standard (NLS) electricity-pegged price formulations ($0.00099 to $0.01) and early documented trades.
2. **Exchange Continuous Aggregation (2010–Present):**
Aggregated daily closes sourced via public API endpoints (CoinGecko / exchange composites).
3. **Data Hygiene & Caching:**
The ingestion client implements a write-once local cache in `data/raw/`. Upstream API calls are bypassed if valid historical snapshots exist on disk, guaranteeing offline reproducibility and execution speed.

---

## 5. Environment & Execution

This project is built using Python 3 and managed via [Poetry](https://python-poetry.org/).

### Installation

```bash
# Clone the repository
git clone [https://github.com/your-org/btc-price-milestones.git](https://github.com/your-org/btc-price-milestones.git)
cd btc-price-milestones

# Install dependencies via Poetry
poetry install

```

### Execution

To fetch data, build the continuous series, and generate the milestone delta analysis:

```bash
poetry run python src/milestones.py

```

To force a fresh upstream data pull bypassing the cache:

```bash
poetry run python src/milestones.py --refresh

```

---

## 6. Research Roadmap & Hypotheses

* [ ] **H1 (Deceleration Monotonicity):** Does $\Delta t_{k, \text{struct}}$ exhibit strictly monotonic growth compared to the cycle-distorted $\Delta t_{k, \text{raw}}$?
* [ ] **H2 (Power-Law Scaling):** Does a log-log regression of cumulative time $t_k$ vs. log-price $k$ yield an invariant scaling exponent, or does capital saturation cause super-power-law deceleration?
* [ ] **H3 (Extrapolation Validity):** Does extrapolating to $M_6$ ($1,000,000) under structural support provide an operationally distinct forecast window compared to naive all-time-high extrapolation?
