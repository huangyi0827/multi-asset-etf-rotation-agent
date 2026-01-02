# PKLemon Public (Reproducible Pipeline)

This repo is a **public, reproducible** version of my PKLemon-style multi-asset ETF allocation/rotation system.
It keeps the **same architecture & interfaces** as the private full version, while using **surrogate signals + CSV demo data** for verification.

## What you can verify here
- Month-end rebalance calendar
- Router-like parameter dispatch (caps / risk_aversion / core-sat spread / topN) via public rules
- Turnover controls: sticky topN + no-trade band + adaptive shrink
- End-to-end pipeline: weights → backtest → trades/turnover → benchmark → audit logs

## Architecture
Public repo implements the reproducible pipeline; the private full version replaces the router with LLM-meta and uses licensed data.
![Architecture](docs/pklemon.png)


## Private full version (verified, not published)
- The full PKLemon version includes licensed data + proprietary signals + LLM-meta routing.
- It runs on CUFEL-Q Arena (simulated live; not real live trading) and is **continuously updated**.

Arena verification links (keep as-is):
- Leaderboard: https://cufel.cufe.edu.cn/cufel-q/leaderboard
- PKLemon Agent: https://cufel.cufe.edu.cn/cufel-q/agents/73
- SanCeagent: https://cufel.cufe.edu.cn/cufel-q/agents/28

---

## TL;DR
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python scripts/run_demo.py --config configs/demo.yaml
```

## Quickstart (public demo)

### One-liner
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python scripts/run_demo.py --config configs/demo.yaml
```

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### 2) Run demo
```bash
python scripts/run_demo.py --config configs/demo.yaml
```
### 3) Outputs
Generated under `outputs/`:
- `weights.csv` : rebalance-day portfolio weights
- `nav_series.csv` : NAV series
- `positions_df.csv` : positions by date
- `trade_records.csv` : trades
- `turnover_records.csv` : turnover records
- `benchmark_nav.csv` : benchmark NAV (EqualWeight)
- `benchmark_equal_weight.csv` : benchmark weights file
- `audit/audit.jsonl` : per-rebalance audit trail (features/params/selected/weights)

## Configuration (configs/demo.yaml)

Time range:
- `start`, `end`: backtest window (YYYY-MM-DD)

Rebalance & signal lookbacks:
- `rebalance_mode`: `month_end` (rebalance on month-end dates)
- `include_first_rebalance`: whether to rebalance at the first available date
- `rebalance_freq`: signal update step in trading days (public surrogate)
- `lookback_mom`, `lookback_vol`: lookback windows for momentum/volatility

Selection & weights:
- `topk_per_class`: top-K ETFs selected per class
- `softmax_temp`: within-class softmax temperature (higher → more diversified)
- `min_weight`: drop very small weights after normalization
- `class_weights`: strategic allocation across classes (EQUITY_CORE / EQUITY_SAT / BOND / COMMODITY)

Turnover controls (PKLemon-style):
- `sticky_buffer`: keep previously selected names for N periods to reduce churn
- `turnover_gamma`: turnover penalty strength
- `shrink_cap`: maximum shrinkage cap

Backtest assumptions:
- `transaction_cost`: [buy_cost, sell_cost] (0.001 = 10 bps each side)
- `rebalance_threshold`: ignore trades below this weight change
- `slippage`: additional slippage (decimal)

## Optional: regenerate demo data
The repo ships with fixed `demo_data/*.csv` for reproducibility.  
If you want to regenerate synthetic demo data locally:
```bash
python scripts/generate_demo_data.py
```

## Reporting (WIP)
A minimal summary script is provided: `python -m src.reporting.summary`.


## How to verify
- Reproduce the run using the one-liner / Quickstart command above.
- Inspect `outputs/audit/audit.jsonl` to audit each rebalance: features → params → selections → final weights.
- Compare strategy NAV (`outputs/nav_series.csv`) vs benchmark NAV (`outputs/benchmark_nav.csv`).


## Notes
- Public demo metrics validate the **pipeline and execution logic** and are not representative of the private full version.
- `.env.example` is reserved for **future database/API integrations** (not required for the public demo).


## Disclaimer
For research/education only. Not investment advice.
