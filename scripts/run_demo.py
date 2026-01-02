import sys
from pathlib import Path
import json
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
from vendor.GeneralBacktest.backtest import GeneralBacktest

from src.data.csv_provider import CsvDataProvider
from src.agent.pklemon_public_agent import PKLemonPublicAgent, PKLemonPublicConfig

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    outdir = ROOT / args.outdir
    safe_mkdir(outdir)

    # 1) load data (public CSV)
    provider = CsvDataProvider()
    universe = provider.load_universe()
    prices = provider.load_prices()

    # 2) agent -> weights_df(date, code, weight)
    agent_cfg = PKLemonPublicConfig(
        start=cfg["start"],
        end=cfg["end"],
        warmup_days=int(cfg.get("warmup_days", 0)),

        sticky_buffer=int(cfg.get("sticky_buffer", 2)),

        shrink_cap=float(cfg.get("shrink_cap", 0.85)),

        turnover_gamma=float(cfg.get("turnover_gamma", 0.8)),

        include_first_rebalance=bool(cfg.get("include_first_rebalance", True)),

        rebalance_mode=str(cfg.get("rebalance_mode", "month_end")),


        # rebalance + public surrogate signals
        rebalance_freq=int(cfg.get("rebalance_freq", 5)),
        lookback_mom=int(cfg.get("lookback_mom", 20)),
        lookback_vol=int(cfg.get("lookback_vol", 60)),

        # within-class softmax temperature & weight cleanup
        within_temperature=float(cfg.get("within_temperature", cfg.get("softmax_temp", 0.8))),
        min_weight=float(cfg.get("min_weight", 0.01)),

        # PKLemon-aligned knobs
        shrink_lambda=float(cfg.get("shrink_lambda", 0.28)),
        no_trade_band=float(cfg.get("no_trade_band", 0.02)),

        cap_equity=float(cfg.get("cap_equity", 0.85)),
        cap_bond=float(cfg.get("cap_bond", 0.85)),
        cap_commodity=float(cfg.get("cap_commodity", 0.40)),
        cap_lo=float(cfg.get("cap_lo", 0.0)),
        cap_hi=float(cfg.get("cap_hi", 1.0)),

        risk_aversion_base=float(cfg.get("risk_aversion_base", 1.0)),
        risk_aversion_lo=float(cfg.get("risk_aversion_lo", 0.5)),
        risk_aversion_hi=float(cfg.get("risk_aversion_hi", 3.0)),

        core_sat_spread_base=float(cfg.get("core_sat_spread_base", 0.0)),
        core_sat_spread_lo=float(cfg.get("core_sat_spread_lo", -0.5)),
        core_sat_spread_hi=float(cfg.get("core_sat_spread_hi", 0.5)),

        base_bond_n=int(cfg.get("base_bond_n", 3)),
        base_commodity_n=int(cfg.get("base_commodity_n", 3)),
        topn_lo=int(cfg.get("topn_lo", 1)),
        topn_hi=int(cfg.get("topn_hi", 8)),

        class_weights=dict(cfg.get("class_weights", {})) or None,
        audit_dir=str(outdir / "audit"),
    )
    agent = PKLemonPublicAgent(agent_cfg)
    weights_df = agent.generate_weights(universe, prices)

    # save weights for inspection
    weights_path = outdir / "weights.csv"
    weights_df.to_csv(weights_path, index=False)

    # 3) backtest (DataFrame interface)
    bt = GeneralBacktest(start_date=cfg['start'], end_date=cfg['end'])

    res = bt.run_backtest(
        weights_data=weights_df,
        price_data=prices,
        buy_price=cfg.get("buy_price", "open"),
        sell_price=cfg.get("sell_price", "close"),
        adj_factor_col=cfg.get("adj_factor_col", "adj_factor"),
        close_price_col=cfg.get("close_price_col", "close"),
        date_col=cfg.get("date_col", "date"),
        asset_col=cfg.get("asset_col", "code"),
        weight_col=cfg.get("weight_col", "weight"),
        rebalance_threshold=float(cfg.get("rebalance_threshold", 0.005)),
        transaction_cost=list(cfg.get("transaction_cost", [0.001, 0.001])),
        initial_capital=float(cfg.get("initial_capital", 1.0)),
        slippage=float(cfg.get("slippage", 0.0)),
        benchmark_weights=None,
        benchmark_name=cfg.get("benchmark_name", "Benchmark"),
    )

    # 4) Save results (generic)
    # res is a dict; we save json + any DataFrames inside if present
    json_path = outdir / "backtest_result.json"
    serializable = {}

    for k, v in res.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(outdir / f"{k}.csv", index=False)
            serializable[k] = f"{k}.csv"
        elif isinstance(v, pd.Series):
            v.to_frame(name=k).reset_index().to_csv(outdir / f"{k}.csv", index=False)
            serializable[k] = f"{k}.csv"
        else:
            # try JSON-safe
            try:
                json.dumps(v)
                serializable[k] = v
            except TypeError:
                serializable[k] = str(v)

    json.dump(serializable, open(json_path, "w"), ensure_ascii=False, indent=2)

    # 5) Optional: call plotting / metrics if methods exist

    print("✅ Done.")
    print("Weights saved to:", weights_path)
    print("Backtest result index saved to:", json_path)
    print("Outputs dir:", outdir)

if __name__ == "__main__":
    main()
