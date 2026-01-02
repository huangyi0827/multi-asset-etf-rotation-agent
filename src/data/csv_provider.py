from dataclasses import dataclass
import pandas as pd

@dataclass
class CsvDataProvider:
    universe_path: str = "demo_data/universe.csv"
    prices_path: str = "demo_data/prices.csv"

    def load_universe(self) -> pd.DataFrame:
        df = pd.read_csv(self.universe_path)
        # 统一字段名：code / asset_class
        assert {"code", "asset_class"}.issubset(df.columns), "universe.csv must have code, asset_class"
        return df

    def load_prices(self) -> pd.DataFrame:
        df = pd.read_csv(self.prices_path)
        df["date"] = pd.to_datetime(df["date"])
        # 统一字段名：date / code / open/high/low/close/adj_factor
        need = {"date", "code", "open", "high", "low", "close", "adj_factor"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"prices.csv missing columns: {missing}")
        return df.sort_values(["date", "code"]).reset_index(drop=True)
