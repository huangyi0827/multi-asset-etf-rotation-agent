import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

OUT = Path("outputs")
PLOTS = OUT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

def read_nav():
    p = OUT / "nav_series.csv"
    if not p.exists():
        raise FileNotFoundError("outputs/nav_series.csv not found")
    df = pd.read_csv(p)
    # 兼容两种格式：Series导出/两列导出
    if df.shape[1] >= 2:
        # 找日期列
        date_col = "date" if "date" in df.columns else df.columns[0]
        val_col = df.columns[-1]
        df[date_col] = pd.to_datetime(df[date_col])
        nav = df.set_index(date_col)[val_col].astype(float).sort_index()
        nav.name = "nav"
        return nav
    raise ValueError("nav_series.csv format not recognized")

def read_turnover():
    p = OUT / "turnover_records.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # 尝试识别字段
    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    # turnover列名可能不同，兜底取最后一列
    val_col = "turnover" if "turnover" in df.columns else df.columns[-1]
    return df.sort_values(date_col)[[date_col, val_col]].rename(columns={date_col:"date", val_col:"turnover"})

def read_positions_long():
    p = OUT / "positions_df.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # 兼容：可能是长表(date, code, weight) 或宽表
    cols = set(df.columns)
    if {"date","code","weight"}.issubset(cols):
        df["date"] = pd.to_datetime(df["date"])
        df["code"] = df["code"].astype(str)
        df["weight"] = df["weight"].astype(float)
        return df
    # 如果是宽表：第一列日期，其余是资产
    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    wide = df.set_index(date_col)
    long = wide.stack().reset_index()
    long.columns = ["date","code","weight"]
    return long

def save_nav_and_dd(nav: pd.Series):
    # NAV
    plt.figure()
    nav.plot()
    plt.title("NAV")
    plt.tight_layout()
    plt.savefig(PLOTS / "nav.png", dpi=200)
    plt.close()

    # Drawdown
    dd = nav / nav.cummax() - 1.0
    plt.figure()
    dd.plot()
    plt.title("Drawdown")
    plt.tight_layout()
    plt.savefig(PLOTS / "drawdown.png", dpi=200)
    plt.close()

def save_monthly_heatmap(nav: pd.Series):
    mret = nav.pct_change().resample("M").apply(lambda x: (1+x).prod()-1)
    df = mret.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    piv = df.pivot(index="year", columns="month", values="ret").fillna(0)

    plt.figure(figsize=(10,4))
    plt.imshow(piv.values, aspect="auto")
    plt.yticks(range(len(piv.index)), piv.index)
    plt.xticks(range(12), list(range(1,13)))
    plt.title("Monthly Return Heatmap")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(PLOTS / "monthly_return_heatmap.png", dpi=200)
    plt.close()

def save_turnover(turn: pd.DataFrame):
    if turn is None or turn.empty:
        return
    plt.figure()
    plt.plot(turn["date"], turn["turnover"])
    plt.title("Turnover per Rebalance")
    plt.tight_layout()
    plt.savefig(PLOTS / "turnover.png", dpi=200)
    plt.close()

def save_position_heatmap(pos_long: pd.DataFrame, topn=20):
    if pos_long is None or pos_long.empty:
        return
    mat = pos_long.pivot_table(index="date", columns="code", values="weight", aggfunc="sum").fillna(0)

    # 只画平均权重TopN，避免太密
    top_cols = mat.abs().mean().sort_values(ascending=False).head(topn).index
    mat2 = mat[top_cols]

    plt.figure(figsize=(10,5))
    plt.imshow(mat2.T.values, aspect="auto")
    plt.yticks(range(len(mat2.columns)), mat2.columns)
    plt.title(f"Position Heatmap (Top {topn} avg weight)")
    plt.tight_layout()
    plt.savefig(PLOTS / "position_heatmap.png", dpi=200)
    plt.close()

def main():
    nav = read_nav()
    save_nav_and_dd(nav)
    save_monthly_heatmap(nav)

    turn = read_turnover()
    save_turnover(turn)

    pos = read_positions_long()
    save_position_heatmap(pos, topn=20)

    print("✅ Saved plots to:", PLOTS)

if __name__ == "__main__":
    main()
