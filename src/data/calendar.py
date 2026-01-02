import pandas as pd

def month_end_rebalance_days(trading_days: pd.DatetimeIndex, include_first: bool = True) -> pd.DatetimeIndex:
    """
    trading_days: 已排序的交易日 DatetimeIndex
    return: 每月最后一个交易日 +（可选）第一个交易日
    """
    if not isinstance(trading_days, pd.DatetimeIndex):
        trading_days = pd.DatetimeIndex(trading_days)
    td = pd.DatetimeIndex(pd.to_datetime(trading_days)).sort_values().unique()

    if len(td) == 0:
        return td

    # 每月最后一个交易日
    s = pd.Series(td, index=td)
    month_end = s.groupby([s.index.year, s.index.month]).max().sort_values().values
    out = pd.DatetimeIndex(month_end)

    if include_first:
        out = pd.DatetimeIndex([td[0]]).append(out)

    out = out.sort_values().unique()
    return out
