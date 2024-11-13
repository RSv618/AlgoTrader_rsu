import polars as pl
import numpy as np
from typing import Any
from itertools import product as iter_product
import polars_talib as plta


def indicators(df: pl.DataFrame, parameter: dict[str, Any]) -> pl.DataFrame:
    lookback: int = parameter['lookback']
    exit_lookback: int = int(parameter['exit_m'] * lookback)

    c: pl.Expr = pl.col('close')
    h: pl.Expr = pl.col('high')
    l: pl.Expr = pl.col('low')

    entry_high: pl.Expr = h.rolling_max(lookback)
    entry_low: pl.Expr = l.rolling_min(lookback)
    exit_high: pl.Expr = h.rolling_max(exit_lookback)
    exit_low: pl.Expr = l.rolling_min(exit_lookback)
    
    df = df.with_columns(
        stdev=c.rolling_std(parameter['stdev']).cast(pl.Float64),
        uptrend_entry=entry_high.cast(pl.Float64),
        downtrend_entry=entry_low.cast(pl.Float64),
        uptrend_exit=exit_low.cast(pl.Float64),
        downtrend_exit=exit_high.cast(pl.Float64),
    )
    return df


def trade_logic(prev_row: np.ndarray, col: dict[str, int], prev_position: int,
                prev_balance: float, parameter: dict[str, Any], risk_per_trade: float,
                sqrt_n_per_day: float, persist: dict[str, Any]) -> list[list[str | int | float]]:
    """
    orders = [[order_type, price, qty], [...], ...]
    order_type = ['buy_market', 'buy_stop', 'buy_limit', 'sell_market', 'sell_stop', 'sell_limit']
    price = 0 if market_order else price
    qty = 0 if exit_order else qty
    """

    def qty():
        return risk_per_trade * prev_balance / (prev_row.item(col['stdev']) * sqrt_n_per_day)

    uptrend_entry: bool = prev_row.item(col['uptrend_entry'])
    downtrend_entry: bool = prev_row.item(col['downtrend_entry'])
    uptrend_exit: bool = prev_row.item(col['uptrend_exit'])
    downtrend_exit: bool = prev_row.item(col['downtrend_exit'])

    if prev_position == 0:  # if no position
        return [['buy_limit', downtrend_entry, qty()], ['sell_limit', uptrend_entry, qty()]]

    elif prev_position == 1:  # if long
        return [['sell_limit', downtrend_exit, 0], ['sell_limit', uptrend_entry, qty()]]

    elif prev_position == -1:  # if short
        return [['buy_limit', uptrend_exit, 0], ['buy_limit', downtrend_entry, qty()]]

    else:
        return [['buy_market', 0, 0], ['sell_market', 0, 0], ['handler', 0, 0]]


def parameters(routine: str | None = None) -> list:
    # parameter scaling: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    headers: list[str] = ['stdev', 'lookback', 'exit_m']
    match routine:
        case 'parameter_range':
            stdev: list[int] = [8, 16, 32, 64, 128, 256, 512]
            lookback: list[int] = [8, 16, 32, 64, 128, 256, 512]
            exit_m: list[float] = [1, 0.5]
        case _:
            stdev = [512]
            exit_m = [1]
            lookback = [128, 256, 512]

    values: Any = iter_product(stdev, lookback, exit_m)

    dict_parameters: list[dict] = [dict(zip(headers, value)) for value in values]
    return dict_parameters
