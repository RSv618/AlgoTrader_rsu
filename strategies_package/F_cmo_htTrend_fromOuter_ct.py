import polars as pl
import numpy as np
from typing import Any
from itertools import product as iter_product
import polars_talib as plta


def indicators(df: pl.DataFrame, parameter: dict[str, Any]) -> pl.DataFrame:
    lookback: int = parameter['lookback']
    threshold: int = parameter['threshold']

    c: pl.Expr = pl.col('close')

    cmo: pl.Expr = plta.ht_trendline(plta.cmo(timeperiod=lookback))
    uptrend_trigger_init: pl.Expr = (cmo < threshold)
    downtrend_trigger_init: pl.Expr = (cmo > -threshold)
    uptrend_trigger: pl.Expr = uptrend_trigger_init & uptrend_trigger_init.shift(1).not_()
    downtrend_trigger: pl.Expr = downtrend_trigger_init & downtrend_trigger_init.shift(1).not_()

    df = df.with_columns(
        stdev=c.rolling_std(parameter['stdev']).cast(pl.Float64),
        uptrend_trigger=uptrend_trigger.cast(pl.Boolean),
        downtrend_trigger=downtrend_trigger.cast(pl.Boolean),
        # uptrend_filter=pl.lit(True).cast(pl.Boolean),
        # downtrend_filter=pl.lit(True).cast(pl.Boolean),
        # uptrend_entry=pl.lit(0.0).cast(pl.Float64),
        # downtrend_entry=pl.lit(0.0).cast(pl.Float64),
        # uptrend_exit=pl.lit(0.0).cast(pl.Float64),
        # downtrend_exit=pl.lit(0.0).cast(pl.Float64),
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

    uptrend_trigger: bool = prev_row.item(col['uptrend_trigger'])
    downtrend_trigger: bool = prev_row.item(col['downtrend_trigger'])

    if prev_position == 0:  # if no position
        if uptrend_trigger:
            return [['buy_market', 0, qty()]]
        if downtrend_trigger:
            return [['sell_market', 0, qty()]]

    elif prev_position == 1:  # if long
        if downtrend_trigger:
            return [['sell_market', 0, 0], ['sell_market', 0, qty()]]

    elif prev_position == -1:  # if short
        if uptrend_trigger:
            return [['buy_market', 0, 0], ['buy_market', 0, qty()]]

    return []


def parameters(routine: str | None = None) -> list:
    # parameter scaling: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    headers: list[str] = ['stdev', 'lookback', 'threshold']
    match routine:
        case 'parameter_range':
            stdev: list[int] = [8, 16, 32, 64, 128, 256, 512]
            lookback: list[int] = [8, 16, 32, 64, 128, 256, 512]
            threshold: list[int] = [0, 50, 100]
        case _:
            stdev = [512]
            lookback = [8, 16, 32, 64, 128, 256, 512]
            threshold = [0, 50, 100]

    values: Any = iter_product(stdev, lookback, threshold)

    dict_parameters: list[dict] = [dict(zip(headers, value)) for value in values]
    return dict_parameters
