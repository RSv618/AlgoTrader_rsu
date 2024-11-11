import polars as pl
import numpy as np
from typing import Any
from itertools import product as iter_product
import polars_talib as plta


def indicators(df: pl.DataFrame, parameter: dict[str, Any]) -> pl.DataFrame:
    c: pl.Expr = pl.col('close')

    pattern: pl.Expr = plta.cdlharamicross()
    uptrend_trigger: pl.Expr = pattern > 0
    downtrend_trigger: pl.Expr = pattern < 0

    # df = df.with_columns(pattern=pattern, max=pattern.max(), min=pattern.min())
    df = df.with_columns(
        stdev=c.rolling_std(parameter['stdev']).cast(pl.Float64),
        uptrend_trigger=uptrend_trigger.cast(pl.Boolean),
        downtrend_trigger=downtrend_trigger.cast(pl.Boolean))
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
    max_count: int = parameter['count']
    count = persist['count']

    if prev_position == 0:  # if no position
        if uptrend_trigger:
            return [['buy_market', 0, qty()]]
        if downtrend_trigger:
            return [['sell_market', 0, qty()]]

    elif prev_position == 1:  # if long
        count += 1
        if count >= max_count:
            persist['count'] = 0
            return [['sell_market', 0, 0]]
        else:
            persist['count'] = count

        if downtrend_trigger:
            return [['sell_market', 0, 0], ['sell_market', 0, qty()]]

    elif prev_position == -1:  # if short
        count += 1
        if count >= max_count:
            persist['count'] = 0
            return [['buy_market', 0, 0]]
        else:
            persist['count'] = count

        if uptrend_trigger:
            return [['buy_market', 0, 0], ['buy_market', 0, qty()]]

    return []


def parameters(routine: str | None = None) -> list:
    # parameter scaling: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    headers: list[str] = ['stdev', 'count']
    match routine:
        case 'parameter_range':
            stdev: list[int] = [8, 16, 32, 64, 128, 256, 512]
            count: list[int] = [8, 16, 32, 64, 128, 256, 512]
        case _:
            stdev = [512]
            count = [8, 16, 32, 64, 128, 256, 512]

    values: Any = iter_product(stdev, count)

    dict_parameters: list[dict] = [dict(zip(headers, value)) for value in values]
    return dict_parameters
