import polars as pl
import numpy as np
from typing import Any
from itertools import product as iter_product
import polars_talib as plta


def indicators(df: pl.DataFrame, parameter: dict[str, Any]) -> pl.DataFrame:
    slow: int = parameter['slow']
    fast: int = int(parameter['fast_m'] * slow)

    c: pl.Expr = pl.col('close')

    macd: pl.Expr = plta.macd(c, fast, slow)
    macd_df: pl.DataFrame = df.select(macd=macd).unnest('macd')
    signal_series: pl.Series = macd_df['macdsignal']
    df = df.with_columns(signal=signal_series)
    linear_regression_angle: pl.Expr = plta.linearreg_angle(pl.col('signal'), timeperiod=fast)
    uptrend_trigger_init: pl.Expr = linear_regression_angle < 0
    downtrend_trigger_init: pl.Expr = linear_regression_angle > 0
    uptrend_trigger: pl.Expr = uptrend_trigger_init & uptrend_trigger_init.shift(1).not_()
    downtrend_trigger: pl.Expr = downtrend_trigger_init & downtrend_trigger_init.shift(1).not_()

    df = df.with_columns(
        stdev=c.rolling_std(parameter['stdev']).cast(pl.Float64),
        linear_regression_angle=linear_regression_angle.cast(pl.Float64),
        uptrend_trigger=uptrend_trigger.cast(pl.Boolean),
        downtrend_trigger=downtrend_trigger.cast(pl.Boolean),
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
    headers: list[str] = ['stdev', 'slow', 'fast_m']
    match routine:
        case 'parameter_range':
            stdev: list[int] = [8, 16, 32, 64, 128, 256, 512]
            slow: list[int] = [16, 32, 64, 128, 256]
            fast_m: list[float] = [0.5]
        case _:
            stdev = [512]
            slow = [8, 16, 32, 64, 128, 256]
            fast_m = [0.5]

    values: Any = iter_product(stdev, slow, fast_m)

    dict_parameters: list[dict] = [dict(zip(headers, value)) for value in values]
    return dict_parameters
