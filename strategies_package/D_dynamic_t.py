import polars as pl
import numpy as np
from typing import Any
from itertools import product as iter_product
import polars_talib as plta


def indicators(df: pl.DataFrame, parameter: dict[str, Any]) -> pl.DataFrame:
    threshold: float = parameter['threshold']

    c: pl.Expr = pl.col('close')

    window_size = df.select(window_size=plta.ht_dcperiod()).to_numpy()

    def calculate_dynamic_rolling(column_array: np.ndarray, window_size_array: np.ndarray) -> (np.ndarray, np.ndarray):
        rolling_max: np.ndarray = np.empty((len(column_array),
    ), dtype=np.float64)
        rolling_min: np.ndarray = np.empty((len(column_array),
    ), dtype=np.float64)

        # Iterate over each row
        for i in range(len(column_array)):
            # Get the window size for the current row
            window_s = window_size_array[i]
            if np.isnan(window_s):
                rolling_max[i] = np.nan
                rolling_min[i] = np.nan
                continue
            else:
                window_s = int(window_s * threshold + 1)

            # Ensure the window size is within bounds
            start: int = max(0, i - window_s + 1)

            # Slice the close column for the current window
            window: np.ndarray = column_array[start:i + 1]

            # Calculate the rolling max and min
            rolling_max[i] = window.max()
            rolling_min[i] = window.min()

        return rolling_max, rolling_min

    # Apply the function
    entry_high, entry_low = calculate_dynamic_rolling(df['close'].to_numpy(), window_size)
    entry_high = pl.Series('entry_high', entry_high)
    entry_low = pl.Series('entry_low', entry_low)

    uptrend_trigger_init: pl.Expr = (c > entry_high.shift(1))
    downtrend_trigger_init: pl.Expr = (c < entry_low.shift(1))
    uptrend_trigger: pl.Expr = uptrend_trigger_init & uptrend_trigger_init.shift(1).not_()
    downtrend_trigger: pl.Expr = downtrend_trigger_init & downtrend_trigger_init.shift(1).not_()

    df = df.with_columns(
        stdev=c.rolling_std(parameter['stdev']).cast(pl.Float64),
        entry_high=entry_high.cast(pl.Float64),
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
    headers: list[str] = ['stdev', 'threshold']
    match routine:
        case 'parameter_range':
            stdev: list[int] = [8, 16, 32, 64, 128, 256, 512]
            threshold: list[float] = [0.5, 1, 2]
        case _:
            stdev = [512]
            threshold = [0.5, 1.0, 2.0]

    values: Any = iter_product(stdev, threshold)

    dict_parameters: list[dict] = [dict(zip(headers, value)) for value in values]
    return dict_parameters
