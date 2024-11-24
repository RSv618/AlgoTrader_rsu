import polars as pl
import numpy as np
from typing import Any
from itertools import product as iter_product
from itertools import combinations

COMBI: np.ndarray = np.load(r'other_functions/3bar_3conditions_104876.npy')
def str_to_expr(string_int: str, reverse=False) -> pl.Expr:
    a, b = string_int.split()
    if reverse:
        if a == 'high':
            a = 'low'
        elif a == 'low':
            a = 'high'
    return pl.col(a).shift(int(b))


def indicators(df: pl.DataFrame, parameter: dict[str, Any]) -> pl.DataFrame:
    seed: int = parameter['seed']

    (a, b), (c, d), (e, f) = COMBI[seed]
    ra = str_to_expr(a, reverse=True)
    rb = str_to_expr(b, reverse=True)
    rc = str_to_expr(c, reverse=True)
    rd = str_to_expr(d, reverse=True)
    re = str_to_expr(e, reverse=True)
    rf = str_to_expr(f, reverse=True)
    a = str_to_expr(a)
    b = str_to_expr(b)
    c = str_to_expr(c)
    d = str_to_expr(d)
    e = str_to_expr(e)
    f = str_to_expr(f)
    uptrend_trigger: pl.Expr = ((a > b) & (c > d) & (e > f))
    downtrend_trigger: pl.Expr = ((ra < rb) & (rc < rd) & (re < rf))

    df = df.with_columns(stdev=c.rolling_std(parameter['stdev']),
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
    headers: list[str] = ['stdev', 'seed', 'count']
    match routine:
        case 'parameter_range':
            stdev: list[int] = [512]
            seed: list[int] = list(range(104876))
            count: list[int] = [512]
        case _:
            stdev = [512]
            seed = []
            count = [512]

    values: Any = iter_product(stdev, seed, count)

    dict_parameters: list[dict] = [dict(zip(headers, value)) for value in values]
    return dict_parameters
