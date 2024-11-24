import numpy as np

def str_to_plstr(string_int: str, reverse=False) -> str:
    a, b = string_int.split()
    if reverse:
        if a == 'high':
            a = 'low'
        elif a == 'low':
            a = 'high'
    if b == '0':
        return f'pl.col("{a}")'
    return f'pl.col("{a}").shift({b})'


def generate_strategy_from_seed(seeds: list[int],
                                combination_path: str = r'other_functions/3bar_3conditions_104876.npy'):
    combi: np.ndarray = np.load(combination_path)

    for seed in seeds:
        lines = '''import polars as pl
import numpy as np
from typing import Any
from itertools import product as iter_product


def indicators(df: pl.DataFrame, parameter: dict[str, Any]) -> pl.DataFrame:
    up_trigger: Any = ((REPLACE_A > REPLACE_B)
                       & (REPLACE_C > REPLACE_D)
                       & (REPLACE_E > REPLACE_F))
    down_trigger: Any = ((REPLACE_RA < REPLACE_RB)
                         & (REPLACE_RC < REPLACE_RD)
                         & (REPLACE_RE < REPLACE_RF))

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
    headers: list[str] = ['stdev', 'count']
    match routine:
        case 'parameter_range':
            stdev: list[int] = [8, 16, 32, 64, 128, 256, 512]
            count: list[int] = [8, 16, 32, 64, 128, 256, 512]
        case _:
            stdev = [512]
            count = [512]

    values: Any = iter_product(stdev, count)

    dict_parameters: list[dict] = [dict(zip(headers, value)) for value in values]
    return dict_parameters

    '''

        (a, b), (c, d), (e, f) = combi[seed]
        print(f'{seed=} {a=} {b=} {f=}')
        ra = str_to_plstr(a, reverse=True)
        rb = str_to_plstr(b, reverse=True)
        rc = str_to_plstr(c, reverse=True)
        rd = str_to_plstr(d, reverse=True)
        re = str_to_plstr(e, reverse=True)
        rf = str_to_plstr(f, reverse=True)
        a = str_to_plstr(a)
        b = str_to_plstr(b)
        c = str_to_plstr(c)
        d = str_to_plstr(d)
        e = str_to_plstr(e)
        f = str_to_plstr(f)

        with open(f'C:\\Users\\rober\\PycharmProjects\\AlgoTrader\\strategies_package\\Y_custom_{seed}.py',
                  'w') as file:
            lines = lines.replace('REPLACE_A', a)
            lines = lines.replace('REPLACE_B', b)
            lines = lines.replace('REPLACE_C', c)
            lines = lines.replace('REPLACE_D', d)
            lines = lines.replace('REPLACE_E', e)
            lines = lines.replace('REPLACE_F', f)
            lines = lines.replace('REPLACE_RA', ra)
            lines = lines.replace('REPLACE_RB', rb)
            lines = lines.replace('REPLACE_RC', rc)
            lines = lines.replace('REPLACE_RD', rd)
            lines = lines.replace('REPLACE_RE', re)
            lines = lines.replace('REPLACE_RF', rf)
            file.writelines(lines)
            print(f'{seed} done. Strategy created.')


if __name__ == '__main__':
    generate_strategy_from_seed([])