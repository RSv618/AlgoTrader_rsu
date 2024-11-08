from itertools import combinations


def is_valid_comparison(left: str, right: str):
    # Extract the base column names (without the shift)
    left_base, left_shift = left.split()
    right_base, right_shift = right.split()

    # Ensure columns are not being compared to themselves
    if left == right:
        return False

    if left_shift == right_shift:  # Same bar
        # open > low (redundant) or open > high (impossible)
        if left_base == 'open' and right_base in ['low', 'high']:
            return False

        # high > [open, low, close] (redundant)
        if left_base == 'high' and right_base in ['open', 'low', 'close']:
            return False

        # low > [open, high, close] (impossible)
        if left_base == 'low' and right_base in ['open', 'high', 'close']:
            return False

        # close > low (redundant) or close > high (impossible)
        if left_base == 'close' and right_base in ['low', 'high']:
            return False

    elif left_shift == str(int(right_shift) - 1):  # Comparing current bar to previous bar
        # don't consider gap up
        if left_base in ['open', 'low'] and right_base == 'high':
            return False
        if left_base == 'open' and right_base in ['close', 'high']:
            return False

    elif str(int(left_shift) - 1) == right_shift:  # Comparing previous bar to current bar
        # don't consider gap down
        if left_base == 'low' and (right_base in ['high', 'open']):
            return False
        if left_base in ['close', 'low'] and right_base == 'open':
            return False

    return True


def extract_price_info(price_pair):
    """Helper function to extract the type (open, high, low) and time (index) from a price pair."""

    def parse_price_str(price_str):
        price_type, time_idx = price_str.split()
        return price_type, int(time_idx)

    return parse_price_str(price_pair[0]), parse_price_str(price_pair[1])


def no_direct_contradiction(pairs: tuple) -> bool:
    """Check if there are direct contradictions in the list of pairs
    (e.g., 'open 0' > 'open 1' and 'open 1' > 'open 0')."""
    seen_relations: set = set()

    for pair in pairs:
        (type1, time1), (type2, time2) = extract_price_info(pair)
        if (type2, time2, type1, time1) in seen_relations:
            return False  # Contradiction found (reverse relation already exists)
        seen_relations.add((type1, time1, type2, time2))
    return True


def no_transitive_contradiction(pairs: tuple) -> bool:
    """Check if there are transitive contradictions in the list of pairs
    (e.g., 'open 0' > 'open 1' and 'open 1' > 'open 0')."""
    facts: set = set()

    for pair in pairs:
        # initialize
        relations: set = set()

        greater, lesser = pair
        relations.add(f'{greater} > {lesser}')

        (type1, time1), (type2, time2) = extract_price_info(pair)
        if type1 in ['open', 'close', 'low']:
            if type2 in ['open', 'close', 'high']:
                relations.add(f'high {time1} > low {time2}')
                relations.add(f'{type1} {time1} > low {time2}')
                relations.add(f'high {time1} > {type2} {time2}')
            else:
                relations.add(f'high {time1} > {type2} {time2}')
        else:
            if type2 in ['open', 'close', 'high']:
                relations.add(f'{type1} {time1} > low {time2}')

        for relation in relations:
            a, b = relation.split(' > ')
            reverse_relation: str = f'{b} > {a}'
            if reverse_relation in facts:
                return False
            else:
                facts.add(relation)
    return True


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


if __name__ == '__main__':
    # Parameters
    seeds: list[int] = [
        841711,
        841688,
        678342,
        841171,
        841404,
        816459,
        792099,
        530930,
        727162,
        841234,
        841707,
        841568,
        12189,
        183397,
        684784,
        665033,
        409925,
        840455,
        882669,
        644233,
        880072,
        438204,
        272787,
        32198,
        841700,
        70501,
        476129,
        841697,
        42353,
        222720,
        832776,
        864490,
        858167,
        91692,
        175092,
        460616,
        521846,
        740775,
        32341,
        103082,
        796993,
        841059,
    ]

    shifts = [0, 1, 2, 3]  # Mistake: 4 should be 3. I don't think it matters much

    # Create a mapping of shifted column expressions
    col_shifts: list[str] = list(f"{col} {shift}" for col in ['open', 'high', 'low', 'close'] for shift in shifts)

    # Generate all valid comparisons
    valid_comparison: list = list((left_col, right_col) for left_col in col_shifts for right_col in col_shifts
                                  if is_valid_comparison(left_col, right_col))

    combi: list = list(combinations(valid_comparison, 3))
    combi = [comb for comb in combi if (no_direct_contradiction(comb) and no_transitive_contradiction(comb))]

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

    df = df.with_columns(stdev=pl.col('close').rolling_std(parameter['stdev']),
                         trigger=pl.when(up_trigger).then(1)
                         .otherwise(pl.when(down_trigger).then(-1).otherwise(0)))
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

    # To initialize persistent variables:
    # if 'example' not in persist:
    #     persist['example'] = 0

    def qty():
        return risk_per_trade * prev_balance / (prev_row.item(col['stdev']) * sqrt_n_per_day)

    trigger: bool = prev_row.item(col['trigger'])
    if 'count' not in persist:
        persist['count'] = 0

    if prev_position == 0:  # if no position
        persist['count'] = 0
        if trigger > 0:
            return [['buy_market', 0, qty()]]
        if trigger < 0:
            return [['sell_market', 0, qty()]]

    elif prev_position == 1:  # if long
        persist['count'] += 1
        if trigger < 0:
            return [['sell_market', 0, 0], ['sell_market', 0, qty()]]
        if persist['count'] >= parameter['time_exit']:
            return [['sell_market', 0, 0]]

    elif prev_position == -1:  # if short
        persist['count'] += 1
        if trigger > 0:
            return [['buy_market', 0, 0], ['buy_market', 0, qty()]]
        if persist['count'] >= parameter['time_exit']:
            return [['buy_market', 0, 0]]

    return []


def parameters(routine: str | None = None) -> list:
    # parameter scaling: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    headers: list[str] = ['stdev', 'time_exit']
    match routine:
        case 'parameter_range':
            stdev: list[int] = [8, 16, 32, 64, 128, 256, 512]
            time_exit: list[int] = [8, 16, 32, 64, 128, 256, 512]
        case _:
            stdev = [512]
            time_exit = [128, 256, 512]

    values: Any = iter_product(stdev, time_exit)

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

        with open(f'C:\\Users\\rober\\PycharmProjects\\AlgoTrader\\strategies_package\\rsPattern{seed}.py',
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
