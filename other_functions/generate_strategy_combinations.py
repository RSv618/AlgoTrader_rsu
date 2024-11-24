import polars as pl
import numpy as np
from typing import Any
from itertools import product as iter_product
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


# Create a mapping of shifted column expressions
COL_SHIFTS: list[str] = list(f"{col} {shift}" for col in ['open', 'high', 'low', 'close'] for shift in [0, 1, 2])

# Generate all valid comparisons
VALID_COMPARISONS: list = list((left_col, right_col) for left_col in COL_SHIFTS for right_col in COL_SHIFTS
                               if is_valid_comparison(left_col, right_col))
print(len(VALID_COMPARISONS))
COMBI: list = list(combinations(VALID_COMPARISONS, 3))
COMBI = [comb for comb in COMBI if (no_direct_contradiction(comb) and no_transitive_contradiction(comb))]

COMBI_np = np.array(COMBI)
print(len(COMBI_np))
np.save('3bar_3conditions_104876.npy', COMBI_np)


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
            stdev: list[int] = [8, 16, 32, 64, 128, 256, 512]
            seed: list[int] = list(range(len(COMBI)//2, len(COMBI)))
            count: list[int] = [8, 16, 32, 64, 128, 256, 512]
        case _:
            stdev = [512]
            seed = []
            count = [64, 256, 512]

    values: Any = iter_product(stdev, seed, count)

    dict_parameters: list[dict] = [dict(zip(headers, value)) for value in values]
    return dict_parameters
