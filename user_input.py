import strategies_package as st
from types import ModuleType
from typing import Any
from datetime import datetime, timedelta
from other_functions.date_and_time_functions import snap_date


def get_user_input() -> tuple[
    str,
    list[str],
    list[ModuleType],
    list[Any] | None,
    bool,
    list[tuple[datetime, datetime]]]:
    use_parallel_compute = False

    symbols: list[str] = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "NEOUSDT",
        "LTCUSDT",
        "QTUMUSDT",
        "ADAUSDT",
        "XRPUSDT",
        "EOSUSDT",
        "IOTAUSDT",
        "XLMUSDT",
        "ONTUSDT",
        "TRXUSDT",
        "ETCUSDT",
        "ICXUSDT",
        "NULSUSDT",
        "VETUSDT",
        "LINKUSDT",
        "WAVESUSDT"
    ]

    strategies_list: list[ModuleType] = [
        st.A_aroon_fromInner_t,
    ]

    strategies_list = [strat for strat in strategies_list if strat.__name__[-3:] != '_ct']

    routine: str = 'parameter_range'
    dates = [get_routine_dates(routine=routine)]

    """
    Example:
    combinations: list[Any] | None = [
        ('BTCUSDT', st.ma_cross_t),
        ('ETHUSDT', st.ma_cross_t2),
    ]
    """
    combinations: list[Any] | None = None

    if combinations is not None:
        if len(combinations) == 0:
            combinations = None
    return routine, symbols, strategies_list, combinations, use_parallel_compute, dates


def get_routine_dates(routine: str | None) -> tuple[datetime, datetime]:
    """
    Gets the start and end dates for a specific routine phase.

    Parameters:
    routine (Optional[str]): The routine phase for which dates are required.

    Returns:
    Tuple[datetime, datetime]: The start and end dates of the specified routine phase.

    Raises:
    ValueError: If the routine is invalid.
    """
    # reference_date == oos_phase_end
    reference_date: datetime = snap_date(datetime(2025, 1, 1))
    days_per_year: float = 365.2425

    # oos_phase: 2 years lookback
    oos_phase_end: datetime = reference_date
    oos_phase_start: datetime = snap_date(oos_phase_end - timedelta(days=days_per_year * 2.0))

    # validation_phase: 2 years lookback
    validation_phase_end: datetime = oos_phase_start
    validation_phase_start: datetime = snap_date(validation_phase_end - timedelta(days=days_per_year * 2.0))

    # initial_selection: 1 year lookback
    initial_selection_end: datetime = validation_phase_start
    initial_selection_start: datetime = snap_date(initial_selection_end - timedelta(days=days_per_year * 1.0))

    # parameter_range: 1 year lookback
    parameter_range_end: datetime = initial_selection_start
    parameter_range_start: datetime = snap_date(parameter_range_end - timedelta(days=days_per_year * 1.0))

    match routine:
        case 'oos_phase':
            return oos_phase_start, oos_phase_end
        case 'validation_phase':
            return validation_phase_start, validation_phase_end
        case 'validation_phase_ave_psr':
            return validation_phase_start, validation_phase_end
        case 'initial_selection':
            return initial_selection_start, initial_selection_end
        case 'initial_selection_ave_psr':
            return initial_selection_start, initial_selection_end
        case 'parameter_count':
            return initial_selection_start, initial_selection_end
        case 'parameter_range':
            return parameter_range_start, parameter_range_end
        case 'custom':
            return snap_date(validation_phase_start - timedelta(days=days_per_year * 2.0)), validation_phase_start
        case _:
            raise ValueError(f'Invalid routine {routine}')
