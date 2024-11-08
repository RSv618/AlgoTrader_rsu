from datetime import datetime, timedelta, timezone
from typing import Callable, Any
import time


def snap_date(date: datetime) -> datetime:
    """
    Snaps a given date to the nearest quarter start date (January 1, April 1, July 1, October 1).

    Parameters:
    date (datetime): The date to be snapped.

    Returns:
    datetime: The nearest quarter start date.
    """
    snap_dates: list[datetime] = [datetime(date.year, 1, 1),
                                  datetime(date.year, 4, 1),
                                  datetime(date.year, 7, 1),
                                  datetime(date.year, 10, 1),
                                  datetime(date.year + 1, 1, 1)]
    return min(snap_dates, key=lambda x: abs(x - date))


def walk_forward_dates(date_from: datetime, date_to: datetime, backtest_years: float = 2.0,
                       step_size: float = 0.5) -> list[tuple[datetime, datetime, datetime, datetime]]:
    """
    Generates a list of walk-forward dates for backtesting.

    Parameters:
    date_from (datetime): The start date for the initial in-sample period.
    date_to (datetime): The end date for the out-of-sample period.
    backtest_years (float): The number of years for the backtest period. Default is 2.0 years.
    step_size (float): The step size in years for each iteration. Default is 0.5 years.

    Returns:
    List[Tuple[datetime, datetime, datetime, datetime]]: List of tuples containing the start and end dates for in-sample
                                                          and out-of-sample periods.
    """
    days_in_a_year: float = 365.2425
    date_start_is: datetime = snap_date(date_from - timedelta(days=backtest_years * days_in_a_year))
    results: list[tuple[datetime, datetime, datetime, datetime]] = []

    while True:
        date_start_oos: datetime = snap_date(date_start_is + timedelta(days=backtest_years * days_in_a_year))
        date_end_oos: datetime = snap_date(date_start_oos + timedelta(days=step_size * days_in_a_year))
        if date_end_oos > date_to:
            break
        results.append((date_start_is, date_start_oos, date_start_oos, date_end_oos))
        date_start_is = snap_date(date_start_is + timedelta(days=step_size * days_in_a_year))
    return results


def next_best_param_dates(date_best_param_to: datetime, backtest_years: float = 2.0) -> tuple[datetime, datetime]:
    """
    Calculates the start and end dates for the next best parameter selection period.

    Parameters:
    date_best_param_to (datetime): The end date for the best parameter selection period.
    backtest_years (float): The number of years for the backtest period. Default is 2.0 years.

    Returns:
    Tuple[datetime, datetime]: The start and end dates for the next best parameter selection period.
    """
    days_in_a_year: float = 365.2425
    date_best_param_from: datetime = snap_date(date_best_param_to - timedelta(days=backtest_years * days_in_a_year))
    return date_best_param_from, date_best_param_to


def time_it(func: Callable) -> Callable:
    """
    Decorator to measure the execution time of a function.

    Parameters:
    func (Callable): The function to be measured.

    Returns:
    Callable: The wrapped function with execution time measurement.
    """

    def inner(*args, **kwargs) -> Any:

        start_time: float = time.time()

        result = func(*args, **kwargs)

        elapsed: float = time.time() - start_time
        function_name: str = func.__name__
        if elapsed >= 3600.0:
            elapsed /= 3600.0
            print(f'Function {function_name} took {elapsed:.2f} hours')
        elif elapsed >= 60.0:
            elapsed /= 60.0
            print(f'Function {function_name} took {elapsed:.2f} minutes')
        else:
            print(f'Function {function_name} took {elapsed:.2f} seconds')

        return result

    return inner


def make_utc(date: datetime) -> datetime:
    """
    Converts a date to UTC timezone.

    Parameters:
    date (datetime): The date to be converted.

    Returns:
    datetime: The date in UTC timezone.
    """
    if date.tzinfo is None:
        return date.replace(tzinfo=timezone.utc)
    return date
