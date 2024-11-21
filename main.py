# update __init__.py of strategies_package
from other_functions.strategies_auto_import import auto_import

# Builtin modules
from datetime import datetime, timezone
from itertools import product as iter_product
from warnings import warn

# Third party modules
import numpy as np
import polars as pl
from joblib import Parallel, delayed
import scipy.stats as ss
from tqdm import tqdm

# Custom modules
import other_functions.more_funcs as mf
import other_functions.path_functions as path
import other_functions.date_and_time_functions as dtf
import user_input

# Typing
from typing import Any, Callable
from types import ModuleType


def main() -> None:
    """
    Main function to execute the trading strategy based on user input.

    The function retrieves user inputs for the routine, symbols, strategies, and combinations. It then executes the
    appropriate algorithm based on the specified routine.

    The following routines are supported:
    - 'oos_phase': Runs the out-of-sample phase algorithm.
    - 'validation_phase': Runs the validation phase algorithm.
    - 'initial_selection': Runs the initial selection algorithm.
    - 'parameter_count': Evaluates parameter counts and generates summary data.
    - 'parameter_range': Evaluates and normalizes parameter ranges.

    Raises:
    - ValueError: If the routine specified by the user is invalid.
    """
    routine, symbols, strategies_list, combinations, use_parallel_compute, dates = user_input.get_user_input()
    match routine:
        case 'oos_phase':
            wf_algo(symbols=None, dates=dates, strategies=None,
                    combinations=combinations,
                    add_next_best=True,
                    use_parallel_compute=use_parallel_compute,
                    save_returns='log_return',
                    save_summary=True,
                    routine=routine)

        # Select specific instances of strategies with:
        # PSR >= 0.95 (will make money)
        # Average PSR across parameter space >= 0.9 (robust against randomness in parameter selection)
        # PBO <= 0.35 (consistently use better parameters)
        case 'validation_phase':
            wf_algo(symbols=symbols, dates=dates, strategies=strategies_list,
                    combinations=combinations,
                    add_next_best=False,
                    use_parallel_compute=use_parallel_compute,
                    save_returns='log_return',
                    save_summary=True,
                    routine=routine)

        case 'validation_phase_ave_psr':
            summary_df, _ = train_algo(symbols=symbols, dates=dates, strategies=strategies_list,
                                       combinations=combinations,
                                       use_parallel_compute=use_parallel_compute,
                                       save_returns='log_return',
                                       save_summary=False,
                                       routine=routine)
            add_ave_psr(summary_df, ['profit', 'psr'])

        # Select specific instances of strategies with:
        # PSR >= 0.90 (will make money)
        # Average PSR across parameter space >= 0.8 (robust against randomness in parameter selection)
        # PBO <= 0.50 (consistently use better parameters)
        case 'initial_selection':
            wf_algo(symbols=symbols, dates=dates, strategies=strategies_list,
                    combinations=combinations,
                    add_next_best=False,
                    use_parallel_compute=use_parallel_compute,
                    save_returns='log_return',
                    save_summary=True,
                    routine=routine)

        # Average PSR across parameter space >= 0.8 (robust against randomness in parameter selection)
        case 'initial_selection_ave_psr':
            summary_df, _ = train_algo(symbols=symbols, dates=dates, strategies=strategies_list,
                                       combinations=combinations,
                                       use_parallel_compute=use_parallel_compute,
                                       save_returns='log_return',
                                       save_summary=False,
                                       routine=routine)
            add_ave_psr(summary_df, ['profit', 'psr'])

        # Adds complexity and selection bias.
        # Initial thoughts: don't worry about parameter counts. Worry more about properly scaling parameter range.
        # Base of 2 scaling is simpler.
        case 'parameter_count':
            final_summary_df: pl.DataFrame | None = None
            for override_routine in ['less', 'usual', 'more']:
                summary_df, summary_path = wf_algo(symbols=symbols, dates=dates, strategies=strategies_list,
                                                   combinations=combinations,
                                                   add_next_best=False,
                                                   use_parallel_compute=use_parallel_compute,
                                                   save_returns=None,
                                                   save_summary=False,
                                                   routine=override_routine)
                summary_df = summary_df.with_columns(pl.lit(override_routine).alias('parameter_count'))
                final_summary_df = summary_df if final_summary_df is None else final_summary_df.vstack(summary_df)
            if final_summary_df is None:
                raise TypeError(f'final_summary_df is None. Loop was not able to change the initial value.')
            final_summary_df.write_csv(summary_path)

        # Use the mean and median psr values of each parameter to find the appropriate range to use.
        case 'parameter_range':
            summary_df, summary_path = train_algo(symbols=symbols, dates=dates, strategies=strategies_list,
                                                  combinations=combinations,
                                                  use_parallel_compute=use_parallel_compute,
                                                  save_returns=None,
                                                  save_summary=True,
                                                  routine=routine)
            summary_df = normalize_metric(summary_df, summary_path,
                                          ['symbol', 'strategy', 'date_to'], metric='psr')
            add_parameter_statistics(summary_df, summary_path, ['psr', 'normalized_psr'])

        case 'custom':
            wf_algo(symbols=symbols, dates=dates, strategies=strategies_list,
                    combinations=combinations,
                    add_next_best=False,
                    use_parallel_compute=use_parallel_compute,
                    save_returns='log_return',
                    save_summary=True,
                    routine=routine,
                    use_synthetic=True)
    return


@dtf.time_it
def train_algo(symbols: list[str] | None = None,
               dates: list[tuple[datetime, datetime]] | None = None,
               strategies: list[ModuleType] | None = None,
               combinations: list[list[Any]] | None = None,
               use_parallel_compute: bool = True,
               save_returns: str | None = None,
               save_summary: bool = True,
               routine: str | None = None,
               use_synthetic: bool = False) -> tuple[pl.DataFrame, str]:
    """
    Trains algorithms using given symbols, dates, strategies, and parameter combinations.

    Parameters:
    symbols (Optional[List[str]]): List of symbols to optimize.
    dates (Optional[List[Tuple[datetime, datetime]]]): List of date ranges for training and testing.
    strategies (Optional[List[ModuleType]]): List of strategy modules to use.
    combinations (Optional[List[List[Any]]]): Predefined combinations of parameters.
    use_parallel_compute (bool): Whether to use parallel computation.
    save_returns (Optional[str]): Type of returns to save ('log_return' or 'pct_return').
    save_summary (bool): Whether to save the summary to a file.
    routine (Optional[str]): Routine for parameter generation.

    Returns:
    Tuple[pl.DataFrame, str]: The summary DataFrame and the path to the saved summary file.
    """
    if combinations is None:
        if symbols is None or dates is None or strategies is None:
            raise ValueError(f'Incorrect argument parameters for train_algo.')
        all_combinations: list[list[Any]] = get_train_combinations(symbols, dates, strategies,
                                                                   add_parameters=True, routine=routine)
    else:
        if dates is None:
            raise TypeError(f'dates is None. Combination needs dates to be passed.')
        combinations = add_dates_combinations(combinations, dates)
        all_combinations = add_parameter_combinations(combinations, routine=routine)
        mf.check_type_combination(all_combinations[0])

    headers: list[str] = ['symbol', 'strategy', 'date_from', 'date_to', 'parameter',
                          'profit', 'days_per_trade', 'trades_per_year', 'ave_drawdown',
                          'max_drawdown', 'omega', 'sortino', 'annual_sharpe', 'psr']

    # Parallel compute
    if use_parallel_compute:
        results_tuple: list[tuple[list[Any], pl.DataFrame | None]] = (
            Parallel(n_jobs=-1)(delayed(train_algo_subprocess)(*combination, headers=headers, save_returns=save_returns,
                                                               use_synthetic=use_synthetic)
                                for combination in tqdm(all_combinations)))
    # Serial compute
    else:
        results_tuple = [train_algo_subprocess(symbol, date_from, date_to, strategy, parameter,
                                               headers=headers, save_returns=save_returns,
                                               use_synthetic=use_synthetic)
                         for symbol, date_from, date_to, strategy, parameter in tqdm(all_combinations)]
    results: tuple[Any]
    dfs: tuple[pl.DataFrame]
    results, dfs = zip(*results_tuple)
    if save_returns is not None:
        save(all_combinations, dfs, group_by='symbols_dates_strategies', save_returns=save_returns)

    results, headers = mf.unpack_parameters(results, headers)

    # Transpose the result to match with headers
    transposed_results: list[list[Any]] = list(map(mf.to_list, zip(*results)))

    # Convert list parameters into strings
    data_dict: dict[str, Any] = {header: values for header, values in zip(headers, transposed_results)}
    data_dict = convert_list_columns(data_dict)

    summary_df: pl.DataFrame = pl.DataFrame(data_dict)

    # Save summary into a file
    date_now: str = f'{datetime.now():%b%d_%Hh%Mm%Ss}'
    summary_path: str = path.from_project(f'results\\train {date_now}.csv')
    if save_summary:
        summary_df.write_csv(summary_path)

    return summary_df, summary_path


def train_algo_subprocess(symbol: str, date_from: datetime, date_to: datetime,
                          strategy: ModuleType, parameter: dict[str, Any], headers: list[str],
                          save_returns: str | None = None,
                          use_synthetic: bool = False) -> tuple[list[Any], pl.DataFrame | None]:
    """
    Subprocess for training a single algorithm combination.

    Parameters:
    symbol (str): The trading symbol.
    date_from (datetime): The start date for the training period.
    date_to (datetime): The end date for the training period.
    strategy (ModuleType): The trading strategy module.
    parameter (dict[str, Any]): The parameters for the strategy.
    headers (List[str]): The list of headers for performance metrics.

    Returns:
    Tuple[List[Any], pl.DataFrame]: The performance metrics and the DataFrame of results.
    """
    df, metadata = backtest(symbol, date_from, date_to, strategy, parameter, use_synthetic=use_synthetic)
    perf_metrics = get_performance(df, metadata, headers)

    # # Add progress bar update
    # print('\u2588', end='')
    if save_returns is not None:
        return perf_metrics, df.slice(offset=1).select(['timestamp', save_returns])
    return perf_metrics, None


@dtf.time_it
def wf_algo(symbols: list[str] | None = None,
            dates: list[tuple[datetime, datetime]] | None = None,
            strategies: list[ModuleType] | None = None,
            combinations: list[list[Any]] | None = None,
            add_next_best: bool = True,
            use_parallel_compute: bool = True,
            save_returns: str | None = None,
            save_summary: bool = True,
            routine: str | None = None,
            use_synthetic: bool = False) -> tuple[pl.DataFrame, str]:
    """
    Performs walk-forward optimization on given symbols, dates, and strategies.

    Parameters:
    symbols (Optional[List[str]]): List of symbols to optimize.
    dates (Optional[List[Tuple[datetime, datetime]]]): List of date ranges for training and testing.
    strategies (Optional[List[ModuleType]]): List of strategy modules to use.
    combinations (Optional[List[List[Any]]]): Predefined combinations of parameters.
    add_next_best (bool): Whether to add the next best parameter for live trading.
    use_parallel_compute (bool): Whether to use parallel computation.
    save_returns (Optional[str]): Type of returns to save ('log_return' or 'pct_return').
    save_summary (bool): Whether to save the summary to a file.
    routine (Optional[str]): Routine for parameter generation.

    Returns:
    Tuple[pl.DataFrame, str]: The summary DataFrame and the path to the saved summary file.
    """
    if combinations is None:
        if symbols is None or dates is None or strategies is None:
            raise ValueError(f'Incorrect argument parameters for train_algo.')
        all_combinations: list[list[Any]] = get_train_combinations(symbols, dates, strategies, add_parameters=False)
    else:
        if dates is None:
            raise TypeError(f'dates is None. Combination needs dates to be passed.')
        all_combinations = add_dates_combinations(combinations, dates)
        mf.check_type_combination(all_combinations[0])

    headers: list[str] = ['symbol', 'strategy', 'date_from', 'date_to', 'parameter',
                          'profit', 'days_per_trade', 'trades_per_year', 'ave_drawdown',
                          'max_drawdown', 'omega', 'sortino', 'annual_sharpe', 'psr']

    # finds the best parameter for live trading
    if add_next_best:
        headers.insert(headers.index('parameter') + 1, 'next_best_param')

    # Progress Bar:
    # print('Total Iterations:')
    # print('\u2588' * len(all_combinations))
    # print('Processed Iterations:')

    if use_parallel_compute:
        results_tuple: list[tuple[list[Any], pl.DataFrame | None]] = \
            (Parallel(n_jobs=-1)(delayed(wf_algo_subprocess)
                                 (symbol, date_from, date_to, strategy,
                                  headers=headers, routine=routine,
                                  save_returns=save_returns, use_synthetic=use_synthetic)
                                 for symbol, date_from, date_to, strategy in tqdm(all_combinations)))
    else:
        results_tuple = [wf_algo_subprocess(symbol, date_from, date_to, strategy,
                                            headers=headers, routine=routine,
                                            save_returns=save_returns, use_synthetic=use_synthetic)
                         for symbol, date_from, date_to, strategy in tqdm(all_combinations)]

    results, dfs = zip(*results_tuple)

    # print('')
    if save_returns is not None:
        save(all_combinations, dfs, group_by='no_group', save_returns=save_returns)

    # Transpose the result to match with headers
    transposed_results: list[list[Any]] = list(map(mf.to_list, zip(*results)))

    # Convert list parameters into strings
    index_of_param: int = headers.index('parameter')
    headers[index_of_param] = 'used_parameter'
    data_dict: dict[str, Any] = {header: values for header, values in zip(headers, transposed_results)}
    data_dict = convert_list_columns(data_dict, 'used_parameter')

    summary_df: pl.DataFrame = pl.DataFrame(data_dict)

    # Save summary into a file
    date_now: str = f'{datetime.now():%b%d_%Hh%Mm%Ss}'
    summary_path: str = path.from_project(f'results\\wf {date_now}.csv')
    if save_summary:
        summary_df.write_csv(summary_path)

    return summary_df, summary_path


def wf_algo_subprocess(symbol: str, date_from: datetime, date_to: datetime,
                       strategy: ModuleType, headers: list[str],
                       routine: str | None = None,
                       save_returns: str | None = None,
                       use_synthetic: bool = False) -> tuple[list[Any], pl.DataFrame | None]:
    """
    Subprocess for walk-forward optimization.

    Parameters:
    symbol (str): The trading symbol.
    date_from (datetime): The start date for the training period.
    date_to (datetime): The end date for the testing period.
    strategy (ModuleType): The trading strategy module.
    headers (List[str]): The list of headers for performance metrics.
    routine (Optional[str]): Routine for parameter generation.

    Returns:
    Tuple[List[Any], pl.DataFrame]: The performance metrics and the combined DataFrame of results.
    """
    wf_dates: list[tuple[datetime, datetime, datetime, datetime]] = dtf.walk_forward_dates(date_from,
                                                                                           date_to)
    df_combined: pl.DataFrame | None = None
    metadata_combined: dict[str, Any] = {}
    start_equity: float = 100_000.0

    for date_train_from, date_train_to, date_test_from, date_test_to in wf_dates:
        # Train Set
        best_param: dict[str, Any] = get_best_param(symbol, date_train_from, date_train_to, strategy,
                                                    optimization_criteria='psr', routine=routine,
                                                    use_synthetic=use_synthetic)

        # Test Set
        df, metadata = backtest(symbol, date_test_from, date_test_to, strategy, best_param, start_equity=start_equity,
                                use_synthetic=use_synthetic)
        df_combined, metadata_combined = (df, metadata) if df_combined is None \
            else wf_combine_df(df_combined, metadata_combined, df, metadata)

        start_equity = df.item(-1, 'equity')

    if df_combined is None:
        raise TypeError('df_combined is None. Loop was not able to change the initial value.')

    perf_metrics: list[Any] = get_performance(df_combined, metadata_combined, headers)
    if 'next_best_param' in headers:
        index: int = headers.index('next_best_param')
        date_next_best_from, date_next_best_to = dtf.next_best_param_dates(date_to)
        parameter: dict[str, Any] = get_best_param(symbol, date_next_best_from, date_next_best_to, strategy,
                                                   optimization_criteria='psr', routine=routine,
                                                   use_synthetic=use_synthetic)
        str_parameter: str = ','.join(f'{key_param}:{value_param}' for key_param, value_param in parameter.items())
        perf_metrics.insert(index, str_parameter)
    # # Add progress bar update
    # print('\u2588', end='')
    if save_returns is not None:
        return perf_metrics, df_combined.slice(offset=1).select(['timestamp', save_returns])
    return perf_metrics, None


def convert_list_columns(data_dict: dict[str, Any], starts_with: str | list[str] = 'P:') -> dict[str, Any]:
    for key, values in data_dict.items():
        if isinstance(starts_with, list):
            for starts_with_a in starts_with:
                if not key.startswith(starts_with_a):
                    continue
        elif isinstance(starts_with, str):
            if not key.startswith(starts_with):
                continue
        else:
            raise TypeError(f'convert_list_columns parameter expected type list or str but got {starts_with}.')
        new_values: list = []
        for value in values:
            if isinstance(value, list):
                if len(value) == 1:
                    try:
                        new_value = float(value[0])
                    except ValueError:
                        new_value = '::' + str(value).strip('[]').replace(' ', '')
                else:
                    new_value = '::' + str(value).strip('[]').replace(' ', '')
                new_values.append(new_value)
            else:
                new_values.append(value)
        data_dict[key] = [str(row).zfill(5) if not isinstance(row, str) and row is not None else row
                          for row in new_values]
    return data_dict


def add_ave_psr(summary_df: pl.DataFrame, metrics: list[str]) -> None:
    strategies: list[tuple] = list(summary_df.unique(subset=['symbol', 'strategy'])
                                   .select(['symbol', 'strategy']).iter_rows())
    dates: list[tuple] = list(summary_df.unique(subset=['date_from', 'date_to'])
                              .select(['date_from', 'date_to']).iter_rows())
    combinations: list = list(iter_product(strategies, dates))
    results: list = []
    for (symbol, strategy), (date_from, date_to) in combinations:
        df_per_group: pl.DataFrame = summary_df.filter((pl.col('strategy') == strategy) &
                                                       (pl.col('symbol') == symbol) &
                                                       (pl.col('date_from') == date_from) &
                                                       (pl.col('psr').is_not_nan()))
        result = {'symbol': symbol, 'strategy': strategy,
                  'date_from': date_from, 'date_to': date_to}
        for metric in metrics:
            result[f'mean_{metric}'] = df_per_group[metric].mean()  # type: ignore
            result[f'median_{metric}'] = df_per_group[metric].median()  # type: ignore
        results.append(result)
    grouped_df: pl.DataFrame = pl.DataFrame(results)
    grouped_df.write_csv(path.from_project('results\\ave_psr.csv'))
    return


def add_parameter_statistics(summary_df: pl.DataFrame, summary_path: str, metrics: list[str]):
    """
    Adds group statistics (mean and median) for specified metrics to a summary DataFrame.

    Parameters:
    summary_df (pl.DataFrame): The input summary DataFrame.
    summary_path (str): The file path to save the updated DataFrame.
    metrics (List[str]): The list of metrics to calculate statistics for.

    Returns:
    Tuple[pl.DataFrame, str]: The updated summary DataFrame and the file path where it was saved.
    """
    parameter_columns: list[Any] = [col for col in summary_df.columns if col.startswith('P:')]
    strategies: list[str] = summary_df['strategy'].unique().to_list()
    results: list[dict[str, Any]] = []

    # Sort function
    def sort_key(x):
        sort_value = x['value']
        if isinstance(sort_value, str) and sort_value.startswith('::'):
            # Extract the number after '::' and before the comma
            number_str = sort_value.lstrip('::').split(',')[0]
            return float(number_str)
        else:
            return sort_value

    for strategy in strategies:
        df_per_strategy: pl.DataFrame = summary_df.filter(pl.col('strategy') == strategy)
        for parameter_column in parameter_columns:
            l_stripped: str = parameter_column.lstrip('P:')
            unique_values: list[Any] = df_per_strategy[parameter_column].unique().drop_nulls().to_list()

            unique_values_result: list[dict[str, Any]] = []
            for value in unique_values:
                df_per_value: pl.DataFrame = df_per_strategy.filter((pl.col(parameter_column) == value) &
                                                                    (pl.col(metrics[0]).is_not_nan()))
                for metric in metrics[1:]:
                    df_per_value = df_per_value.filter(pl.col(metric).is_not_nan())

                if not df_per_value.is_empty():
                    result: dict[str, str | float | int] = {
                        'strategy': strategy,
                        'parameter': l_stripped,
                        'value': float(value) if not isinstance(value, (int, str)) else value
                    }
                    for metric in metrics:
                        result[f'trades_per_year'] = df_per_value['trades_per_year'].mean()  # type: ignore
                        result[f'sample_size'] = df_per_value['trades_per_year'].count()  # type: ignore
                        result[f'mean_{metric}'] = df_per_value[metric].mean()  # type: ignore
                        result[f'median_{metric}'] = df_per_value[metric].median()  # type: ignore
                    unique_values_result.append(result)

            # sort by value and add to result
            results.extend(sorted(unique_values_result, key=sort_key))
        results.append({})  # add space per strategy

    grouped_df: pl.DataFrame = pl.DataFrame(results)
    date_now: str = f'{datetime.now():%b%d_%Hh%Mm%Ss}'
    grouped_df.write_csv(path.join(path.get_directory(summary_path), f'summary {date_now}.csv'))

    # Renaming columns to prepend '::' to column names
    # rename_dict: dict[str, str] = {col: f'::{col}' for col in grouped_df.columns}
    # grouped_df = grouped_df.rename(rename_dict)

    # Concatenating the original DataFrame with the new grouped statistics DataFrame
    # summary_df = pl.concat([summary_df, grouped_df], how='horizontal')

    # Writing the resulting DataFrame to CSV
    # summary_df.write_csv(summary_path)

    return summary_df, summary_path


def normalize_metric(df: pl.DataFrame, file_path: str, group_cols: list[str], metric: str = 'psr') -> pl.DataFrame:
    """
    Normalizes a specified metric within groups and saves the result to a file.

    Parameters:
    df (pl.DataFrame): The input DataFrame.
    file_path (str): The file path to save the normalized DataFrame.
    group_cols (List[str]): The columns to group by for normalization.
    metric (str): The metric to normalize (default is 'psr').

    Returns:
    pl.DataFrame: The DataFrame with the normalized metric.
    """
    # Calculate the mean and standard deviation for each group
    df_drop_nan: pl.DataFrame = df.filter(pl.col(metric).is_not_nan())
    mean_column: str = f'mean_{metric}'
    stdev_column: str = f'stdev_{metric}'
    mean_df: pl.DataFrame = df_drop_nan.group_by(group_cols).agg(pl.col(metric).mean().alias(mean_column))
    std_df: pl.DataFrame = df_drop_nan.group_by(group_cols).agg(pl.col(metric).std().alias(stdev_column))

    # Join the mean and standard deviation back to the original DataFrame
    df = df.join(mean_df, on=group_cols, how='left', coalesce=True)
    df = df.join(std_df, on=group_cols, how='left', coalesce=True)

    # Normalize the 'psr' values
    normalized_metric: str = f'normalized_{metric}'
    df = df.with_columns(pl.when(pl.col(mean_column) == 0).then(-100.0)
                         .otherwise(pl.when(pl.col(stdev_column) == 0).then(np.nan)
                                    .otherwise((pl.col(metric) - pl.col(mean_column)) / pl.col(stdev_column)))
                         .alias(normalized_metric))
    df = df.drop([mean_column, stdev_column])
    df.write_csv(file_path)

    return df


def ensure_hstack(df_main: pl.DataFrame, df_other: pl.DataFrame) -> pl.DataFrame:
    # ensure dfs have the same height
    row_diff: int = len(df_main) - len(df_other)
    if row_diff > 0:
        zeros: pl.DataFrame = pl.DataFrame({col: [0.0] * row_diff for col in df_other.columns})
        df_other = zeros.vstack(df_other)
    elif row_diff < 0:
        raise ValueError('Improper stacking of dataframes. Larger dataframe should be the first parameter.')
    return df_main.hstack(df_other)


def save(all_combinations: list[list[Any]],
         dfs: list[pl.DataFrame] | tuple[pl.DataFrame],
         group_by: str = 'symbols_dates_strategies',
         save_returns: str = 'log_return') -> None:
    """
    Saves dataframes grouped by combinations to CSV files.

    Parameters:
    all_combinations (List[List[Any]]): A list of parameter combinations.
    dfs (Union[List[pl.DataFrame], Tuple[pl.DataFrame]]): A list or tuple of dataframes to be saved.
    group_by (str): The grouping criterion ('symbols_dates_strategies' or 'no_group').
    save_returns (str): The type of returns to save ('log_return' or 'pct_return').

    Raises:
    ValueError: If `save_returns` is not 'log_return' or 'pct_return'.
    """
    if save_returns != 'log_return' and save_returns != 'pct_return':
        warn(f'Cannot save file. Expected save_returns = log_return or pct_return. Got {save_returns} instead')

    # Filters based on symbols_dates_strategies, then saves each group into a file
    if group_by == 'symbols_dates_strategies':
        grouped_dfs: dict[Any, list[tuple[dict[str, Any], pl.DataFrame]]] = {}
        for index, (symbol, date_from, date_to, strategy, parameter) in enumerate(all_combinations):
            key = symbol, date_from, date_to, strategy.__name__.split('.')[1]
            if key not in grouped_dfs:
                grouped_dfs[key] = []
            grouped_dfs[key].append((parameter, dfs[index]))

        for key, value in grouped_dfs.items():
            symbol, date_from, date_to, strategy = key
            max_length: int = 0
            max_df_index: int = 0
            for index, (_, df) in enumerate(value):
                len_df: int = len(df)
                if len_df > max_length:
                    max_length = len_df
                    max_df_index = index
            df_to_save: pl.DataFrame = value[max_df_index][1].drop(save_returns)
            for parameter, df in value:
                rename_column: str = ','.join(f'{key_param}:{value_param}'
                                              for key_param, value_param in parameter.items())
                df_to_stack: pl.DataFrame = df.rename({save_returns: rename_column}).drop('timestamp')
                df_to_save = ensure_hstack(df_to_save, df_to_stack)

            df_to_save.write_csv(
                path.from_project(
                    f'results\\{save_returns}\\train {symbol} {strategy} {date_from:%b_%Y} to {date_to:%b_%Y}.csv'))
        return

    # All combination returns are saved in one csv. This is for walk-forward results.
    elif group_by == 'no_group':
        df_to_save: pl.DataFrame = max(dfs, key=lambda x: len(x)).drop(save_returns)
        date_from_save: datetime = all_combinations[0][1]
        date_to_save: datetime = all_combinations[0][2]
        for (symbol, date_from, date_to, strategy), df in zip(all_combinations, dfs):
            strategy = strategy.__name__.split('.')[1]
            rename_column = f'{symbol}:{strategy}'
            df_to_stack: pl.DataFrame = df[save_returns].rename(rename_column).to_frame()
            df_to_save = ensure_hstack(df_to_save, df_to_stack)  # type: ignore

        df_to_save.write_csv(
            path.from_project(f'results\\{save_returns}\\wf {date_from_save:%b_%Y} to {date_to_save:%b_%Y}.csv'))
        return

    else:
        mf.warn_errors(f'Cannot save file. Expected group_by = symbols_dates_strategies or no_group. '
                       f'Got {group_by} instead.')
    return


def wf_combine_df(df_1: pl.DataFrame, metadata_1: dict[str, Any],
                  df_2: pl.DataFrame, metadata_2: dict[str, Any]) -> tuple[pl.DataFrame, dict[str, Any]]:
    """
    Combines two dataframes and their metadata, ensuring they are continuous in time.

    Parameters:
    df_1 (pl.DataFrame): First dataframe.
    metadata_1 (Dict[str, Any]): Metadata for the first dataframe.
    df_2 (pl.DataFrame): Second dataframe.
    metadata_2 (Dict[str, Any]): Metadata for the second dataframe.

    Returns:
    Tuple[pl.DataFrame, Dict[str, Any]]: Combined dataframe and combined metadata.

    Raises:
    ValueError: If the dataframes are not continuous.
    """
    if metadata_1['date_to'] >= metadata_2['date_from']:
        mf.warn_errors(metadata_1['symbol'], metadata_1['strategy'], metadata_1['date_from'],
                       metadata_1['date_to'], metadata_1['parameter'])
        mf.warn_errors(metadata_2['symbol'], metadata_2['strategy'], metadata_2['date_from'],
                       metadata_2['date_to'], metadata_2['parameter'])
        raise ValueError('Cannot concat overlapping dataframes.')

    tf: str = metadata_1['timeframe']
    df_combined: pl.DataFrame = pl.concat([df_1, df_2.slice(offset=1)])
    if df_1.item(-1, 'timestamp') != df_2.item(0, 'timestamp'):
        warn(f'Indicator buffer size insufficient for {metadata_1['strategy']}.')
        df_combined = df_combined.set_sorted(column='timestamp').upsample(time_column='timestamp', every=tf)
        df_combined = df_combined.with_columns(pl.col('pct_return').fill_null(strategy='zero'),
                                               pl.col('log_return').fill_null(strategy='zero'),
                                               pl.col('equity').fill_null(strategy='forward')
                                               )

    date_from: datetime = metadata_1['date_from']
    date_to: datetime = metadata_2['date_to']
    start_equity: float = metadata_1['start_equity']

    meta_1_param: list[dict[str, Any]] | dict[str, Any] = metadata_1['parameter']
    meta_2_param: dict[str, Any] = metadata_2['parameter']
    if isinstance(meta_1_param, list):
        meta_1_param.append(meta_2_param)
    else:
        meta_1_param = [meta_1_param, meta_2_param]

    return df_combined, {
        'date_from': date_from,
        'date_to': date_to,
        'slippage': metadata_1['slippage'],
        'timeframe': tf,
        'symbol': metadata_1['symbol'],
        'strategy': metadata_1['strategy'],
        'parameter': meta_1_param,
        'number_of_years': (date_to - date_from).total_seconds() / 31_556_952,
        'risk_per_trade': metadata_1['risk_per_trade'],
        'taker': metadata_1['taker'],
        'maker': metadata_1['maker'],
        'funding': metadata_1['funding'],
        'start_equity': start_equity,
        'total_days_traded': metadata_1['total_days_traded'] + metadata_2['total_days_traded'],
        'number_of_trades': metadata_1['number_of_trades'] + metadata_2['number_of_trades'],
        'total_pct_gain': df_combined.item(-1, 'equity') / start_equity - 1.0,
    }


def add_dates_combinations(combinations: list[list[Any]], dates: list[tuple[datetime, datetime]]) -> list[list[Any]]:
    """
    Adds date ranges to the list of combinations.

    Parameters:
    combinations (List[List[Any]]): The list of combinations (without dates).
    dates (List[Tuple[datetime, datetime]]): The list of date ranges to add.

    Returns:
    List[List[Any]]: A new list of combinations with date ranges included.
    """
    return [[symbol, date_from, date_to, strategy] for date_from, date_to in dates
            for symbol, strategy in combinations]


def add_parameter_combinations(combinations: list[list[Any]], routine: str | None = None) -> list[list[Any]]:
    """
    Adds parameter combinations to the list of combinations.

    Parameters:
    combinations (List[List[Any]]): The list of combinations (without parameters).
    routine (Optional[str]): Optional routine parameter for the strategy's parameters function.

    Returns:
    List[List[Any]]: A new list of combinations with parameters included.
    """
    all_combinations: list[Any] = []
    for symbol, date_from, date_to, strategy in combinations:
        parameters: list[dict[str, Any]] = strategy.parameters(routine)
        all_combinations.extend([symbol, date_from, date_to, strategy, parameter] for parameter in parameters)
    return all_combinations


def get_train_combinations(symbols: list[str], dates: list[tuple[datetime, datetime]],
                           strategies_list: list[ModuleType],
                           add_parameters: bool = True,
                           routine: str | None = None) -> list[list[Any]]:
    """
    Generates all possible training combinations of symbols, dates, and strategies, with optional parameters.

    Parameters:
    symbols (List[str]): List of symbols for backtesting.
    dates (List[Tuple[datetime, datetime]]): List of date ranges for backtesting.
    strategies_list (List[ModuleType]): List of strategy modules.
    add_parameters (bool): Whether to include strategy parameters in the combinations. Defaults to True.
    routine (Optional[str]): Optional routine parameter for the strategy's parameters function.

    Returns:
    List[List[Any]]: A list of all possible combinations for training.
    """
    all_combinations: list[Any] = []

    if add_parameters and routine:
        for strategy in strategies_list:
            parameters: list[dict[str, Any]] = strategy.parameters(routine)
            combinations: list = list(iter_product([strategy], dates, symbols, parameters))
            all_combinations.extend(combinations)
        expanded_combinations = [[symbol, *dates, strategy, parameter] for
                                 strategy, dates, symbol, parameter in all_combinations]
    else:
        all_combinations = list(iter_product(strategies_list, dates, symbols))
        expanded_combinations = [[symbol, *dates, strategy] for strategy, dates, symbol in all_combinations]

    mf.check_type_combination(expanded_combinations[0])

    return expanded_combinations


def get_best_param(symbol: str, date_from: datetime, date_to: datetime, strategy: ModuleType,
                   optimization_criteria: str = 'psr', routine: str | None = None,
                   use_synthetic: bool = False) -> dict[str, Any]:
    """
    Finds the best parameters for a given strategy based on an optimization criterion.

    Parameters:
    symbol (str): The symbol for which to run the backtest.
    date_from (datetime): The start date of the backtest period.
    date_to (datetime): The end date of the backtest period.
    strategy (ModuleType): The trading strategy module.
    optimization_criteria (str): The performance metric to optimize for. Defaults to 'psr'.
    routine (Optional[str]): Optional routine parameter for the strategy's parameters function.

    Returns:
    Dict[str, Any]: The best parameters based on the optimization criterion.
    """
    highest_perf_metrics: float = float('-inf')
    best_param: dict[str, Any] = {}

    # Get parameters to evaluate from the strategy
    parameters: list[dict[str, Any]] = strategy.parameters(routine)

    # Skips backtesting if there are no alternative parameters
    if len(parameters) == 1:
        return parameters[0]

    for parameter in parameters:
        df, metadata = backtest(symbol, date_from, date_to, strategy, parameter, use_synthetic=use_synthetic)
        perf_metrics: float = get_performance(df, metadata, [optimization_criteria])[0]

        if perf_metrics > highest_perf_metrics:
            best_param = parameter
            highest_perf_metrics = perf_metrics

    if len(best_param) == 0:
        with open(f'Logger.txt', 'a') as file:
            message: str = (f'{datetime.now()}\n'
                            f'Error: No best_param found.\n'
                            f'Used last param {parameters[-1]} for {symbol} {strategy.__name__} {date_from} {date_to}')
            file.write(message + '\n')
            file.write('----------------------------------------------\n')
        return parameters[-1]
    return best_param


def get_performance(df: pl.DataFrame, metadata: dict[str, Any], metrics: list[str]) -> list[Any]:
    """
    Calculate performance metrics for the given DataFrame and metadata.

    Parameters:
    df (pl.DataFrame): The DataFrame containing market data with performance calculations.
    metadata (dict[str, Any]): Metadata dictionary with backtest information.
    metrics (List[str]): List of performance metrics to calculate.

    Returns:
    List[Any]: A list containing the calculated performance metrics.

    Raises:
    Exception: If an error occurs during the performance metric calculations.
    """

    # Initialize required variables
    number_of_trades: int = metadata['number_of_trades']
    log_return: np.ndarray = df['log_return'].slice(offset=1).to_numpy()
    mean_return: float = log_return.mean()
    std_of_return: float = log_return.std()
    sharpe: float = mean_return / std_of_return if number_of_trades > 0 else np.nan
    sqrt_n_per_year: float = np.sqrt(len(log_return) / metadata['number_of_years'])

    dd_series: pl.Series | None = (1.0 - df['equity'] / df['equity'].cum_max()).slice(offset=1) \
        if 'ave_drawdown' in metrics or 'max_drawdown' in metrics else None

    performance: list[Any] = []
    for metric in metrics:
        if metric == 'next_best_param':
            continue

        if metric in metadata:
            performance.append(metadata[metric]
                               if isinstance(metadata[metric], (float, int, str)) or metric == 'parameter'
                               else str(metadata[metric]))
            continue

        match metric:
            case 'profit':
                performance.append(df.item(-1, 'equity') - metadata['start_equity'])
            case 'trades_per_year':
                performance.append(number_of_trades / metadata['number_of_years'])
            case 'days_per_trade':
                performance.append(metadata['total_days_traded'] / number_of_trades if number_of_trades > 0 else np.nan)
            case 'ave_drawdown':
                performance.append(dd_series.mean() if dd_series is not None else np.nan)
            case 'max_drawdown':
                performance.append(dd_series.max() if dd_series is not None else np.nan)
            case 'mean_return':
                performance.append(mean_return)
            case 'std_of_return':
                performance.append(std_of_return)
            case 'annual_std':
                performance.append(std_of_return * sqrt_n_per_year)
            case 'sharpe':
                performance.append(sharpe)
            case 'annual_sharpe':
                performance.append(sharpe * sqrt_n_per_year)
            case 'omega':
                if number_of_trades == 0:
                    performance.append(np.nan)
                    continue
                lpm: float = np.power(np.maximum((-log_return), 0), 1).mean()
                if lpm == 0:
                    mf.warn_errors('Omega calculation error where lpm == 0.')
                    mf.warn_errors(metadata['symbol'], metadata['strategy'], metadata['date_from'],
                                   metadata['date_to'], metadata['parameter'])
                    performance.append(np.nan)
                    continue
                performance.append(1.0 + mean_return / lpm)
            case 'sortino':
                if number_of_trades == 0:
                    performance.append(np.nan)
                    continue
                lpm = np.power(np.maximum((-log_return), 0), 2).mean()
                if lpm == 0:
                    mf.warn_errors('Sortino calculation error where lpm == 0.')
                    mf.warn_errors(metadata['symbol'], metadata['strategy'], metadata['date_from'],
                                   metadata['date_to'], metadata['parameter'])
                    performance.append(np.nan)
                    continue
                performance.append(mean_return / np.sqrt(lpm))
            case 'psr':
                if number_of_trades == 0:
                    performance.append(np.nan)
                    continue
                target_sharpe: float = 0.0
                n: int = log_return.shape[0]
                try:
                    skew = ss.skew(log_return)
                    kurtosis = ss.kurtosis(log_return) + 3
                    value = ((sharpe - target_sharpe) * np.sqrt(n - 1) /
                             np.sqrt(1.0 - skew * sharpe + sharpe ** 2 * (kurtosis - 1) / 4.0))
                    performance.append(ss.norm.cdf(value, 0, 1))
                except Exception as e:
                    mf.warn_errors(f'PSR calculation error. {e}')
                    mf.warn_errors(metadata['symbol'], metadata['strategy'], metadata['date_from'],
                                   metadata['date_to'], metadata['parameter'])
                    performance.append(np.nan)
            case _:
                mf.warn_errors(f'Performance metric {metric} not implemented.')
                performance.append(np.nan)
    return performance


def backtest(symbol: str, date_from: datetime, date_to: datetime, strategy: ModuleType, parameter: dict[str, Any],
             start_equity: float = 100_000.0, use_synthetic: bool = False) -> tuple[pl.DataFrame, dict[str, Any]]:
    """
    Perform a backtest on a financial instrument using a given strategy module and strategy parameters.

    Function Parameters:
    symbol (str): The financial instrument symbol (e.g., 'BTCUSDT').
    date_from (datetime): The starting date of the backtest.
    date_to (datetime): The ending date of the backtest.
    strategy (module): The trading strategy module.
    parameter (dict[str, Any]): Parameters for the strategy indicators.
    start_equity (float): Initial equity for the backtest.

    Returns:
    Tuple[pl.DataFrame, dict[str, Any]]: A tuple containing the backtest results DataFrame and metadata.

    Raises:
    ValueError: If the date range is invalid or data processing fails.
    ImportError: If the strategy module is not found.
    Exception: For other unexpected errors.
    """
    # Fetch and process data
    df, metadata = get_data(symbol, date_from, date_to, use_synthetic=use_synthetic)
    df = strategy.indicators(df, parameter)
    initial_len: int = len(df)

    # Add metadata
    metadata['strategy'] = strategy.__name__.split('.')[-1]
    metadata['parameter'] = parameter

    # Drop nulls and readjusts date_from
    df = df.filter(pl.sum_horizontal(pl.col(list(pl.FLOAT_DTYPES)).is_nan()) == 0).drop_nulls()
    df = df.set_sorted('timestamp')
    if initial_len - len(df) > 512 + 128:
        warn(f'{metadata['strategy']} indicator with parameter {parameter} requires more than 512 buffer size.')
    timestamps_np: np.ndarray = df['timestamp'].to_numpy()
    date_from_index: int = max(1, np.searchsorted(timestamps_np, metadata['date_from'].timestamp() * 1e9))
    date_from = dtf.make_utc(df.item(date_from_index, 'timestamp'))
    metadata['date_from'] = date_from
    metadata['number_of_years'] = (metadata['date_to'] - date_from).total_seconds() / 31_556_952

    # Slice the DataFrame to include 1 row before date_from for the loop
    df = df.slice(offset=date_from_index - 1)

    # Run the trading strategy loop
    return run_loop(df, metadata, strategy.trade_logic, start_equity=start_equity, verbose=False,
                    use_synthetic=use_synthetic)


def run_loop(df: pl.DataFrame,
             metadata: dict[str, Any],
             strategy_logic: Callable,
             start_equity: float = 100_000.0,
             verbose: bool = False, use_synthetic: bool = False) -> tuple[pl.DataFrame, dict[str, Any]]:
    """
    Execute the backtesting loop over the provided DataFrame with the given strategy.

    Parameters:
    df (pl.DataFrame): The DataFrame containing market data.
    metadata (dict[str, Any]): Metadata dictionary with backtest information.
    strategy_logic (Callable): The trading strategy logic function.
    start_equity (float): Initial equity for the backtest.
    verbose (bool): If True, additional columns for detailed information are created.

    Returns:
    Tuple[pl.DataFrame, dict[str, Any]]: A tuple containing the updated DataFrame and metadata.

    Raises:
    Exception: If an error occurs during the backtesting loop.
    """

    # Default values
    if use_synthetic:
        risk_per_trade: float = 0.00001  # Synthetic data movement can be extreme so readjust risks
    else:
        risk_per_trade = 0.001
    taker: float = 0.0004
    maker: float = 0.0002
    funding: float = 0.0001

    # Parameters
    metadata['risk_per_trade'] = risk_per_trade
    metadata['taker'] = taker
    metadata['maker'] = maker
    metadata['funding'] = funding
    metadata['start_equity'] = start_equity
    symbol: str = metadata['symbol']
    strategy_name: str = metadata['strategy']
    parameter: dict[str, Any] = metadata['parameter']
    date_from: datetime = metadata['date_from']
    date_to: datetime = metadata['date_to']
    slippage: float = metadata['slippage']
    length: int = df.shape[0]
    sqrt_n_per_day: float = np.sqrt((length - 1) /
                                    ((date_to - date_from).total_seconds() / 86_400))
    col: dict[str, int] = {name: index for index, name in enumerate(df.columns)}

    # Create numpy arrays which will later be added into the dataframe
    position_array: np.ndarray = np.zeros(length, dtype=np.int8)
    balance_array: np.ndarray = np.zeros(length, dtype=float)
    equity_array: np.ndarray = np.zeros(length, dtype=float)
    balance_array[0] = start_equity
    equity_array[0] = start_equity

    if verbose:
        funding_array: np.ndarray | None = np.zeros(length, dtype=float)
        long_fee_array: np.ndarray | None = np.zeros(length, dtype=float)
        short_fee_array: np.ndarray | None = np.zeros(length, dtype=float)
        long_equity_change_array: np.ndarray | None = np.zeros(length, dtype=float)
        short_equity_change_array: np.ndarray | None = np.zeros(length, dtype=float)
        long_unrealized_array: np.ndarray | None = np.zeros(length, dtype=float)
        short_unrealized_array: np.ndarray | None = np.zeros(length, dtype=float)
        number_of_trades_array: np.ndarray | None = np.zeros(length, dtype=int)
        long_qty_array: np.ndarray | None = np.zeros(length, dtype=float)
        short_qty_array: np.ndarray | None = np.zeros(length, dtype=float)
    else:
        funding_array = None
        long_fee_array = None
        short_fee_array = None
        long_equity_change_array = None
        short_equity_change_array = None
        long_unrealized_array = None
        short_unrealized_array = None
        number_of_trades_array = None
        long_qty_array = None
        short_qty_array = None

    # Initialize variables outside loop
    balance: float = start_equity
    equity: float = start_equity
    position: int = 0
    long_qty: float = 0.0
    short_qty: float = 0.0
    long_unrealized: float = 0.0
    short_unrealized: float = 0.0
    prev_timestamp: datetime | None = None
    trade_start_long: datetime | None = None
    trade_start_short: datetime | None = None
    total_days_traded: float = 0.0
    number_of_trades: int = 0
    df_np: np.ndarray = df.to_numpy()
    persist: dict[str, Any] = {'count': 0, 'uptrend_entry': 0.0, 'downtrend_entry': 0.0,
                               'uptrend_exit': 0.0, 'downtrend_exit': 0.0}
    orders: list[Any] = strategy_logic(df_np[0, :], col, position, balance, parameter,
                                       risk_per_trade, sqrt_n_per_day, persist)
    prev_price: float = df.item(0, 'close')

    # Main loop
    for index, row in enumerate(df_np[1:, :], start=1):
        current_timestamp: datetime = datetime.fromtimestamp(row[col['timestamp']] / 1e9, timezone.utc)
        open_price: float = row[col['open']]
        close_price: float = row[col['close']]
        long_equity_change: float = 0.0
        short_equity_change: float = 0.0

        # Check if unhandled hedge.
        if position not in [-1, 0, 1]:
            if len(orders) != 0 and (orders[-1][0] == 'handler'):
                orders.pop()
            else:
                warn(f'Unhandled hedge. Position {position} but handler is not found at '
                     f'index {index} for symbol {symbol} and strategy {strategy_name} '
                     f'and parameter {parameter} for dates {date_from} to {date_to}')

        # Modify values based on orders
        if orders:
            # Check orders for correctness
            if not all(isinstance(order_type, str) and
                       isinstance(order_price, (float, int)) and
                       isinstance(order_qty, (float, int)) and
                       (order_price >= 0) and
                       (order_qty >= 0) for order_type, order_price, order_qty in orders):
                raise ValueError(f'strategy_logic return invalid value at '
                                 f'index {index} for symbol {symbol} and strategy {strategy_name} '
                                 f'and parameter {parameter} for dates {date_from} to {date_to}')

            # Sort orders by non-zero qty. Exit orders should be executed first.
            orders.sort(key=lambda x: x[2] != 0)
            filled_orders: list = fill_orders(orders, slippage,
                                              open_price, row[col['first']],
                                              row[col['second']], close_price)
        else:
            filled_orders = []

        # Include new market orders into the calculation of funding fees at open
        fund_long_qty: float = long_qty
        fund_short_qty: float = short_qty
        for order_type, order_price, order_qty, taker_maker, executed_at in filled_orders:
            if (executed_at == 'at_open') and (order_qty > 0):
                if order_type == 'long_entry':
                    fund_long_qty += order_qty
                elif order_type == 'short_entry':
                    fund_short_qty += order_qty

        # Deduct funding fee at open
        if fund_long_qty > 0.0 or fund_short_qty > 0.0:
            if prev_timestamp is None:
                current_hour: int = current_timestamp.hour
                if current_hour in {0, 8, 16}:
                    long_fee = funding * open_price * fund_long_qty
                    short_fee = funding * open_price * fund_short_qty
                    if verbose and (funding_array is not None):
                        funding_array[index] += long_fee + short_fee
                    long_equity_change -= long_fee
                    long_unrealized -= long_fee
                    short_equity_change -= short_fee
                    short_unrealized -= short_fee
                    prev_timestamp = current_timestamp
            else:
                # 28_800 seconds = 8 hour
                elapsed_8hours: int = int((current_timestamp - prev_timestamp).total_seconds() / 28_800)
                if elapsed_8hours >= 1:
                    long_fee = funding * open_price * fund_long_qty * elapsed_8hours
                    short_fee = funding * open_price * fund_short_qty * elapsed_8hours
                    if verbose and (funding_array is not None):
                        funding_array[index] += long_fee + short_fee
                    long_equity_change -= long_fee
                    long_unrealized -= long_fee
                    short_equity_change -= short_fee
                    short_unrealized -= short_fee
                    prev_timestamp = current_timestamp

        # Update equity from filled orders
        for order_type, price, order_qty, fee_type, _ in filled_orders:
            fee: float = (maker if fee_type == 'maker' else taker) * price
            if order_type == 'long_entry':
                fee = fee * order_qty
                if verbose and (long_fee_array is not None):
                    long_fee_array[index] += fee
                long_equity_change_value: float = (price - prev_price) * long_qty - fee
                short_equity_change_value: float = (prev_price - price) * short_qty

                # Update position
                long_qty += order_qty
                if position == 0:
                    position = 1
                elif position == -1:
                    position = 2
                elif position in [2, -2]:
                    raise ValueError(f'Unhandled hedge. Position {position} but long_entry found at '
                                     f'index {index} for symbol {symbol} and strategy {strategy_name} '
                                     f'and parameter {parameter} for dates {date_from} to {date_to}')

                long_equity_change += long_equity_change_value
                short_equity_change += short_equity_change_value
                long_unrealized += long_equity_change_value
                short_unrealized += short_equity_change_value
                if trade_start_long is None:
                    trade_start_long = current_timestamp
                else:
                    raise TypeError(f'trade_start_long None but long_entry found at '
                                    f'index {index} for symbol {symbol} and strategy {strategy_name} '
                                    f'and parameter {parameter} for dates {date_from} to {date_to}')

            elif order_type == 'short_entry':
                fee = fee * order_qty
                if verbose and (short_fee_array is not None):
                    short_fee_array[index] += fee
                long_equity_change_value = (price - prev_price) * long_qty
                short_equity_change_value = (prev_price - price) * short_qty - fee

                # Update position
                short_qty += order_qty
                if position == 0:
                    position = -1
                elif position == 1:
                    position = -2
                elif position in [2, -2]:
                    raise ValueError(f'Unhandled hedge. Position {position} but short_entry found at '
                                     f'index {index} for symbol {symbol} and strategy {strategy_name} '
                                     f'and parameter {parameter} for dates {date_from} to {date_to}')

                long_equity_change += long_equity_change_value
                short_equity_change += short_equity_change_value
                long_unrealized += long_equity_change_value
                short_unrealized += short_equity_change_value
                if trade_start_short is None:
                    trade_start_short = current_timestamp
                else:
                    raise TypeError(f'trade_start_short None but short_entry found at '
                                    f'index {index} for symbol {symbol} and strategy {strategy_name} '
                                    f'and parameter {parameter} for dates {date_from} to {date_to}')

            elif order_type == 'long_exit':
                fee = fee * long_qty
                if verbose and (long_fee_array is not None):
                    long_fee_array[index] += fee
                long_equity_change_value = (price - prev_price) * long_qty - fee
                short_equity_change_value = (prev_price - price) * short_qty

                # Update position
                long_qty = 0.0
                if position == 1:
                    position = 0
                elif position == -2:
                    position = -1
                elif position == 2:
                    position = -1
                elif position in [-1, 0]:
                    raise ValueError(f'Position is {position} but long_exit found at'
                                     f'index {index} for symbol {symbol} and strategy {strategy_name} '
                                     f'and parameter {parameter} for dates {date_from} to {date_to}')

                balance += long_unrealized + long_equity_change_value
                long_equity_change += long_equity_change_value
                short_equity_change += short_equity_change_value
                long_unrealized = 0.0
                short_unrealized += short_equity_change_value
                number_of_trades += 1
                if trade_start_long is not None:
                    total_days_traded += (current_timestamp - trade_start_long).total_seconds() / 86400.0
                    trade_start_long = None
                else:
                    raise TypeError(f'trade_start_long None but long_exit found at '
                                    f'index {index} for symbol {symbol} and strategy {strategy_name} '
                                    f'and parameter {parameter} for dates {date_from} to {date_to}')

            else:  # order_type == 'short_exit'
                fee = fee * short_qty
                if verbose and (short_fee_array is not None):
                    short_fee_array[index] += fee
                long_equity_change_value = (price - prev_price) * long_qty
                short_equity_change_value = (prev_price - price) * short_qty - fee

                # Update position
                short_qty = 0.0
                if position == -1:
                    position = 0
                elif position == 2:
                    position = 1
                elif position == -2:
                    position = 1
                elif position in [1, 0]:
                    raise ValueError(f'Position is {position} but short_exit found at'
                                     f'index {index} for symbol {symbol} and strategy {strategy_name} '
                                     f'and parameter {parameter} for dates {date_from} to {date_to}')

                balance += short_unrealized + short_equity_change_value
                long_equity_change += long_equity_change_value
                short_equity_change += short_equity_change_value
                short_unrealized = 0.0
                long_unrealized += long_equity_change_value
                number_of_trades += 1
                if trade_start_short is not None:
                    total_days_traded += (current_timestamp - trade_start_short).total_seconds() / 86400.0
                    trade_start_short = None
                else:
                    raise TypeError(f'trade_start_short None but short_exit found at '
                                    f'index {index} for symbol {symbol} and strategy {strategy_name} '
                                    f'and parameter {parameter} for dates {date_from} to {date_to}')

            # Update prev price
            prev_price = price

        # Update equity at close
        if index < length - 1:
            long_equity_change_value = (close_price - prev_price) * long_qty
            short_equity_change_value = (prev_price - close_price) * short_qty

        else:  # Exit all open positions at last bar
            # Long Exit
            if long_qty > 0.0:
                final_closing_price = close_price * (1 - slippage)
                long_equity_change_value = (final_closing_price - prev_price) * long_qty
                long_fee = taker * long_qty * final_closing_price
                if verbose and (long_fee_array is not None):
                    long_fee_array[index] += long_fee
                long_equity_change_value -= long_fee
                balance += long_unrealized + long_equity_change_value
                number_of_trades += 1
                if trade_start_long is not None:
                    total_days_traded += (current_timestamp - trade_start_long).total_seconds() / 86400.0
                    long_qty = 0.0
                    position = 0
                else:
                    raise TypeError(f'trade_start_long None but long_exit found at '
                                    f'index {index} for symbol {symbol} and strategy {strategy_name} '
                                    f'and parameter {parameter} for dates {date_from} to {date_to}')
            else:
                long_equity_change_value = 0.0

            # Short Exit
            if short_qty > 0.0:
                final_closing_price = close_price * (1 + slippage)
                short_equity_change_value = (final_closing_price - prev_price) * short_qty
                short_fee = taker * short_qty * final_closing_price
                if verbose and (short_fee_array is not None):
                    short_fee_array[index] += short_fee
                short_equity_change_value -= short_fee
                balance += short_unrealized + short_equity_change_value
                number_of_trades += 1
                if trade_start_short is not None:
                    total_days_traded += (current_timestamp - trade_start_short).total_seconds() / 86400.0
                    short_qty = 0.0
                    position = 0
                else:
                    raise TypeError(f'trade_start_short None but short_exit found at '
                                    f'index {index} for symbol {symbol} and strategy {strategy_name} '
                                    f'and parameter {parameter} for dates {date_from} to {date_to}')
            else:
                short_equity_change_value = 0.0

        long_equity_change += long_equity_change_value
        short_equity_change += short_equity_change_value
        long_unrealized += long_equity_change_value if long_qty > 0.0 else 0.0
        short_unrealized += short_equity_change_value if short_qty > 0.0 else 0.0
        if verbose:
            if number_of_trades_array is not None:
                number_of_trades_array[index] = number_of_trades
            if long_equity_change_array is not None:
                long_equity_change_array[index] = long_equity_change
            if short_equity_change_array is not None:
                short_equity_change_array[index] = short_equity_change
            if long_unrealized_array is not None:
                long_unrealized_array[index] = long_unrealized
            if short_unrealized_array is not None:
                short_unrealized_array[index] = short_unrealized
            if long_qty_array is not None:
                long_qty_array[index] = long_qty
            if short_qty_array is not None:
                short_qty_array[index] = short_qty

        # Set values per row
        balance_array[index] = balance
        equity += long_equity_change + short_equity_change
        equity_array[index] = equity

        # breaks loop and assumes this is last trade.
        if equity < 0:
            # How to handle when equity drops to zero?
            equity_array[index + 1:] = equity
            balance_array[index + 1:] = equity
            position_array[index + 1:] = 0
            df = df.with_columns(pl.Series('position', position_array),
                                 pl.Series('balance', balance_array),
                                 pl.Series('equity', equity_array))
            df = df.with_columns(pct_return=pl.col('equity').pct_change(),
                                 log_return=pl.col('equity').log() - pl.col('equity').shift(1).log())
            df = df.with_columns(log_return=pl.col('log_return').fill_nan(0.0))
            if verbose:
                df = df.with_columns(pl.Series('long_qty', long_qty_array),
                                     pl.Series('short_qty', short_qty_array),
                                     pl.Series('funding', funding_array),
                                     pl.Series('long_fee', long_fee_array),
                                     pl.Series('short_fee', short_fee_array),
                                     pl.Series('long_equity_change', long_equity_change_array),
                                     pl.Series('short_equity_change', short_equity_change_array),
                                     pl.Series('long_unrealized', long_unrealized_array),
                                     pl.Series('short_unrealized', short_unrealized_array),
                                     pl.Series('number_of_trades', number_of_trades_array),
                                     )
            metadata['total_days_traded'] = total_days_traded
            metadata['number_of_trades'] = number_of_trades
            metadata['total_pct_gain'] = equity / start_equity - 1.0

            # Resampling. Will produce NULL values for some columns.
            df = df.upsample(time_column='timestamp', every=metadata['timeframe'])
            df = df.with_columns(pl.col('pct_return').fill_null(strategy='zero'),
                                 pl.col('log_return').fill_null(strategy='zero'),
                                 pl.col('equity').fill_null(strategy='forward'),
                                 # ADD MORE COLUMNS HERE if needed for other calculations
                                 )
            return df, metadata

        if position == 0:
            prev_timestamp = None

        position_array[index] = position

        # Prepare for next loop
        orders = strategy_logic(row, col, position, balance, parameter,
                                risk_per_trade, sqrt_n_per_day, persist)
        prev_price = row[col['close']]

    # Convert numpy array to series and add it to the dataframe
    df = df.with_columns(pl.Series('position', position_array),
                         pl.Series('balance', balance_array),
                         pl.Series('equity', equity_array))

    df = df.with_columns(pct_return=pl.col('equity').pct_change(),
                         log_return=pl.col('equity').log() - pl.col('equity').shift(1).log())

    if verbose:
        df = df.with_columns(pl.Series('long_qty', long_qty_array),
                             pl.Series('short_qty', short_qty_array),
                             pl.Series('funding', funding_array),
                             pl.Series('long_fee', long_fee_array),
                             pl.Series('short_fee', short_fee_array),
                             pl.Series('long_equity_change', long_equity_change_array),
                             pl.Series('short_equity_change', short_equity_change_array),
                             pl.Series('long_unrealized', long_unrealized_array),
                             pl.Series('short_unrealized', short_unrealized_array),
                             pl.Series('number_of_trades', number_of_trades_array),
                             )

    metadata['total_days_traded'] = total_days_traded
    metadata['number_of_trades'] = number_of_trades
    metadata['total_pct_gain'] = equity / start_equity - 1.0

    # Resampling. Will produce NULL values for some columns.
    df = df.upsample(time_column='timestamp', every=metadata['timeframe'])
    df = df.with_columns(pl.col('pct_return').fill_null(strategy='zero'),
                         pl.col('log_return').fill_null(strategy='zero'),
                         pl.col('equity').fill_null(strategy='forward'),
                         # ADD MORE COLUMNS HERE if needed for other calculations
                         )

    return df, metadata


def fill_orders(orders: list[Any],
                slippage: float,
                *prices: float) -> list[Any]:
    """
    Fills market, stop, and limit orders based on given prices and slippage.

    Parameters:
    orders (T_orders): List of orders, where each order is a tuple of (order_type, order_price, order_qty).
    slippage (float): The percentage of slippage to apply to the fill price.
    prices (float): Variable number of prices to consider for filling orders. Should be in order.

    Returns:
    T_orders: A list of filled orders.
    """
    filled: list[Any] = []

    for index, price in enumerate(prices):
        unfilled: list[Any] = []

        for order_type, order_price, order_qty in orders:
            order_type = order_type.lower()

            if order_type == 'buy_market':
                fill_price: float = price * (1.0 + slippage)
                filled.append(['long_entry', fill_price, order_qty, 'taker', 'at_open'] if order_qty > 0
                              else ['short_exit', fill_price, 0, 'taker', 'at_open'])

            elif order_type == 'sell_market':
                fill_price = price * (1 - slippage)
                filled.append(['short_entry', fill_price, order_qty, 'taker', 'at_open'] if order_qty > 0
                              else ['long_exit', fill_price, 0, 'taker', 'at_open'])

            elif order_type == 'buy_stop' and price >= order_price:
                fill_price = (price if index == 0 else order_price) * (1 + slippage)
                executed_at: str = 'at_open' if index == 0 else 'pending'
                filled.append(['long_entry', fill_price, order_qty, 'taker', executed_at] if order_qty > 0
                              else ['short_exit', fill_price, 0, 'taker', executed_at])

            elif order_type == 'sell_stop' and price <= order_price:
                fill_price = (price if index == 0 else order_price) * (1 - slippage)
                executed_at = 'at_open' if index == 0 else 'pending'
                filled.append(['short_entry', fill_price, order_qty, 'taker', executed_at] if order_qty > 0
                              else ['long_exit', fill_price, 0, 'taker', executed_at])

            elif order_type == 'buy_limit' and price <= order_price:
                fill_price = (price if index == 0 else order_price) * (1 + slippage)
                executed_at = 'at_open' if index == 0 else 'pending'
                filled.append(['long_entry', fill_price, order_qty, 'maker', executed_at] if order_qty > 0
                              else ['short_exit', fill_price, 0, 'maker', executed_at])

            elif order_type == 'sell_limit' and price >= order_price:
                fill_price = (price if index == 0 else order_price) * (1 - slippage)
                executed_at = 'at_open' if index == 0 else 'pending'
                filled.append(['short_entry', fill_price, order_qty, 'maker', executed_at] if order_qty > 0
                              else ['long_exit', fill_price, 0, 'maker', executed_at])

            else:
                unfilled.append([order_type, order_price, order_qty])
        orders = unfilled

    return filled


def get_data(symbol: str,
             date_from: datetime,
             date_to: datetime,
             use_synthetic: bool = False) -> tuple[pl.DataFrame, dict[str, Any]]:
    """
    Retrieve and process financial data for a given symbol within a specified date range.

    Function Parameters:
    symbol (str): The financial instrument symbol (e.g., 'BTCUSDT').
    date_from (datetime): The starting date of the data range.
    date_to (datetime): The ending date of the data range.

    Returns:
    Tuple[pl.DataFrame, dict[str, Any]]: A tuple containing the filtered DataFrame and a dictionary
    with metadata (date range, slippage, timeframe, and symbol).

    Raises:
    ValueError: If the symbol is invalid or the price file type is not supported.
    IndexError: If the symbol is not found in the slippage file.
    """
    # Default file paths and settings
    slippage_file: str = path.from_pycharm(r'Slippage.csv')
    price_filetype: str = 'feather'
    timeframe: str = r'1h'
    if use_synthetic:
        price_folder: str = path.from_pycharm(r'garch')
        symbol, synthetic_index = symbol.split('_')
        price_file_path: str = path.join(price_folder, f'{symbol}_{synthetic_index} {timeframe}.{price_filetype}')
        return_symbol: str = f'{symbol}_{synthetic_index}'
    else:
        price_folder = path.from_pycharm(r'BinanceDownload\Downloaded')
        price_file_path = path.join(price_folder, f'{symbol} {timeframe} sorted.{price_filetype}')
        return_symbol = symbol

    # Validate and format the symbol
    symbol = symbol.upper()
    if not symbol.endswith('USDT'):
        raise ValueError(f'Invalid symbol {symbol}.')

    # Convert dates to utc
    date_from = dtf.make_utc(date_from)
    date_to = dtf.make_utc(date_to)

    # Read slippage data from the CSV file
    try:
        slippage: float = pl.read_csv(slippage_file, raise_if_empty=True) \
            .filter(pl.col('Symbol') == symbol).item(0, 'Slippage')
    except pl.exceptions.NoRowsReturnedError:
        raise IndexError(f'Could not find symbol {symbol} in slippage file {slippage_file}.')
    except pl.exceptions.NoDataError:
        raise ValueError(f'Slippage file is empty {slippage_file}.')
    except FileNotFoundError:
        raise FileNotFoundError(f'Slippage file not found: {slippage_file}.')

    # Read the price data based on the specified file type
    try:
        if price_filetype == 'feather':
            df: pl.DataFrame = pl.scan_ipc(price_file_path, memory_map=False).tail(70_000).collect()
        elif price_filetype == 'csv':
            df = pl.scan_csv(price_file_path).tail(70_000).collect()
        else:
            raise ValueError(f'Cannot open price file. File format {price_filetype} not supported.')
    except FileNotFoundError:
        raise FileNotFoundError(f'Price data file not found: {price_file_path}.')

    # Ensure the DataFrame is sorted by the 'timestamp' column
    if not df['timestamp'].is_sorted():
        raise ValueError('Dataframe column timestamp is not sorted.')
    df = df.set_sorted(['timestamp'])

    # Buffer size: number of bars included before date_from to calculate indicator values
    buffer: int = 768
    timestamp_array: np.ndarray = df['timestamp'].to_numpy()

    # Find indices for the date range with buffer consideration
    date_from_index: int = int(np.searchsorted(timestamp_array, date_from.timestamp() * 1e9))
    date_to_index: int = min(int(np.searchsorted(timestamp_array, date_to.timestamp() * 1e9)) - 1,
                             timestamp_array.shape[0] - 1)
    if date_to_index - date_from_index <= buffer:
        raise ValueError(f'{symbol} Date range date_from {date_from} and date_to {date_to} '
                         f'returned too little data to work with.')

    # Adjust the indices to include the buffer
    buffer_index: int = max(0, date_from_index - buffer)
    df = df.slice(offset=buffer_index, length=date_to_index - buffer_index + 1)
    date_from_index -= buffer_index

    return df, {'date_from': dtf.make_utc(df.item(date_from_index, 'timestamp')),
                'date_to': dtf.make_utc(df.item(-1, 'timestamp')),
                'slippage': slippage,
                'timeframe': timeframe,
                'symbol': return_symbol}


if __name__ == '__main__':
    main()
