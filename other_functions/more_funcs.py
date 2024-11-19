from datetime import datetime
from warnings import warn
from types import ModuleType
from typing import Any, TypeVar
import polars as pl
import pandas as pd
from itertools import product
import numpy as np

import os
import glob
from other_functions import path_functions as pt
from math import comb
from heapq import nlargest

T = TypeVar('T')


def initial_selection_results(initial_selection_summary_path: str, initial_selection_log_returns_path: str):
    top_strategies_selected: dict[str, list[str]] = get_top_results(initial_selection_summary_path)
    list_of_columns: list[str] = []
    for strategy, symbols in top_strategies_selected.items():
        strategy_columns: list[str] = [f'{symbol}:{strategy}' for symbol in symbols]
        list_of_columns.extend(strategy_columns)
    df: pl.DataFrame = pl.read_csv(initial_selection_log_returns_path)
    df = df.select(list_of_columns)

    # preprocess dataframe:
    df_pd: pd.DataFrame = df.to_pandas().dropna(axis=1)
    df_pd = df_pd.loc[:, (df_pd != 0).any(axis=0)]  # Drops zero columns
    df_pd = df_pd.T.drop_duplicates().T  # Drops duplicates

    # Correlation Matrix
    corr0: pd.DataFrame = pd.DataFrame(np.corrcoef(df_pd.values, rowvar=False), columns=df_pd.columns).T
    corr0.columns = df_pd.columns

    # perform ONC to cluster returns
    clusters: list[list[str]] = onc(corr0)

    # get performance of each cluster
    initial_selection_summary_df: pl.DataFrame = pl.read_csv(initial_selection_summary_path)
    initial_selection_summary_df = (initial_selection_summary_df.
                                    with_columns(cluster=pl.concat_str(["symbol", "strategy"], separator=":")))
    max_performance: float = 0.0
    best_cluster: list[str] = []
    for cluster in clusters:
        cluster_df: pl.DataFrame = initial_selection_summary_df.filter(pl.col('cluster').is_in(cluster))
        psr_values: list[float] = cluster_df['psr'].to_list()
        performance = average_best_of_k(psr_values, min(4, len(psr_values)))
        if performance > max_performance:
            max_performance = performance
            best_cluster = cluster

    for sym_strategy in best_cluster:
        symbol, strategy = sym_strategy.split(':')
        print(f'("{symbol}", st.{strategy}),')


def average_best_of_k(scores: T, k: int) -> float:
    """
    Compute the average of the best-of-k values from an array of scores.

    Parameters:
        scores (list or np.ndarray): Array of numbers (e.g., model performances).
        k (int): Number of top values to consider in combinations.

    Returns:
        float: Average of the best-of-k values.
    """
    # Sort scores in ascending order
    scores: T = np.sort(scores)
    # drop nans
    scores = scores[~np.isnan(scores)]
    n: int = len(scores)

    if k > n:
        raise ValueError("k cannot be greater than the number of elements in scores.")

    # Compute weights for each score
    total_comb: int = comb(n, k)
    weights: np.ndarray = np.array([
        comb(i, k - 1) / total_comb
        for i in range(n)
    ])

    # Calculate the weighted average
    average: float = (weights * scores).sum()

    return average


def get_top_results(filepath: str, n_top_models: int | None=None,
                      n_top_variants: int=3, n_top_symbols: int | None=None,
                      print_to_cli: bool = False) -> dict:
    df: pl.DataFrame = pl.read_csv(filepath)
    df = df.with_columns(group=pl.col('strategy').str.slice(0, 1))
    df_filtered: pl.DataFrame = df.filter(pl.col('psr').is_not_nan()).sort('psr', descending=True)
    # variant_performance: pl.DataFrame = (df_filtered.group_by(['strategy', 'group']).agg(pl.col('psr').mean())
    #                                      .sort('psr', descending=True))
    # model_performance: pl.DataFrame = (variant_performance.group_by('group').agg(pl.col('psr').mean())
    #                                    .sort('psr', descending=True))
    #
    # top_models: list = sorted(model_performance['group'].head(n_top_models).to_list())
    # top_variants: list = [variant_performance.filter(pl.col('group') == model)['strategy'].head(n_top_variants).to_list()
    #                       for model in top_models]
    # top_variants = sum(top_variants, [])  # Flattens the list
    # top_symbols: dict = {variant:df_filtered.filter(pl.col('strategy') == variant)['symbol']
    #                      .head(n_top_symbols).to_list() for variant in top_variants}
    # print(f'Top {len(top_models)} Models:\n{top_models}')
    # print(f'Top {n_top_variants} Variants:\n{top_variants}')
    # print('Top Combinations:')
    # for variant, symbols in top_symbols.items():
    #     for symbol in symbols:
    #         print(f'("{symbol}", st.{variant}),')


    # use best of k algorithm to determine performance
    all_variants: list[str] = df['strategy'].unique().to_list()
    variant_performance: dict[str, float] = {}
    for variant in all_variants:
        df_filtered_variant: pl.DataFrame = df_filtered.filter(pl.col('strategy') == variant)
        psr_values: list[float] = df_filtered_variant['psr'].to_list()
        variant_performance[variant] = average_best_of_k(psr_values, min(4, len(psr_values)))

    all_models: list[str] = df['group'].unique().to_list()
    if n_top_models is None:
        top_models: list[str] = all_models
    else:
        model_performance: dict[str, float] = {}
        for model in all_models:
            model_variants: list[str] = df_filtered.filter(pl.col('group') == model)['strategy'].unique().to_list()
            psr_values = [variant_performance[model_variant] for model_variant in model_variants]
            model_performance[model] = average_best_of_k(psr_values, min(4, len(psr_values)))
        top_models = nlargest(n_top_models, model_performance, key=model_performance.get)

    top_variants: list[str] = []
    for model in top_models:
        model_variants = df_filtered.filter(pl.col('group') == model)['strategy'].unique().to_list()
        subset_variants: dict[str, float] = {key:value for key, value in variant_performance.items()
                                             if key in model_variants}
        top_variants.extend(nlargest(n_top_variants, subset_variants, key=subset_variants.get))

    if n_top_symbols is None:
        top_symbols: dict = {variant:df_filtered.filter(pl.col('strategy') == variant)['symbol'].to_list()
                             for variant in top_variants}
    else:
        top_symbols = {variant:df_filtered.filter(pl.col('strategy') == variant, pl.col('psr') >= 0.5)['symbol']
        .head(n_top_symbols).to_list() for variant in top_variants}

    if print_to_cli:
        print(f'Top {len(top_models)} Models:\n{top_models}')
        print(f'Top {len(top_variants)} Variants:\n{top_variants}')
        print('Top Combinations:')
        for variant, symbols in top_symbols.items():
            for symbol in symbols:
                print(f'("{symbol}", st.{variant}),')
    return top_symbols


def print_best_parameters(filepath: str):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(filepath)

    # Initialize a dictionary to store results
    result = {}

    # Group by strategy
    for strategy, group in data.groupby('::strategy'):
        strategy_result = {}

        # Initialize parameter groups
        param_groups = {param: param_group.sort_values('::mean_psr', ascending=False)
                        for param, param_group in group.groupby('::parameter')}

        # Select initial values where mean_psr > 0.3
        for parameter, param_group in param_groups.items():
            filtered_group = param_group[param_group['::mean_psr'] > 0.3]
            if not filtered_group.empty:
                strategy_result[parameter] = filtered_group['::value'].tolist()
            else:
                # If no values > 0.3, select the row with the highest mean_psr
                strategy_result[parameter] = [param_group.iloc[0]['::value']]

        # Calculate combinations
        combinations = list(product(*strategy_result.values()))

        # If less than 3 combinations, prioritize adding higher mean_psr values
        if len(combinations) < 3:
            # Create a list to store additional potential values sorted by highest mean_psr
            additional_values = []
            for parameter, param_group in param_groups.items():
                for _, row in param_group.iterrows():
                    if row['::value'] not in strategy_result[parameter]:
                        additional_values.append((row['::mean_psr'], parameter, row['::value']))

            # Sort additional values by mean_psr in descending order
            additional_values.sort(reverse=True, key=lambda x: x[0])

            # Add values to ensure at least three combinations
            for _, parameter, value in additional_values:
                if value not in strategy_result[parameter]:
                    strategy_result[parameter].append(value)
                    combinations = list(product(*strategy_result.values()))
                    if len(combinations) >= 3:
                        break

        # Store the strategy result
        result[strategy] = strategy_result

    # Print the aggregated results
    for strategy, params in result.items():
        print(f"For {strategy}:")
        for param, values in params.items():
            if ":" not in values[0]:
                for string_value in values:
                    if "." in string_value:
                        values_list = [float(value) for value in values]
                        break
                else:
                    values_list = [int(value) for value in values]
            else:
                values_list = values
            values_list = sorted(values_list)
            print(f"{param} = {values_list}")
        print('')

def update_best_parameters(param_trim_summary_filepath: str):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(param_trim_summary_filepath)

    # Get best stdev
    stdev_df = pl.from_pandas(data).filter(pl.col('parameter').eq('stdev'))
    stdev_df = stdev_df.group_by('value').agg(pl.col('mean_psr').mean())
    stdev = stdev_df.filter(pl.col('mean_psr') == pl.col('mean_psr').max())['value'].item()
    stdev = int(stdev) if stdev.is_integer() else stdev
    data = data.loc[(data['parameter'] != 'stdev') & (data['parameter'].notnull())]

    # Initialize a dictionary to store results
    result = {}
    strategy_directory = r'C:\Users\rober\PycharmProjects\AlgoTrader\strategies_package'

    # Group by strategy
    for strategy, group in data.groupby('strategy'):
        strategy_result = {}

        # Initialize parameter groups
        param_groups = {param: param_group.sort_values('mean_psr', ascending=False)
                        for param, param_group in group.groupby('parameter')}

        # Select initial values where mean_psr > 0.3
        for parameter, param_group in param_groups.items():
            filtered_group = param_group[param_group['mean_psr'] > 0.3]
            if not filtered_group.empty:
                strategy_result[parameter] = filtered_group['value'].tolist()
            else:
                # If no values > 0.3, select the row with the highest mean_psr
                strategy_result[parameter] = [param_group.iloc[0]['value']]

        # Calculate combinations
        combinations = list(product(*strategy_result.values()))

        # If less than 3 combinations, prioritize adding higher mean_psr values
        if len(combinations) < 3:
            # Create a list to store additional potential values sorted by highest mean_psr
            additional_values = []
            for parameter, param_group in param_groups.items():
                for _, row in param_group.iterrows():
                    if row['value'] not in strategy_result[parameter]:
                        additional_values.append((row['mean_psr'], parameter, row['value']))

            # Sort additional values by mean_psr in descending order
            additional_values.sort(reverse=True, key=lambda x: x[0])

            # Add values to ensure at least three combinations
            for _, parameter, value in additional_values:
                if value not in strategy_result[parameter]:
                    strategy_result[parameter].append(value)
                    combinations = list(product(*strategy_result.values()))
                    if len(combinations) >= 3:
                        break

        # Store the strategy result
        result[strategy] = strategy_result

    # process the aggregated results
    for strategy, params in result.items():
        print(f"For {strategy}:")
        for param, values in params.items():
            if isinstance(values[0], str):
                if ":" not in values[0]:
                    for string_value in values:
                        if "." in string_value:
                            values_list = [float(value) for value in values]
                            break
                    else:
                        values_list = [int(value) for value in values]
                else:
                    values_list = values
            else:
                for value in values:
                    if not value.is_integer():
                        values_list = [float(value) for value in values]
                        break
                else:
                    values_list = [int(value) for value in values]

            values_list = sorted(values_list)
            params[param] = values_list

        string_params = ''
        for param, values in params.items():
            string_params += f'''
            {param} = {values}'''
        strategy_file = os.path.join(strategy_directory, f'{strategy}.py')
        with open(strategy_file, 'r') as f:
            # Read the file
            content = f.read()

        start_find = content.find(r'    match routine:')
        end = 0
        start = 0
        for i in range(start_find, 0, -1):
            if content[i] == ']':
                end = i
            elif content[i] == '[':
                start = i
                break
        string_params_list = content[start + 1: end].replace("'", "")
        start = content.find('case _:')
        old = content[start:]
        new = f'''case _:
            stdev = [{stdev}]{string_params}

    values: Any = iter_product({string_params_list})

    dict_parameters: list[dict] = [dict(zip(headers, value)) for value in values]
    return dict_parameters
'''
        content = content.replace(old, new)
        with open(strategy_file, 'w') as f:
            f.write(content)
            print(f'Updated {strategy_file}')


def find_code_in_files(code_snippet, directory, search_for_presence=True):
    # Use glob to find all .py files in the directory
    files = glob.glob(os.path.join(directory, "*.py"))

    # Iterate over the list of files
    for file in files:
        # Skip __init__.py files
        if os.path.basename(file) == '__init__.py':
            continue

        with open(file, 'r') as f:
            # Read the file
            content = f.read()

            # Check if the code snippet is in the content
            if search_for_presence and code_snippet in content:
                print(f"The {code_snippet=} was found in: {file}")
            elif not search_for_presence and code_snippet not in content:
                print(f"The {code_snippet=} was not found in: {file}")


def change_code_in_files(old, new, keyword, directory):
    # Use glob to find all .py files in the directory
    files = glob.glob(os.path.join(directory, "*.py"))

    # Iterate over the list of files
    for file in files:
        # Skip __init__.py files
        if os.path.basename(file) == '__init__.py':
            continue
        if os.path.basename(file).find(keyword) == -1:
            continue

        with open(file, 'r') as f:
            # Read the file
            content = f.read()

        if content.find(old) > 0:
            content = content.replace(old, new)

            with open(file, 'w') as f:
                f.write(content)
                print(f'Replaced {old=} to {new=} for {file}')


def cycle_through_numbers(iterate_over: list, qty: int):
    length = len(iterate_over)
    iterate_over.extend(iterate_over)
    all_list = []
    for i in range(length):
        all_list.append(iterate_over[i:i + qty])
    return all_list


def bootstrap_confidence_set(csv_file, n_bootstrap, n_rows, output_csv):
    # Read the original CSV file
    df = pd.read_csv(csv_file)

    # Columns to calculate the average and median
    columns_to_calculate = ['test_profit', 'test_days_per_trade', 'test_trades_per_year',
                            'test_ave_drawdown', 'test_max_drawdown', 'test_omega',
                            'test_sortino', 'test_annual_sharpe', 'test_psr']

    # Store results for each bootstrap sample
    results = []

    for _ in range(n_bootstrap):
        # Bootstrap sample: randomly select n_rows with replacement
        bootstrap_sample = df.sample(n=n_rows, replace=False)

        # Filter based on the conditions
        filtered_sample = bootstrap_sample[
            (bootstrap_sample['train_psr'] >= 0.95)
            # & (bootstrap_sample['train_median_psr'] >= 0.9)
        ]

        # If no rows match the filtering criteria, continue to the next bootstrap iteration
        if filtered_sample.empty:
            # Create a dictionary with the results for this bootstrap sample
            result_row = {f'avg_{col}': np.nan for col in columns_to_calculate}
            result_row.update({f'median_{col}': np.nan for col in columns_to_calculate})

            results.append(result_row)
            continue

        # Calculate average and median for the specified columns
        avg_values = filtered_sample[columns_to_calculate].mean()
        median_values = filtered_sample[columns_to_calculate].median()

        # Create a dictionary with the results for this bootstrap sample
        result_row = {f'avg_{col}': avg_values[col] for col in columns_to_calculate}
        result_row.update({f'median_{col}': median_values[col] for col in columns_to_calculate})

        results.append(result_row)

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Save the results to a new CSV file
    results_df.to_csv(output_csv, index=False)
    print(f"Bootstrap results saved to {output_csv}")


def warn_errors(sym_or_msg: str, strategy_name: str | None = None,
                date_from: datetime | None = None, date_to: datetime | None = None,
                parameter: dict | None = None, index: int | None = None) -> None:
    """
    Logs warning messages with optional details about the error context.

    Parameters:
    sym_or_msg (str): Symbol or message to be included in the warning.
    strategy_name (Optional[str]): The name of the strategy, if available.
    date_from (Optional[datetime]): The starting date, if available.
    date_to (Optional[datetime]): The ending date, if available.
    parameter (Optional[dict]): The parameter dictionary, if available.
    index (Optional[int]): The index, if available.

    Returns:
    None
    """
    if (strategy_name is None) or (date_from is None) or (date_to is None) or (parameter is None):
        warn(sym_or_msg)
    else:
        warn(f'symbol: {sym_or_msg} , strategy: {strategy_name}\n'
             f'date_from: {date_from}, date_to: {date_to}\n'
             f'parameter: {parameter}\n'
             f'index: {index}')
    return


def unpack_parameters(result: list[Any] | tuple[Any], headers: list[str]
                      ) -> tuple[tuple[Any, ...], list[str]]:
    """
    Unpacks parameter dictionaries into separate columns in the result headers.

    Parameters:
    result (Union[List[Any], Tuple[Any]]): The list of results to unpack.
    headers (List[str]): The headers for the results.

    Returns:
    Tuple[Tuple[Any], List[str]]: The updated result and headers with unpacked parameters.
    """
    result = tuple(result)
    if 'parameter' not in headers:
        return result, headers

    # Loops through the list to find unique keys
    index_parameter: int = headers.index('parameter')
    unique_keys: list[str] = []

    for row in result:
        for key in row[index_parameter]:
            if key not in unique_keys:
                unique_keys.append(key)
    unique_keys.reverse()

    # Add unique keys to headers
    for key in unique_keys:
        headers.insert(index_parameter, 'P:' + key)
    del headers[index_parameter + len(unique_keys)]

    # Add unique keys to results
    updated_results: list[Any] = []
    del_index: int = index_parameter + len(unique_keys)
    for row in result:
        parameter: dict = row[index_parameter]
        if not isinstance(parameter, dict):
            raise TypeError(f'Parameter {parameter} is not a dictionary.')
        new_row: list[Any] = list(row)
        for key in unique_keys:
            new_row.insert(index_parameter, parameter.get(key, None))
        del new_row[del_index]
        updated_results.append(tuple(new_row))

    return tuple(updated_results), headers


def to_list(iterable):
    """
    Converts an iterable to a list.
    This function removes the warning from pycharm.
    The IDE confuses list as a function vs list as a type.

    Parameters:
    iterable (Any): The iterable to convert.

    Returns:
    List[Any]: The converted list.
    """
    return list(iterable)


def check_type_combination(combination: list[list]) -> None:
    """
    Checks the types of elements in a combination list.

    Parameters:
    combination (List[Any]): The combination list to check.

    Raises:
    TypeError: If the types of the elements are incorrect.
    ValueError: If the length of the combination list is invalid.
    """
    if not isinstance(combination[0], str):
        raise TypeError(f'Expecting str for symbol. Got {type(combination[0])} instead.')
    if not isinstance(combination[1], datetime):
        raise TypeError(f'Expecting datetime for date_from. Got {type(combination[1])} instead.')
    if not isinstance(combination[2], datetime):
        raise TypeError(f'Expecting datetime for date_to. Got {type(combination[2])} instead.')
    if not isinstance(combination[3], ModuleType):
        raise TypeError(f'Expecting ModuleType for strategy. Got {type(combination[3])} instead.')

    if len(combination) == 5:
        if not isinstance(combination[4], dict):
            raise TypeError(f'Expecting dict for parameter. Got {type(combination[4])} instead.')
        return
    elif len(combination) != 4:
        raise ValueError(f'Invalid length of combinations {combination}.')
    return
