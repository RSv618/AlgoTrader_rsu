# Automatically imports strategy modules
import os
from other_functions import path_functions as path
import pkgutil


def auto_import():
    # Define directory
    directory: str = path.from_project(f'strategies_package')

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Create the file
    list_of_strategies: list[str] = []
    str_init: str = ''
    for mod in pkgutil.iter_modules([directory]):
        # This loop will go through each .py file in the directory
        str_init += f'from strategies_package.{mod.name} import *\n'
        list_of_strategies.append(f'{mod.name}')
    with open(os.path.join(directory, r'__init__.py'), 'w') as fp:
        fp.write(str_init)

    # Save list of strategies
    with open(os.path.join(directory, r'00 list_of_strategies.txt'), 'w') as file:
        for strategy in list_of_strategies:
            file.write(f'st.{strategy},\n')


auto_import()
