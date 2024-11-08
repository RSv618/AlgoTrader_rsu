import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PYCHARM_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir))


def find_make(directory: str, check_abs=True) -> str:
    """
    Creates a directory if it does not exist and returns the absolute path.

    Parameters:
    directory (str): The absolute directory path to create.
    check_abs (bool): If True, will check if directory is absolute.

    Returns:
    str: The absolute path of the created or existing directory.
    """
    if check_abs and (not os.path.isabs(directory)):
        raise ValueError(f'Path {directory} is not absolute.')
    if not os.path.isdir(directory):
        directory = os.path.dirname(directory)
    os.makedirs(directory, exist_ok=True)
    return directory


def from_project(directory: str) -> str:
    """
    Converts a relative directory path to an absolute path based on the project root directory.

    Parameters:
    directory (str): The relative directory path.

    Returns:
    str: The absolute path of the directory.

    Raises:
    ValueError: If the provided directory path is absolute.
    """
    if os.path.isabs(directory):
        raise ValueError(f'Parameter directory {directory} is absolute. '
                         f'Cannot use absolute path in from_project function.')
    abs_directory: str = os.path.join(PROJECT_ROOT, directory)
    find_make(abs_directory, check_abs=False)
    return abs_directory


def from_pycharm(directory: str) -> str:
    """
    Constructs an absolute path from the PyCharm projects directory.

    Parameters:
    directory (str): The relative directory path within the PyCharm projects directory.

    Returns:
    str: The absolute path of the directory within the PyCharm projects directory.

    Raises:
    ValueError: If the provided directory path is absolute.
    """
    if os.path.isabs(directory):
        raise ValueError(f'Parameter directory {directory} is absolute. '
                         f'Cannot use absolute path in from_pycharm function.')
    abs_directory: str = os.path.join(PYCHARM_ROOT, directory)
    find_make(abs_directory, check_abs=False)
    return abs_directory


def join(abs_directory: str, sub_directory: str) -> str:
    """
    Joins an absolute directory path with a subdirectory path.

    Parameters:
    abs_directory (str): The absolute directory path.
    sub_directory (str): The subdirectory path to join.

    Returns:
    str: The combined directory path.

    Raises:
    ValueError: If the provided abs_directory is not absolute.
    """
    if not os.path.isabs(abs_directory):
        raise ValueError(f'Parameter abs_directory {abs_directory} is not absolute. First argument must be absolute')
    return os.path.join(abs_directory, sub_directory)


def get_directory(filepath: str) -> str:
    return os.path.dirname(filepath)
