from __future__ import annotations

import platform
from logging import Logger, getLogger
from pathlib import Path
from typing import Union, Optional, Any

from tqdm import tqdm

import numpy as np
from astropy import units

Numeric = Union[int, float]
AngleType = units.Quantity["angle"]
TimeType = units.Quantity["time"]
NAngleType = Union[Numeric, units.Quantity["angle"]]
NTimeType = Union[Numeric, units.Quantity["time"]]


def normalize(value: float, actual_min: float, actual_max: float, to_min=0.0, to_max=1.0) -> float:
    """
    Normalizes a given value within a defined range.

    This function transforms a given value from an actual range
    (original minimum and maximum) to a target range (specified
    by to_min and to_max). Optionally, the function can handle
    inverted ranges (where the minimum value is greater than
    the maximum value).

    Args:
        value: The value to be normalized.
        actual_min: The minimum value of the actual range.
        actual_max: The maximum value of the actual range.
        to_min: The minimum value of the target range, default is 0.0.
        to_max: The maximum value of the target range, default is 1.0.

    Returns:
        The normalized value within the target range.
    """
    flip = actual_min > actual_max
    actual_min, actual_max = min(actual_min, actual_max), max(actual_min, actual_max)
    to_min, to_max = min(to_min, to_max), max(to_min, to_max)

    normalized = (value - actual_min) / (actual_max - actual_min)

    if flip:
        normalized = 1.0 - normalized

    return float(to_min + normalized * (to_max - to_min))


def interpolate(value: float, start_value: float, end_value: float) -> float:
    """
    Interpolates a value between a start and end value.

    Notes
    -----
    Ensures start_value is less than or equal to end_value

    Parameters
    ----------
    value: float
        A float value between 0 and 1 used to interpolate between start_value and end_value.
    start_value: float
        The starting value in the interpolation range.
    end_value: float
        The ending value in the interpolation range.

    Returns
    -------
    float
        The interpolated value
    """
    start_value, end_value = min(start_value, end_value), max(start_value, end_value)
    return start_value + (end_value - start_value) * value


def neighbor(array, value):
    """
    Find the index of the nearest neighbor to a given value in an array.

    Parameters
    ----------
    array: numpy.ndarray
        The array in which to search for the nearest neighbor.
    value: float
        The reference value to which the nearest neighbor is sought.

    Returns
    -------
    int
        The index of the nearest neighbor in the array.
    """
    return (np.abs(array - value)).argmin()


def unit_checker(value: Any, unit: units.Quantity) -> units.Quantity:
    """
    Convert a numerical value into a specified unit or validate if a value has compatible units.

    Parameters
    ----------
    value: Any
        The value to be converted or validated.
    unit: units.Quantity
        The unit to convert the value to, or validate against.

    Returns
    -------
    units.Quantity
        The value in the specified units.

    Raises
    ------
    ValueError
        If the value has incompatible units with the specified unit.
    """
    if isinstance(value, (float, int)):
        return value * unit

    if unit.is_equivalent(value):
        return value
    else:
        raise ValueError("Bad unit")


def degree_checker(value: Any, unit: units.Quantity = units.deg) -> units.Quantity:
    """
    Converts the input value into angle.

    Parameters
    ----------
    value: Any
        The input value that needs to be converted.
    unit: units.Quantity, default = units.deg
        The target unit for conversion, default is degrees.

    Returns
    -------
    units.Quantity
        The converted value in angle.
    """
    return unit_checker(value, unit)


def time_checker(value: Any, unit: units.Quantity = units.s) -> units.Quantity:
    """
    Checks if the given time value is valid for the specified unit.

    Parameters
    ----------
    value: Any
        The time value to check.
    unit: units.Quantity, default = units.s
        The unit of the time value. Defaults to units.s.

    Returns
    -------
    units.Quantity
        The converted value in time.
    """
    return unit_checker(value, unit)


def logger_checker(logger: Optional[Logger] = None, name: Optional[str] = None) -> Logger:
    """
    Checks if a logger is passed as an argument. If not, it returns a logger with the specified name
    or a default name.


    Parameters
    ----------
    logger: Logger, default = None
        An optional Logger instance.
    name: str, default = None
        An optional string representing the name of the logger.

    Returns
    -------
    units.Quantity
        The converted value in time.
    """
    if logger is None:
        if name is None:
            return getLogger("octans")
        else:
            return getLogger(name)
    else:
        return logger


def database_dir():
    """
    Determines and creates if necessary the application settings directory based on the operating system.

    Notes
    -----
    - For Windows, it returns the path: C:\\Users\\<username>\\AppData\\Local\\octans
    - For macOS, it returns the path: /Users/<username>/Library/Application Support/octans
    - For Linux, it returns the path: /home/<username>/.config/octans
    - For any other operating system, it defaults to: /home/<username>/octans

    Returns
    -------
    Path
        The path to the application settings directory.

    """
    home_dir = Path.home()
    if platform.system() == "Windows":
        settings_dir = home_dir / "AppData" / "Local" / "octans"
    elif platform.system() == "Darwin":
        settings_dir = home_dir / "Library" / "Application Support" / "octans"
    elif platform.system() == "Linux":
        settings_dir = home_dir / ".config" / "octans"
    else:
        settings_dir = home_dir / "octans"

    if not settings_dir.exists():
        settings_dir.mkdir()

    return settings_dir


def tqdmify(iterable: Any, verbose: bool = True, **kwargs) -> Union[tqdm, Any]:
    """
    Wraps an iterable with tqdm's progress bar functionality.

    Parameters
    ----------
    iterable: Any
        The iterable to wrap with a progress bar.
    verbose: verbose, default = True
        If True, wraps the iterable with tqdm progress bar.
    **kwargs: dict
        Additional keyword arguments to pass to tqdm when verbose is True.

    Returns
    -------
    tqdm, Any
        If verbose is True, returns the iterable wrapped with tqdm progress bar.
        Otherwise, returns the original iterable unchanged.
    """
    if verbose:
        return tqdm(iterable, **kwargs)

    return iterable
