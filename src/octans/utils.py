from typing import Union
from astropy import units
import numpy as np

Numeric = Union[int, float]
Angle = units.Quantity["angle"]
Time = units.Quantity["time"]
NAngle = Union[Numeric, Angle]
NTime = Union[Numeric, Time]


def unit_checker(value, unit: units.Quantity) -> units.Quantity:
    if unit.is_equivalent(value):
        return value
    else:
        if isinstance(value, units.Quantity):
            raise ValueError("Bad unit")
        else:
            return value * unit


def degree_checker(val, unit: units.Quantity = units.deg) -> units.Quantity:
    return unit_checker(val, unit)


def flux_checker(val, unit: units.Quantity = units.electron / units.s) -> units.Quantity:
    return unit_checker(val, unit)


def time_checker(val, unit: units.Quantity = units.s) -> units.Quantity:
    return unit_checker(val, unit)


def normalize(value: float, actual_min: float, actual_max: float, to_min=0.0, to_max=1.0) -> float:
    flip = actual_min > actual_max
    actual_min, actual_max = min(actual_min, actual_max), max(actual_min, actual_max)
    to_min, to_max = min(to_min, to_max), max(to_min, to_max)

    normalized = (value - actual_min) / (actual_max - actual_min)

    if flip:
        normalized = 1.0 - normalized

    return to_min + normalized * (to_max - to_min)


def interpolate(value: float, start_value: float, end_value: float) -> float:
    start_value, end_value = min(start_value, end_value), max(start_value, end_value)
    return start_value + (end_value - start_value) * value


def neighbor(array, value):
    return (np.abs(array - value)).argmin()
