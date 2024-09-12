from typing import Union
from astropy import units
import numpy as np
from dataclasses import dataclass, field
from astropy.time import Time as AstTime, TimeDelta as AstTimeDelta
from typing_extensions import Self


Numeric = Union[int, float]
Angle = units.Quantity["angle"]
Time = units.Quantity["time"]
NAngle = Union[Numeric, Angle]
NTime = Union[Numeric, Time]


@dataclass
class TimeError:
    time: AstTime
    error: AstTimeDelta

    def __str__(self):
        if self.time.isscalar:
            return f"{self.time.jd} ± {self.error.jd}"

        return super.__str__(self)

    def __post_init__(self):
        if self.time.isscalar:
            if not self.error.isscalar:
                raise ValueError("Length of time must be equal to length of error")
        else:
            if len(self.time) != len(self.error):
                raise ValueError("Length of time must be equal to length of error")

    def __getitem__(self, key: Union[int, slice]) -> Self:
        return TimeError(self.time[key], self.error[key])


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
