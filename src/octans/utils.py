import platform
from dataclasses import dataclass
from logging import Logger, getLogger
from pathlib import Path
from typing import Union, Optional, Iterator

from typing_extensions import Self

import numpy as np
from astropy import units
from astropy.time import Time, TimeDelta

Numeric = Union[int, float]
AngleType = units.Quantity["angle"]
TimeType = units.Quantity["time"]
NAngleType = Union[Numeric, units.Quantity["angle"]]
NTimeType = Union[Numeric, units.Quantity["time"]]


@dataclass
class Boundaries:
    lower: Time
    upper: Time
    logger: Optional[Logger] = None

    def __str__(self) -> str:
        if self.lower.isscalar:
            return f"[{self.lower.jd}, {self.upper.jd}]"

        return str([each for each in self])

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.lower)

    def __iter__(self) -> Iterator[Self]:
        for lower, upper in zip(self.lower, self.upper):
            yield self.__class__(lower, upper)

    def __post_init__(self) -> None:

        if self.logger is None:
            self.logger = getLogger(__name__)

        if self.lower.isscalar:
            if not self.upper.isscalar:
                self.logger.error("Number of Lower and Upper bounds must match")
                raise ValueError("Number of Lower and Upper bounds must match")

        else:
            if len(self.lower) != len(self.upper):
                self.logger.error("Number of Lower and Upper bounds must match")
                raise ValueError("Number of Lower and Upper bounds must match")

    def __getitem__(self, key: Union[int, slice]) -> Self:
        return self.__class__(self.lower[key], self.upper[key])


@dataclass
class Minima:
    time: Time
    error: TimeDelta
    logger: Optional[Logger] = None

    def __str__(self) -> str:
        if self.time.isscalar:
            return f"{self.time.jd} ± {self.error.jd}"

        return str([each for each in self])

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.time)

    def __iter__(self) -> Iterator[Self]:
        for time, error in zip(self.time, self.error):
            yield self.__class__(time, error)

    def __post_init__(self) -> None:
        if self.logger is None:
            self.logger = getLogger(__name__)

        if self.time.isscalar:
            if not self.error.isscalar:
                self.logger.error("Length of time must be equal to length of error")
                raise ValueError("Length of time must be equal to length of error")

        else:
            if len(self.time) != len(self.error):
                self.logger.error("Length of time must be equal to length of error")
                raise ValueError("Length of time must be equal to length of error")

    def __getitem__(self, key: Union[int, slice]) -> Self:
        return self.__class__(self.time[key], self.error[key])


def normalize(value: float, actual_min: float, actual_max: float, to_min=0.0, to_max=1.0) -> float:
    flip = actual_min > actual_max
    actual_min, actual_max = min(actual_min, actual_max), max(actual_min, actual_max)
    to_min, to_max = min(to_min, to_max), max(to_min, to_max)

    normalized = (value - actual_min) / (actual_max - actual_min)

    if flip:
        normalized = 1.0 - normalized

    return float(to_min + normalized * (to_max - to_min))


def interpolate(value: float, start_value: float, end_value: float) -> float:
    start_value, end_value = min(start_value, end_value), max(start_value, end_value)
    return start_value + (end_value - start_value) * value


def neighbor(array, value):
    return (np.abs(array - value)).argmin()


def unit_checker(value, unit: units.Quantity) -> units.Quantity:
    if isinstance(value, (float, int)):
        return value * unit

    if unit.is_equivalent(value):
        return value
    else:
        raise ValueError("Bad unit")


def degree_checker(value, unit: units.Quantity = units.deg) -> units.Quantity:
    return unit_checker(value, unit)


def time_checker(value, unit: units.Quantity = units.s) -> units.Quantity:
    return unit_checker(value, unit)


def database_dir():
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
