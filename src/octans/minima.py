from logging import Logger
from typing import Optional, Iterator, Union
from typing_extensions import Self
import numpy as np

from astropy import units
from astropy.time import TimeDelta, Time

from .oc import OC
from .utils import Numeric, logger_checker


class Minima:
    def __init__(self, time: Time, error: Optional[TimeDelta] = None, logger: Optional[Logger] = None) -> None:
        self.logger = logger_checker(logger, __name__)
        self.time = time
        self.error = self.__verify(error)

        self.time, self.error = self.__resort()

    def __verify(self, error: Optional[TimeDelta] = None) -> TimeDelta:
        if error is None:
            if self.time.isscalar:
                return TimeDelta(np.nan, format='jd')
            else:
                nans = np.empty((len(self.time),))
                nans[:] = np.nan
                return TimeDelta(nans, format='jd')
        else:
            if self.time.isscalar:
                if not error.isscalar:
                    self.logger.error("Both time and error must be either scalar or list")
                    raise ValueError("Both time and error must be either scalar or list")
            else:
                if error.isscalar:
                    self.logger.error("Both time and error must be either scalar or list")
                    raise ValueError("Both time and error must be either scalar or list")

                if len(self.time) != len(error):
                    self.logger.error("Both time and error must be of same length")
                    raise ValueError("Both time and error must be of same length")

            return error

    def __resort(self):
        if self.time.isscalar:
            return self.time, self.error

        sorted_indices = np.argsort(self.time.jd)
        time = self.time.jd[sorted_indices]
        error = self.error.jd[sorted_indices]

        return Time(time, format='jd'), TimeDelta(error, format='jd')

    def __str__(self) -> str:
        if self.time.isscalar:
            return f"{self.time} ± {self.error}"

        return f"time:{self.time}, error:{self.error}"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        if self.time.isscalar:
            return 1

        return len(self.time)

    def __iter__(self) -> Iterator[Self]:
        if self.time.isscalar:
            yield self.__class__(self.time, self.error)
        else:
            for time, error in zip(self.time, self.error):
                yield self.__class__(time, error)

    def oc(self, minimum_time: Optional[Union[Self, Time, Numeric]] = None,
           period: Optional[Union[units.Quantity["time"], Numeric]] = None) -> OC:
        """

        Calculates the difference between the observed times and the calculated times of minima based on a given minimum time and period.

        Parameters
        ----------
        minimum_time: Union[Self, Time, Numeric]
            The minimum time to be used as a reference for calculations. This can be provided in various formats:
            - If None, the first time in the object's time attribute is used.
            - If of type Time, it is used directly.
            - If of the same class as the object, it uses the time attribute of the provided object.
            - Otherwise, it converts the provided numeric value to a Time object formatted in Julian Date (JD).
        period: Union[units.Quantity["time"], Numeric], default = None
            The time period for calculations. This can be provided in different formats:
            - If None, it is calculated as the mean of the sorted differences of the time attribute in JD, multiplied by 1 day.
            - If provided as a numeric value, it is assumed to be in days.
            - If provided as a quantity with a time dimension, it is used directly.

        Returns
        -------
        Time
            The differences between the observed times and the calculated times of the minima.
        """
        if minimum_time is None:
            the_minimum_time = self.time[0]
        else:
            if isinstance(minimum_time, Time):
                the_minimum_time = minimum_time
            elif isinstance(minimum_time, self.__class__):
                the_minimum_time = minimum_time.time
            else:
                the_minimum_time = Time(minimum_time, format='jd')

        if period is None:
            the_period = np.sort(np.diff(self.time.jd)).mean() * units.day
        else:
            if isinstance(period, (float, int)):
                the_period = period * units.day
            else:
                the_period = period

        epochs = np.round((self.time - the_minimum_time) / the_period)
        calculated_minima = the_minimum_time + epochs * the_period

        return OC(self.time, self.time - calculated_minima, logger=self.logger)

    def append(self, other: Self) -> Self:
        """
        Appends two Minima objects

        Parameters
        ----------
        other : Self
            Another instance of minima

        Returns
        -------
        Self
            A new instance of Minima
        """
        return self.__class__(
            Time(self.time.jd.tolist() + other.time.jd.tolist(), format='jd'),
            TimeDelta(self.error.jd.tolist() + other.error.jd.tolist(), format='jd'),
            logger=self.logger
        )
