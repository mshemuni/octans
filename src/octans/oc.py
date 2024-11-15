from logging import Logger
from typing import Iterator, Optional

import numpy as np
from astropy.time import Time, TimeDelta
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.interpolate import splrep, splev
from typing_extensions import Self

from .utils import logger_checker


class OC:
    def __init__(self, time: Time, oc: TimeDelta, logger: Optional[Logger] = None):
        self.logger = logger_checker(logger, __name__)
        self.time = time
        self.oc = oc
        self.__verify()

    def __verify(self) -> None:
        if self.time.isscalar:
            if not self.oc.isscalar:
                self.logger.error("Both time and oc must be either scalar or list")
                raise ValueError("Both time and oc must be either scalar or list")
        else:
            if self.oc.isscalar:
                self.logger.error("Both time and oc must be either scalar or list")
                raise ValueError("Both time and oc must be either scalar or list")

            if len(self.time) != len(self.oc):
                self.logger.error("Both time and oc must be of same length")
                raise ValueError("Both time and oc must be of same length")

    def __str__(self) -> str:
        if self.time.isscalar:
            return f"{self.time}({self.oc})"

        return f"time:{self.time}, oc:{self.oc}"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        if self.time.isscalar:
            return 1

        return len(self.time)

    def __iter__(self) -> Iterator[Self]:
        if self.time.isscalar:
            yield self.__class__(self.time, self.oc)
        else:
            for time, error in zip(self.time, self.oc):
                yield self.__class__(time, error)

    def plot(self, ax: Optional[Axes] = None, **kwargs) -> None:
        """
        Plots teh OC to the given axis
        """

        the_ax = plt.gca() if ax is None else ax

        the_ax.plot(self.time.jd, self.oc.jd, **kwargs)

    def smooth(self, s: float = 1.0) -> Self:
        """
        Generates a smoothed version of the OC using B-spline fitting.

        Parameters
        ----------
        s: float, default = 1.0
            Smoothing factor used in the B-spline fitting. Default is 1.0.

        Returns
        -------
        Self
            An instance of the OC class with smoothed o-c values.
        """
        oc_mask = np.isnan(self.oc.jd)
        masked_oc = self.oc[~oc_mask]
        masked_time = self.time[~oc_mask]
        unique_values, inverse_indices = np.unique(masked_time.jd, return_inverse=True)
        mean_values = np.array([np.mean(masked_oc.jd[inverse_indices == i]) for i in range(len(unique_values))])

        bspl = splrep(unique_values, mean_values, s=s)

        return self.__class__(
            time=Time(unique_values, format="jd"),
            oc=TimeDelta(splev(unique_values, bspl), format="jd"),
            logger=self.logger
        )

    def append(self, other: Self) -> Self:
        """
        Appends two OC objects

        Parameters
        ----------
        other : Self
            Another instance of oc

        Returns
        -------
        Self
            A new instance of OC
        """
        return self.__class__(
            Time(self.time.jd.tolist() + other.time.jd.tolist(), format='jd'),
            TimeDelta(self.oc.jd.tolist() + other.oc.jd.tolist(), format='jd'),
            logger=self.logger
        )
