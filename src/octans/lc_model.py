from abc import ABC, abstractmethod
from logging import Logger
from typing import Self, Iterator, Optional, Callable, Any

from astropy.time import Time
from lightkurve import LightCurve

from .utils import Boundaries, Minima


class ModelXLightCurve(ABC):
    @classmethod
    @abstractmethod
    def from_lightcurve(cls, lightcurve: LightCurve, logger: Optional[Logger] = None):
        """Create an XLightCurve object from a LightCurve object"""

    @abstractmethod
    def boundaries_extrema(self, sigma_multiplier: float = 1.0) -> Boundaries:
        """Find boundaries of each minimum using curve fit"""

    @abstractmethod
    def boundaries_period(self, t0: Time, period: float, gap: float = 0.1) -> Boundaries:
        """Find boundaries of each minimum using period"""

    @abstractmethod
    def chop(self, boundaries: Optional[Boundaries] = None) -> Iterator[Self]:
        """Chops the XLightCurve according to boundaries"""

    @abstractmethod
    def smooth_b_spline(self, s: float = 1.0) -> Self:
        """Smooths the lightcurve using splrep"""

    @abstractmethod
    def smooth_savgol_filter(self, window: int = 21, order: int = 2) -> Self:
        """Smooths the lightcurve using savgol filter"""

    @abstractmethod
    def smooth_butterworth_filter(self, cutoff_freq: float = 0.5, sampling_rate: float = 10.0,
                                  order: int = 4) -> Self:
        """Smooths the lightcurve using butterworth filter"""

    @abstractmethod
    def minima_mean(self, boundaries: Optional[Boundaries] = None) -> Minima:
        """"Returns minima times using mean value of flux in each minima"""

    @abstractmethod
    def minima_median(self, boundaries: Optional[Boundaries] = None) -> Minima:
        """Returns minima times using median value of flux in each minima"""

    @abstractmethod
    def minima_local(self, boundaries: Optional[Boundaries] = None) -> Minima:
        """Returns minima times using local minima value of flux in each minima"""

    @abstractmethod
    def minima_fit(self, boundaries: Optional[Boundaries] = None, deg: int = 2) -> Minima:
        """Returns minima times by fitting a polynomial and finding the root of it for each minima"""

    @abstractmethod
    def minima_periodogram(self) -> Minima:
        """Returns minima times using periodogram of each minima"""

    @abstractmethod
    def minima_kwee_van_woerden(self, number_of_folds: int = 5, time_off: bool = True, init_min_flux: bool = False,
                                boundaries: Optional[Boundaries] = None) -> Minima:
        """Returns minima times using Kwee Van Woerden of each minima"""

    @abstractmethod
    def minima_bisector(self, middle_selector: Optional[Callable[[Any], float]] = None, number_of_chords: int = 5,
                        sigma_multiplier: float = 0.1, boundaries: Optional[Boundaries] = None) -> Minima:
        """Returns minima times using bisector of each minima"""

    @abstractmethod
    def mid_egress_fielder(self, boundaries: Optional[Boundaries] = None) -> Minima:
        """Returns mid-egress time of each minima using Fielder method"""

    @abstractmethod
    def mid_ingress_fielder(self, boundaries: Optional[Boundaries] = None) -> Minima:
        """Returns mid-ingress time of each minima using Fielder method"""

    @abstractmethod
    def minima_fielder(self, boundaries: Optional[Boundaries] = None) -> Minima:
        """Returns minima times using Fielder method"""

    @abstractmethod
    def mid_egress_wood(self, boundaries: Optional[Boundaries] = None, median_width: int = 5,
                        avg_width: int = 5, egress_width: float = 10.0) -> Minima:
        """Returns mid-egress time of each minima using Wood method"""

    @abstractmethod
    def minima_thoroughgood(self, boundaries: Optional[Boundaries] = None) -> Minima:
        """Returns minima times using Thoroughgood method"""
