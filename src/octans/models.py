from abc import ABC, abstractmethod

from .utils import NAngle, TimeError

from typing import List, Optional, Union, Callable, Literal, Dict, Iterable

import numpy as np
from astropy import units
from astropy.time import Time, TimeDelta

from astropy.coordinates import SkyCoord, EarthLocation
from lightkurve import LightCurve
from typing_extensions import Self


class ModelXLightCurve(ABC):
    @abstractmethod
    def from_lightkurve(self, lightkurve: LightCurve) -> Self:
        """Create a data object from a lightkurve"""

    @abstractmethod
    def to_hjd(self, sky: SkyCoord, loc: EarthLocation) -> Self:
        """Update time with Heliocentric Julian Date correction"""

    @abstractmethod
    def to_bjd(self, sky: SkyCoord, loc: EarthLocation) -> Self:
        """Update time with Barycentric Julian Date correction"""

    @abstractmethod
    def boundaries_extrema(self) -> List[List[int]]:
        """Return a 2D array of boundaries using deviations from average"""

    @abstractmethod
    def set_time(self, times: Time) -> None:
        """Set time"""

    @abstractmethod
    def update_time(self, amount: TimeDelta) -> None:
        """Update time"""

    @abstractmethod
    def minima_mean(self, boundaries: Optional[Union[np.ndarray, List]] = None) -> TimeError:
        """Find minima using the mean of time"""

    @abstractmethod
    def minima_median(self, boundaries: Optional[Union[np.ndarray, List]] = None) -> TimeError:
        """Find minima using the median of time"""

    @abstractmethod
    def minima_local(self, boundaries: Optional[Union[np.ndarray, List]] = None) -> TimeError:
        """Find minima using the lowest of time"""

    @abstractmethod
    def minima_fit(self, boundaries: Optional[Union[np.ndarray, List]] = None, deg: int = 2) -> TimeError:
        """Find minima using Polynomial fit"""

    @abstractmethod
    def minima_periodogram(self) -> TimeError:
        """Find minima using a periodogram"""

    @abstractmethod
    def minima_kwee_van_woerden(self, number_of_folds: int = 5, time_off: bool = True, init_min_flux: bool = False,
                                boundaries: Optional[Union[np.ndarray, List]] = None) -> TimeError:
        """Find minima using kwee van woerden method"""

    @abstractmethod
    def minima_bisector(self, middle_selector: Optional[Callable] = None, number_of_chords: int = 5,
                        sigma_multiplier: float = 0.1, boundaries: Optional[Union[np.ndarray, List]] = None,
                        fit_degree: int = 2) -> TimeError:
        """Find minima using chord method"""

    @abstractmethod
    def mid_egress_fielder(self, boundaries: Optional[Union[np.ndarray, List]] = None) -> TimeError:
        """Find minima using field method"""

    @abstractmethod
    def mid_egress_wood(self, boundaries: Optional[Union[np.ndarray, List]] = None, median_width: int = 5,
                        avg_width: int = 5, egress_width: float = 10.0) -> TimeError:
        """Find minima using wood method"""

    @abstractmethod
    def smooth_b_spline(self, s: float = 1.0) -> Self:
        """Smoothing lightcurve using spline method"""

    @abstractmethod
    def smooth_savgol_filter(self, window: int = 21, order: int = 2) -> Self:
        """Smoothing lightcurve using savgol method"""

    @abstractmethod
    def smooth_butterworth_filter(self, cutoff_freq: float = 0.5, sampling_rate: float = 10.0, order: int = 4) -> Self:
        """Smoothing lightcurve using butterworth filter method"""

    @abstractmethod
    def fold_periodogram(self, unit: Literal["ppm", "ppt"] = "ppm") -> Self:
        """Fold the lightcurve using periodogram"""

    @abstractmethod
    def fold_phase(self, minimum_time: Time, period: float) -> Self:
        """Fold the lightcurve using phase"""


class ModelSky(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the sky object"""

    @property
    @abstractmethod
    def skycoord(self) -> SkyCoord:
        """Return the SkyCoord of the sky object"""

    @property
    @abstractmethod
    def resolve(self) -> SkyCoord:
        """Resolves the name of the sky object"""

    @classmethod
    @abstractmethod
    def from_coordinates(cls, ra: NAngle, dec: NAngle, radius: NAngle = 2 * units.arcmin) -> Self:
        """Creates a Sky object from a given coordinate"""


class ModelPortal(ABC):
    @classmethod
    @abstractmethod
    def from_name(cls, name: str) -> Self:
        """Create a portal from a given name"""

    @classmethod
    @abstractmethod
    def from_coordinates(cls, ra: NAngle, dec: NAngle, radius: NAngle = 2 * units.arcmin) -> Self:
        """Create a portal from a given coordinates"""

    @property
    @abstractmethod
    def sky(self) -> ModelSky:
        """Return the sky Object fo the portal"""

    @abstractmethod
    def kkt(self, mission: Optional[Literal['kepler', 'k2', 'tess']] = None) -> List[ModelXLightCurve]:
        """Return a list of LightCurves of given mission"""

    @abstractmethod
    def kepler(self) -> List[ModelXLightCurve]:
        """Return a list of LightCurves of the kepler mission"""

    @abstractmethod
    def k2(self) -> List[ModelXLightCurve]:
        """Return a list of LightCurves of the kepler/k2 mission"""

    @abstractmethod
    def tess(self) -> List[ModelXLightCurve]:
        """Return a list of LightCurves of the tess mission"""

    @abstractmethod
    def all(self) -> List[ModelXLightCurve]:
        """Return a list of LightCurves of all kepler, kepler/k2, and tess missions"""
