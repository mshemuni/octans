from abc import ABC, abstractmethod
from logging import Logger
from typing import Self, Optional, Literal, Iterator

from astropy import units

from . import Sky, XLightCurve
from .utils import NAngleType, Minima


class ModelPortal(ABC):
    @classmethod
    @abstractmethod
    def from_name(cls, name: str) -> Self:
        """Create a portal object from a given name"""

    @classmethod
    @abstractmethod
    def from_coordinates(cls, ra: NAngleType, dec: NAngleType, radius: NAngleType = 2 * units.arcmin) -> Self:
        """Create a portal object from a given coordinates"""

    @property
    @abstractmethod
    def sky(self) -> Sky:
        """Return the sky object related to the portal"""

    @sky.setter
    @abstractmethod
    def sky(self, sky: Sky) -> None:
        """Prevents setting the sky value"""

    @abstractmethod
    def kt(self, mission: Optional[Literal['kepler', 'tess']] = None) -> Iterator[XLightCurve]:
        """Retrieves either kepler, k2 or tess lightcurve from server"""

    @abstractmethod
    def kepler(self) -> Iterator[XLightCurve]:
        """Retrieves kepler lightcurve from server"""

    @abstractmethod
    def tess(self) -> Iterator[XLightCurve]:
        """Retrieves tess lightcurve from server"""

    @abstractmethod
    def asas(self) -> Iterator[XLightCurve]:
        """Retrieves ASAS lightcurve from server"""

    @abstractmethod
    def var_astro(self) -> Minima:
        """Returns minimas from VarAstro"""

    @abstractmethod
    def oc_gateway(self) -> Minima:
        """Returns minimas from VarAstro"""

    @abstractmethod
    def etd(self) -> Minima:
        """Returns minimas from VarAstro ETD"""


class ModelASAS(ABC):
    @classmethod
    @abstractmethod
    def from_sky(cls, sky: Sky, max_dist: NAngleType = 20 * units.arcsec, time_out: NAngleType = 3 * units.s) -> Self:
        """Retrieves the asas_id of the given coordinates"""

    @abstractmethod
    def data(self) -> str:
        """Retrieves the raw lightcurve"""

    @abstractmethod
    def get(self) -> Iterator[XLightCurve]:
        """Retrieves the XLightCurve """


class ModelVarAstro(ABC):
    @classmethod
    @abstractmethod
    def from_sky(cls, sky: Sky, max_dist: NAngleType = 20 * units.arcmin, time_out: NAngleType = 3 * units.s,
                 logger: Optional[Logger] = None) -> Self:
        """Creates a VarAstro object from a given coordinates"""

    @abstractmethod
    def get(self) -> Minima:
        """Retrieves the minima times from VarAstro"""
