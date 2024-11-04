from abc import ABC, abstractmethod
from typing_extensions import Self

from astropy import units
from astropy.coordinates import SkyCoord

from .utils import NAngleType


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
    def from_coordinates(cls, ra: NAngleType, dec: NAngleType, radius: NAngleType = 2 * units.arcmin) -> Self:
        """Creates a Sky object from a given coordinate"""
