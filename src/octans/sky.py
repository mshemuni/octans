from logging import Logger, getLogger
from typing import Optional

from typing_extensions import Self

from .errors import ObjectNotFoundError
from .utils import NAngleType, unit_checker
from .utils import degree_checker
from .model_sky import ModelSky

from astropy import units
from astropy.coordinates import SkyCoord

from astroquery.simbad import Simbad


class Sky(ModelSky):
    def __init__(self, name: str, logger: Optional[Logger] = None) -> None:
        if logger is None:
            self.logger = getLogger(__name__)
        else:
            self.logger = logger

        self.__name = name
        self.__skycoord = self.resolve

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name: {self.name})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.__name}')"

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name
        self.__skycoord = self.resolve

    @property
    def skycoord(self) -> SkyCoord:
        return self.__skycoord

    @skycoord.setter
    def skycoord(self, skycoord: SkyCoord) -> None:
        self.logger.error("This attribute is immutable and cannot be changed.")
        raise AttributeError("This attribute is immutable and cannot be changed.")

    @property
    def resolve(self) -> SkyCoord:
        return SkyCoord.from_name(self.__name)

    @classmethod
    def from_coordinates(cls, ra: NAngleType, dec: NAngleType, radius: NAngleType = 2 * units.arcmin,
                         logger: Optional[Logger] = None) -> Self:
        if logger is None:
            logger = getLogger(__name__)

        the_radius = unit_checker(radius, units.arcsec)
        the_ra = degree_checker(ra)
        the_dec = degree_checker(dec)
        coord = SkyCoord(ra=the_ra, dec=the_dec)
        result_table = Simbad.query_region(coord, radius=the_radius)

        if result_table is None:
            logger.error("No objects found.")
            raise ObjectNotFoundError("No objects found.")

        sky_coords = SkyCoord(result_table["RA"], result_table["DEC"], unit=(units.hourangle, units.deg))
        closest_ids, closest_dists, _ = coord.match_to_catalog_sky(sky_coords)
        return cls(result_table[closest_ids]["MAIN_ID"], logger=logger)
