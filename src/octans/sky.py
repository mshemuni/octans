from typing_extensions import Self

from .utils import NAngle
from .utils import degree_checker

from .models import ModelSky

from astropy import units
from astropy.coordinates import SkyCoord

from astroquery.simbad import Simbad


class Sky(ModelSky):
    def __init__(self, name: str) -> None:
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
        raise AttributeError("This attribute is immutable and cannot be changed.")

    @property
    def resolve(self) -> SkyCoord:
        return SkyCoord.from_name(self.__name)

    @classmethod
    def from_coordinates(cls, ra: NAngle, dec: NAngle, radius: NAngle = 2 * units.arcmin) -> Self:
        the_ra = degree_checker(ra)
        the_dec = degree_checker(dec)
        the_radius = degree_checker(radius, unit=units.arcmin)
        coord = SkyCoord(ra=the_ra, dec=the_dec)
        result_table = Simbad.query_region(coord, radius=the_radius)

        if result_table is None:
            raise ValueError("No objects found.")

        sky_coords = SkyCoord(result_table["RA"], result_table["DEC"], unit=(units.hourangle, units.deg))
        closest_ids, closest_dists, closest_dists3d = coord.match_to_catalog_sky(sky_coords)
        return cls(result_table[closest_ids]["MAIN_ID"])
