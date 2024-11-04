from logging import getLogger, Logger

import pandas as pd
import requests
from typing import Optional

from astropy import units
from astropy.coordinates import SkyCoord
from tqdm import tqdm

from . import Sky
from .model_catalgues import ExoplanetEUModel, NASAExoplanetArchiveModel, PeriodModel
from .errors import NoExoplanetEUObjectError, NoNASAExoplanetArchiveObjectError
from .utils import database_dir, NAngleType, degree_checker

from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive


class ExoplanetEU(ExoplanetEUModel):
    def __init__(self, sky: Sky, max_dist: NAngleType = 20 * units.arcsec, logger: Optional[Logger] = None) -> None:
        if logger is None:
            self.logger = getLogger(__name__)
        else:
            self.logger = logger

        self.database_file = database_dir() / "exoplanet_eu.csv"
        self.url = "https://exoplanet.eu/catalog/csv/"

        self.sky = sky
        self.max_dist = max_dist
        self.the_max_dist = degree_checker(self.max_dist)
        self.database = self.load()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.sky})"

    def __repr__(self) -> str:
        return self.__str__()

    def exist(self) -> bool:
        return bool(self.database_file.exists())

    def download(self, force=False) -> None:
        if self.exist() and not force:
            self.logger.warning("Databse already exist. If you want to force download it use `force=True`")
            return

        with (open(self.database_file, "w", encoding="utf-8") as database,
              requests.get(self.url, stream=True) as response):
            response.raise_for_status()
            for chunk in tqdm(response.iter_content(chunk_size=1024)):
                if chunk:
                    database.write(chunk.decode("utf-8"))

    def load(self) -> pd.DataFrame:
        if not self.exist():
            self.download()

        try:
            return pd.read_csv(self.database_file)
        except Exception as e:
            self.logger.warning(e)

        self.download(force=True)
        return pd.read_csv(self.database_file)

    def resolve(self) -> pd.DataFrame:
        database = self.database[self.database.ra.notna() | self.database.dec.notna()]
        skycoord = SkyCoord(ra=database["ra"], dec=database["dec"], unit="deg")

        closest_ids, closest_dists, _ = skycoord.match_to_catalog_sky(SkyCoord([self.sky.skycoord]))
        mask = closest_dists < self.the_max_dist

        if mask.sum() == 0:
            raise NoExoplanetEUObjectError("There's no such ExoplanetEU object")

        data = database[mask]
        return data

    def period(self) -> pd.DataFrame:
        data = self.resolve()
        data = data[["name", "orbital_period", "orbital_period_error_min", "orbital_period_error_max"]]
        data.rename(columns={
            'orbital_period': 'P',
            'orbital_period_error_min': 'PEmin',
            'orbital_period_error_max': 'PEmax'
        }, inplace=True)
        data = data.reset_index()
        return data


class NASAExoplanetArchive(NASAExoplanetArchiveModel):
    def __init__(self, sky: Sky, max_dist: NAngleType = 20 * units.arcsec, logger: Optional[Logger] = None) -> None:
        if logger is None:
            self.logger = getLogger(__name__)
        else:
            self.logger = logger

        self.sky = sky
        self.max_dist = max_dist
        self.the_max_dist = degree_checker(self.max_dist)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.sky})"

    def __repr__(self) -> str:
        return self.__str__()

    def resolve(self) -> pd.DataFrame:
        table = NasaExoplanetArchive.query_region(table="pscomppars", coordinates=self.sky.skycoord,
                                                  radius=self.max_dist)
        closest_ids, closest_dists, _ = table["sky_coord"].match_to_catalog_sky(SkyCoord([self.sky.skycoord]))
        mask = closest_dists < self.the_max_dist

        if mask.sum() == 0:
            raise NoNASAExoplanetArchiveObjectError("There's no such ExoplanetEU object")

        data = table.to_pandas()[mask]
        return data

    def period(self) -> pd.DataFrame:
        data = self.resolve()
        data = data[["pl_name", "pl_orbper", "pl_orbpererr1", "pl_orbpererr2"]]
        data.rename(
            columns={
                'pl_name': 'name',
                'pl_orbper': 'P',
                'pl_orbpererr1': 'PEmin',
                'pl_orbpererr2': 'PEmax'
            },
            inplace=True)
        data = data.reset_index()
        return data


class Period(PeriodModel):
    def __init__(self, sky: Sky, max_dist: NAngleType = 20 * units.arcsec, logger: Optional[Logger] = None):
        if logger is None:
            self.logger = getLogger(__name__)
        else:
            self.logger = logger

        self.sky = sky
        self.max_dist = max_dist
        self.the_max_dist = degree_checker(self.max_dist)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.sky})"

    def __repr__(self) -> str:
        return self.__str__()

    def nasa_exoplanet_archive(self) -> pd.DataFrame:
        nasa = NASAExoplanetArchive(self.sky, max_dist=self.max_dist, logger=self.logger)
        return nasa.period()

    def exoplanet_eu(self) -> pd.DataFrame:
        eu = ExoplanetEU(self.sky, max_dist=self.max_dist, logger=self.logger)
        return eu.period()

    def all(self) -> pd.DataFrame:
        nasa = NASAExoplanetArchive(self.sky, max_dist=self.max_dist, logger=self.logger)
        nasa_data = nasa.period()
        nasa_data["source"] = "NEA"
        eu = ExoplanetEU(self.sky, max_dist=self.max_dist, logger=self.logger)
        eu_data = eu.period()
        eu_data["source"] = "EEU"
        data = pd.concat([nasa_data, eu_data], ignore_index=True)
        del data["index"]
        return data[["source", "P", "PEmin", "PEmax"]]
