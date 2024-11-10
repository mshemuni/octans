from logging import Logger

import numpy as np
import pandas as pd
import requests
from typing import Optional, Self

from astropy import units
from astropy.coordinates import SkyCoord

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException

from tqdm import tqdm

from . import Sky
from .model_catalogues import CatalogueModel, PeriodModel
from .errors import NoExoplanetEUObjectError, NoNASAExoplanetArchiveObjectError, NoVarAstroObjectError
from .utils import database_dir, NAngleType, degree_checker, logger_checker, time_checker
from .portal import VarAstro as VarAstroPortal, VarAstro_XPATHS

from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive


class ExoplanetEU(CatalogueModel):
    def __init__(self, sky: Sky, max_dist: NAngleType = 20 * units.arcmin, verbose: bool = False,
                 logger: Optional[Logger] = None) -> None:
        self.logger = logger_checker(logger, __name__)
        self.verbose = verbose

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

    @classmethod
    def from_name(cls, name: str, max_dist: NAngleType = 20 * units.arcmin,
                  verbose: bool = False, logger: Optional[Logger] = None) -> Self:
        """
        Creates an ExoplanetEU object from a given name

        Parameters
        ----------
        name : str
            The name of the object
        max_dist : NAngleType, default=20 * units.arcsec
            The maximum distance for XMatch
        verbose : bool, default=False
            Verbosity
        logger : Optional[Logger], default=None
            A logger

        Returns
        -------
        ExoplanetEU
            An ExoplanetEu object from a given object name
        """
        sky = Sky(name)
        return cls(sky, verbose=verbose, max_dist=max_dist, logger=logger)

    def exist(self) -> bool:
        """
        Checks if the database file exists in the file system.

        Returns
        -------
        bool
            True if the database file exists, False otherwise.
        """
        return bool(self.database_file.exists())

    def download(self, force=False) -> None:
        """
        Downloads the database from the URL.

        Parameters
        ----------
        force : bool
            If True, forces the download even if the database already exists. Default is False.

        Returns
        -------
            None
        """
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
        """
        Loads data from the database file. If the database does not exist, it attempts to download it.
        If an error occurs while loading the database file, it retries a forceful download.

        Returns
        -------
            pd.DataFrame: The data loaded from the CSV file.
        """
        if not self.exist():
            self.download()

        try:
            return pd.read_csv(self.database_file)
        except Exception as e:
            self.logger.warning(e)

        self.download(force=True)
        return pd.read_csv(self.database_file)

    def resolve(self) -> pd.DataFrame:
        """
        Resolves and XMatches the current database based on the provided sky object

        Returns
        -------
            pd.DataFrame: The filtered database that matches certain sky coordinate criteria.
        """
        database = self.database[self.database.ra.notna() | self.database.dec.notna()]
        skycoord = SkyCoord(ra=database["ra"], dec=database["dec"], unit="deg")

        closest_ids, closest_dists, _ = skycoord.match_to_catalog_sky(SkyCoord([self.sky.skycoord]))
        mask = closest_dists < self.the_max_dist

        if mask.sum() == 0:
            raise NoExoplanetEUObjectError("There's no such ExoplanetEU object")

        data = database[mask]
        return data

    def period(self) -> pd.DataFrame:
        """
        Retrieves the period from ExoplanetEU database.

        Returns
        -------
            DataFrame: A pandas DataFrame containing the columns "name",
            "P", "PEmin", and "PEmax" with data reset to default indexing.
        """
        data = self.resolve()
        data = data[["name", "orbital_period", "orbital_period_error_min", "orbital_period_error_max"]]
        data.rename(columns={
            'orbital_period': 'P',
            'orbital_period_error_min': 'PEmin',
            'orbital_period_error_max': 'PEmax'
        }, inplace=True)
        data = data.reset_index()
        return data


class NASAExoplanetArchive(CatalogueModel):
    def __init__(self, sky: Sky, max_dist: NAngleType = 20 * units.arcmin, verbose: bool = False,
                 logger: Optional[Logger] = None) -> None:
        self.logger = logger_checker(logger, __name__)
        self.verbose = verbose

        self.sky = sky
        self.max_dist = max_dist
        self.the_max_dist = degree_checker(self.max_dist)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.sky})"

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def from_name(cls, name: str, max_dist: NAngleType = 20 * units.arcmin,
                  verbose: bool = False, logger: Optional[Logger] = None) -> Self:
        """
        Creates a NASAExoplanetArchive object from a given name

        Parameters
        ----------
        name : str
            The name of the object
        max_dist : NAngleType, default=20 * units.arcsec
            The maximum distance for XMatch
        verbose : bool, default=False
            Verbosity
        logger : Optional[Logger], default=None
            A logger

        Returns
        -------
        ExoplanetEU
            A NASAExoplanetArchive object from a given object name
        """
        sky = Sky(name)
        return cls(sky, verbose=verbose, max_dist=max_dist, logger=logger)

    def resolve(self) -> pd.DataFrame:
        """
        Resolves and XMatches the current database based on the provided sky object

        Returns
        -------
            pd.DataFrame: The filtered database that matches certain sky coordinate criteria.
        """
        table = NasaExoplanetArchive.query_region(table="pscomppars", coordinates=self.sky.skycoord,
                                                  radius=self.max_dist)
        closest_ids, closest_dists, _ = table["sky_coord"].match_to_catalog_sky(SkyCoord([self.sky.skycoord]))
        mask = closest_dists < self.the_max_dist

        if mask.sum() == 0:
            raise NoNASAExoplanetArchiveObjectError("There's no such ExoplanetEU object")

        data = table.to_pandas()[mask]
        return data

    def period(self) -> pd.DataFrame:
        """
        Retrieves the period from NASAExoplanetArchive database.

        Returns
        -------
            DataFrame: A pandas DataFrame containing the columns "name",
            "P", "PEmin", and "PEmax" with data reset to default indexing.
        """
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


class VarAstro(CatalogueModel):
    def __init__(self, sky: Sky, max_dist: NAngleType = 20 * units.arcmin, verbose: bool = False,
                 logger: Optional[Logger] = None) -> None:
        self.logger = logger_checker(logger, __name__)
        self.verbose = verbose

        self.sky = sky
        self.max_dist = max_dist
        self.the_max_dist = degree_checker(self.max_dist)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.sky})"

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def from_name(cls, name: str, max_dist: NAngleType = 20 * units.arcmin,
                  verbose: bool = False, logger: Optional[Logger] = None) -> Self:
        sky = Sky(name)
        return cls(sky, verbose=verbose, max_dist=max_dist, logger=logger)

    def get_id(self) -> int:
        var_astro = VarAstroPortal.from_sky(self.sky)
        return var_astro.var_astro_id

    def var_astro_id(self, time_out: NAngleType = 3 * units.s) -> int:
        the_max_dist = degree_checker(self.max_dist)
        the_time_out = time_checker(time_out)
        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

        driver.get(
            f"https://var.astro.cz/en/Stars?pageId=1&pageSize=20&ra="
            f"{self.sky.skycoord.ra.degree}&dec={self.sky.skycoord.dec.degree}&radiusArcmin="
            f"{int(the_max_dist.to(units.arcmin).value)}")

        WebDriverWait(driver, the_time_out.to(units.s).value).until(
            EC.presence_of_element_located((By.XPATH, VarAstro_XPATHS["result_table_first_row"]))
        )

        rows = driver.find_elements(By.XPATH, VarAstro_XPATHS["result_table_first_row"])

        data = rows[0].find_elements(By.TAG_NAME, "td")
        if data[0].text.strip() == "No data available in table":
            self.logger.error("There's no such VarAstro object")
            raise NoVarAstroObjectError("There's no such VarAstro object")

        var_astro_id = int(data[0].text.strip())

        driver.close()
        return var_astro_id

    def resolve(self) -> pd.DataFrame:
        """
        Resolves and XMatches the current database based on the provided sky object

        Returns
        -------
            pd.DataFrame: The filtered database that matches certain sky coordinate criteria.
        """
        respond = requests.get(f"https://var.astro.cz/en/Stars/{self.var_astro_id()}")
        if respond.status_code != 200:
            self.logger.error("There's no such ASAS object")
            raise NoNASAExoplanetArchiveObjectError("There's no such VarAstro object")

        page_soup = BeautifulSoup(respond.text, 'html.parser')
        cards = page_soup.find_all("div", class_="card")
        data = {"source": "varastro", "name": self.sky.name, "Period (d)": None, "Epoch (JD)": None, "Secondary Epoch (JD)": None}
        for card in cards:
            if "Periodic" in card.text.strip():
                lis = card.find_all("li")
                for li in lis:
                    key = " ".join(li.text.split(":")[:-1]).strip()
                    value = li.find("strong").text.strip()
                    data[key] = float(value)
                break
        return pd.DataFrame([data])[["source", "name", "Period (d)", "Epoch (JD)", "Secondary Epoch (JD)"]]

    def period(self) -> pd.DataFrame:
        data = self.resolve()
        data["PEmin"] = np.nan
        data["PEmax"] = np.nan
        data.rename(
            columns={
                'Period (d)': 'P',
            },
            inplace=True)
        data = data.reset_index()
        return data[["name", "source", "P", "PEmin", "PEmax"]]


class Period(PeriodModel):
    def __init__(self, sky: Sky, max_dist: NAngleType = 20 * units.arcmin, verbose: bool = False,
                 logger: Optional[Logger] = None):
        self.logger = logger_checker(logger, __name__)
        self.verbose = verbose

        self.sky = sky
        self.max_dist = max_dist
        self.the_max_dist = degree_checker(self.max_dist)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.sky})"

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def from_name(cls, name: str, max_dist: NAngleType = 20 * units.arcmin,
                  verbose: bool = False, logger: Optional[Logger] = None) -> Self:
        """
        Creates a Period object from a given name

        Parameters
        ----------
        name : str
            The name of the object
        max_dist : NAngleType, default=20 * units.arcsec
            The maximum distance for XMatch
        verbose : bool, default=False
            Verbosity
        logger : Optional[Logger], default=None
            A logger

        Returns
        -------
        Period
            An Period object from a given object name
        """
        sky = Sky(name)
        return cls(sky, verbose=verbose, max_dist=max_dist, logger=logger)

    def nasa_exoplanet_archive(self) -> pd.DataFrame:
        """
        Fetches exoplanet data from the NASA Exoplanet Archive and returns it as a DataFrame.

        The function initializes a NASAExoplanetArchive object using the current instance's sky, max_dist,
        and logger attributes. It then calls the period method on the NASAExoplanetArchive object to retrieve
        the exoplanet data.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing exoplanet data from the NASA Exoplanet Archive.
        """
        nasa = NASAExoplanetArchive(self.sky, max_dist=self.max_dist, logger=self.logger)
        return nasa.period()

    def exoplanet_eu(self) -> pd.DataFrame:
        """
        Retrieve exoplanet data from the ExoplanetEU dataset


        Returns
        -------
        pd.DataFrame
            A DataFrame containing the exoplanet data, including the period of each exoplanet
        """
        eu = ExoplanetEU(self.sky, max_dist=self.max_dist, logger=self.logger)
        return eu.period()

    def var_astro(self) -> pd.DataFrame:
        """
        Retrieve exoplanet data from the ExoplanetEU dataset


        Returns
        -------
        pd.DataFrame
            A DataFrame containing the exoplanet data, including the period of each exoplanet
        """
        va = VarAstro(self.sky, max_dist=self.max_dist, logger=self.logger)
        return va.period()

    def all(self) -> pd.DataFrame:
        """
        Retrieve and consolidate exoplanet data from NASA Exoplanet Archive and ExoplanetEU.

        Returns
        -------
        pd.DataFrame
            A consolidated DataFrame containing the period data from both the NASA Exoplanet Archive and
            ExoplanetEU with the sources identified. The DataFrame includes columns for 'source', 'P', 'PEmin', and 'PEmax'.
        """
        try:
            nasa = NASAExoplanetArchive(self.sky, max_dist=self.max_dist, logger=self.logger)
            nasa_data = nasa.period()
            nasa_data["source"] = "NEA"
        except Exception as e:
            nasa_data = pd.DataFrame(columns=["source", "name", "P", "PEmin", "PEmax"])
            self.logger.warning(e)

        try:
            eu = ExoplanetEU(self.sky, max_dist=self.max_dist, logger=self.logger)
            eu_data = eu.period()
            eu_data["source"] = "EEU"
        except Exception as e:
            eu_data = pd.DataFrame(columns=["source", "name", "P", "PEmin", "PEmax"])
            self.logger.warning(e)

        try:
            va = VarAstro(self.sky, max_dist=self.max_dist, logger=self.logger)
            va_data = va.period()
            va_data["source"] = "VAS"
        except Exception as e:
            va_data = pd.DataFrame(columns=["source", "name", "P", "PEmin", "PEmax"])
            self.logger.warning(e)

        data = pd.concat([nasa_data, eu_data, va_data], ignore_index=True)
        return data[["name", "source", "P", "PEmin", "PEmax"]]
