from logging import Logger
from typing import Optional, Literal, Iterator, List, Any
from typing_extensions import Self

import numpy as np
import pandas as pd
from astropy import units
import lightkurve as lk
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException

from bs4 import BeautifulSoup

import requests

from . import XLightCurve
from .errors import NoLightCurveError, PageNotFoundError, NoASASObjectError, NoVarAstroObjectError, \
    NoDataFoundError
from .model_portal import ModelPortal, ModelASAS, ModelVarAstro
from .sky import Sky
from .utils import NAngleType, degree_checker, time_checker, logger_checker, tqdmify
from .minima import Minima

ASAS_XPATHS = {
    "input_coordinate": "/html/body/div[2]/form/table/tbody/tr[2]/td[1]/input",
    "input_radius": "/html/body/div[2]/form/table/tbody/tr[2]/td[2]/input",
    "input_arcsec_radio": "/html/body/div[2]/form/table/tbody/tr[2]/td[3]/input[1]",
    "input_show": "/html/body/div[2]/form/table/tbody/tr[2]/td[5]/input[1]",
    "input_results": "/html/body/div[2]/form/table/tbody/tr[1]/th[5]/input",
    "result_table": "/html/body/div[2]/table"
}

VarAstro_XPATHS = {
    "result_table_first_row": "/html/body/div[2]/div[2]/div[3]/div[2]/div[2]/div[2]/div/div/div/div/table/tbody/tr[1]"
}


def check_exists_by_xpath(driver: webdriver, xpath: str, timeout: int = 3):
    """
    Checks if an element specified by an XPath exists within the given timeout.

    Parameters
    ----------
    driver: webdriver
        The Selenium WebDriver instance to use for checking the element.
    xpath: str
        The XPath of the element to be checked.
    timeout: int, default = 3
        The maximum time to wait for the element to appear, in seconds.

    Returns
    -------
    bool
        True if the element is found within the timeout period, False otherwise.
    """
    try:
        _ = WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, xpath)))
        return True
    except TimeoutException:
        return False


def line_formatter(line: List[str]) -> List[Any]:
    """

    This function formats a line of string values by converting the first 11 elements
    to float, the 12th element to a string, and the 13th element to an integer.
    It also adds 2450000 to the first element in the returned list.

    Parameters
    ----------
    line: List[str]
        A list of strings representing the input data.

    Returns
    -------
    List[Any]
        A list with the first 11 elements as floats (except the first which has an additional 2450000 added to it),
        the 12th element as a string, and the 13th element as an integer.
    """
    the_line = list(map(float, line[:11])) + [line[11], int(line[12])]
    the_line[0] += 2450000
    return the_line


def magnitude_to_flux(magnitude: float, ref_magnitude: float = 0.0, ref_flux: float = 1.0) -> units.Quantity:
    """
    Calculate the flux density given a magnitude.

    Parameters
    ----------
    magnitude: float
        The magnitude for which the flux density is to be calculated.
    ref_magnitude: float, default = 0.0
        The reference magnitude.
    ref_flux: float, default = 1.0
        The reference flux corresponding to the reference magnitude.

    Returns
    -------
    units.Quantity
        The calculated flux density, decomposed into base units.
    """
    flux_density = np.power(
        10,
        (magnitude - ref_magnitude) / - 2.5
    ) * ref_flux / (units.electron * units.second)
    return flux_density.decompose()


class Portal(ModelPortal):
    def __init__(self, sky: Sky, verbose: bool = False, logger: Optional[Logger] = None) -> None:
        self.logger = logger_checker(logger, __name__)
        self.verbose = verbose

        self.__sky = sky

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(sky: {self.sky})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.sky})"

    @classmethod
    def from_name(cls, name: str, verbose: bool = False, logger: Optional[Logger] = None) -> Self:
        """
        Creates an instance of the class using the provided name.

        Parameters
        ----------
        name: str
            The name used to create the Sky instance.
        verbose: bool, default = False
            Verbosity
        logger: Logger, default = None
            An optional logger for the instance.

        Returns
        -------
        Self
            An instance of the class.
        """
        return cls(Sky(name), verbose=verbose, logger=logger)

    @classmethod
    def from_coordinates(cls, ra: NAngleType, dec: NAngleType, radius: NAngleType = 2 * units.arcmin,
                         verbose: bool = False, logger: Optional[Logger] = None) -> Self:
        """
        Constructs an instance of the class using right ascension (RA), declination (DEC), and radius.

        Parameters
        ----------
        ra: NAngleType
            Right ascension in angle type.
        dec: NAngleType
            Declination in angle type.
        radius: NAngleType, default = 2 * units.arcmin
            Radius in angle type. Default is 2 arcminutes.
        verbose: bool, default = False
            Verbosity
        logger: logger, default = None
            Logger instance.

        Returns
        -------
        Self
            An instance of the class with the specified coordinates.
        """
        return cls(Sky.from_coordinates(ra, dec, radius), logger=logger)

    @property
    def sky(self) -> Sky:
        return self.__sky

    @sky.setter
    def sky(self, sky: Sky) -> None:
        self.logger.error("This attribute is immutable and cannot be changed.")
        raise AttributeError("This attribute is immutable and cannot be changed.")

    def kt(self, mission: Optional[Literal['kepler', 'tess']] = None) -> Iterator[XLightCurve]:
        """
        kt method searches for light curve data for a given celestial target using the Lightkurve package.

        Parameters
        ----------
        mission: Optional[Literal['kepler', 'tess']], default = None
            Specifies the mission data to search for. Can be 'kepler' or 'tess'.

        Yields
        -------
        XLightCurve
            An iterator over light curve data objects containing time, flux, and flux error.

        Raises
        ------
        NoLightCurveError
            If no data is found for the specified target and mission.
        """
        search_result = lk.search_lightcurve(self.sky.name, mission=mission)

        if len(search_result) == 0:
            self.logger.error(f"No data found for target {self.sky.name} at {mission}.")
            raise NoLightCurveError(f"No data found for target {self.sky.name} at {mission}.")

        lc_collection = search_result.download_all()
        for lc in tqdmify(lc_collection, verbose=self.verbose):
            yield XLightCurve(time=lc.time.jd, flux=lc.flux.value, flux_err=lc.flux_err.value)

    def kepler(self) -> Iterator[XLightCurve]:
        """
        Returns an iterator over Kepler light curves.

        The function `kepler` provides an iterator that yields `XLightCurve` objects from Kepler data.

        Returns
        -------
        Iterator[XLightCurve]
            An iterator over `XLightCurve` objects from Kepler data.
        """
        return self.kt("kepler")

    def tess(self) -> Iterator[XLightCurve]:
        """
        Yields XLightCurve objects from the 'tess' Kepler task pipeline.

        The method utilizes the `kt` method to retrieve and process data specific to the 'tess' task.

        Returns
        -------
        Iterator[XLightCurve]
            An iterator of XLightCurve objects generated from the 'tess' task.
        """
        return self.kt("tess")

    def asas(self) -> Iterator[XLightCurve]:
        """
        This method retrieves light curves from the ASAS (All Sky Automated Survey) database
        corresponding to the current sky object and returns an iterator over the extracted light curves.

        Returns
        -------
        Iterator[XLightCurve]
            An iterator over the extracted light curves.
        """
        asas = ASAS.from_sky(self.sky)
        return asas.get()

    def var_astro(self) -> Minima:
        """
        Converts the given sky data into a VarAstro object and returns the minimum value.

        Converts sky data stored in the instance's `sky` attribute into a VarAstro object
        and calculates its minimum value.

        Returns
        -------
        Minima
            The minimum value calculated from the VarAstro object.
        """
        var_astro = VarAstro.from_sky(self.sky, verbose=self.verbose, logger=self.logger)
        return var_astro.get()

    def oc_gateway(self) -> Minima:
        """
        Returns the result of the O-C Gateway method.

        The oc_gateway method invokes the var_astro method and returns its output. The return value is an instance of the Minima class.

        Returns
        -------
        Minima
            The result of the var_astro method.
        """
        return self.var_astro()

    def etd(self) -> Minima:
        """
        Converts the ETD from the sky component
        and retrieves the minimum value from VarAstro's ETD database.

        Returns
        -------
        Minima
            The minimum ETD value.
        """
        etd = ETD.from_sky(self.sky)
        return etd.get()


class ASAS(ModelASAS):
    def __init__(self, asas_id: str, verbose: bool = False, logger: Optional[Logger] = None) -> None:
        self.logger = logger_checker(logger, __name__)
        self.verbose = verbose

        self.asas_id = asas_id

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(asas_id: {self.asas_id})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.asas_id}')"

    @classmethod
    def from_sky(cls, sky: Sky, max_dist: NAngleType = 20 * units.arcsec, time_out: NAngleType = 3 * units.s,
                 verbose: bool = False, logger: Optional[Logger] = None) -> Self:
        """
        classmethod to create an instance of the class from sky coordinates.

        Parameters
        ----------
        sky: Sky
            Sky object containing sky coordinates.
        max_dist: NAngleType, default = 20 * units.arcsec
            Maximum distance for the ASAS search, default is 20 arcsecs.
        time_out: NAngleType, default = 3 * units.s
            Timeout for web interactions, default is 3 seconds.
        verbose: bool, default = False
            Verbosity
        logger: Logger, default = None
            Optional logger object for logging messages.

        Returns
        -------
        Self
            Instance of the class if a match is found.

        Raises
        ------
        PageNotFoundError
            If the ASAS page is not found.
        NoASASObjectError
            If no ASAS object matches the search criteria.
        """
        logger = logger_checker(logger, __name__)

        the_max_dist = degree_checker(max_dist)
        the_time_out = time_checker(time_out)

        options = Options()
        options.add_argument("--headless")
        # driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        driver = webdriver.Chrome(options=options)
        driver.get("https://www.astrouw.edu.pl/asas/?page=acvs")

        if not check_exists_by_xpath(driver, ASAS_XPATHS["input_coordinate"], timeout=the_time_out.to(units.s).value):
            logger.error("Page Not Found")
            raise PageNotFoundError("Page Not Found")

        coordinates_box = driver.find_element(By.XPATH, ASAS_XPATHS["input_coordinate"])
        coordinates_box.clear()

        coordinates_box.send_keys(f"{sky.skycoord.ra.to_string('hour', sep=':', precision=0)} "
                                  f"{sky.skycoord.dec.to_string('deg', sep=':', precision=0)}")

        radius_box = driver.find_element(By.XPATH, ASAS_XPATHS["input_radius"])
        radius_box.clear()
        radius_box.send_keys(f"{int(the_max_dist.to(units.arcsec).value)}")

        arcsec_radio = driver.find_element(By.XPATH, ASAS_XPATHS["input_arcsec_radio"])
        arcsec_radio.click()

        all_radio = driver.find_element(By.XPATH, ASAS_XPATHS["input_show"])
        all_radio.click()

        result_button = driver.find_element(By.XPATH, ASAS_XPATHS["input_results"])
        result_button.click()

        if not check_exists_by_xpath(driver, ASAS_XPATHS["result_table"], timeout=the_time_out.to(units.s).value):
            logger.error("There's no such ASAS object")
            raise NoASASObjectError("There's no such ASAS object")

        page_html = driver.page_source
        page_soup = BeautifulSoup(page_html, 'html.parser')

        table_soup = page_soup.select_one("table")
        if len(table_soup) == 0:
            logger.error("There's no such ASAS object")
            raise NoASASObjectError("There's no such ASAS object")

        trs_soup = table_soup.find_all("tr", {"align": "center"})

        if len(trs_soup) == 0:
            logger.error("There's no such ASAS object")
            raise NoASASObjectError("There's no such ASAS object")

        min_distance = 180 * units.deg
        best_match = None

        for tr_soup in tqdmify(trs_soup):
            tds_soup = tr_soup.find_all("td")
            if len(tds_soup) != 10:
                logger.warning("The table is not found")
                continue

            each_sky_coord = SkyCoord(
                ra=tds_soup[1].text.strip(),
                dec=tds_soup[2].text.strip(),
                unit=("hourangle", "degree")
            )
            each_separation = each_sky_coord.separation(sky.skycoord)

            if each_separation < the_max_dist:
                if each_separation < min_distance:
                    min_distance = each_separation
                    best_match = tds_soup[0]

        if best_match is None:
            logger.error("There's no such ASAS object")
            raise NoASASObjectError("There's no such ASAS object")

        return cls(best_match.text.strip(), verbose=verbose, logger=logger)

    def data(self) -> str:
        """

        Fetches data from the ASAS service based on the ASAS ID.

        Returns
        -------
        str
            The text response from the ASAS service.

        Raises
        ------
        NoASASObjectError
            If the ASAS object does not exist or the status code is not 200.
        """
        respond = requests.get(f"https://www.astrouw.edu.pl/cgi-asas/asas_cgi_get_data?{self.asas_id},asas3")
        if respond.status_code != 200:
            self.logger.error("There's no such ASAS object")
            raise NoASASObjectError("There's no such ASAS object")

        return str(respond.text)

    def get(self) -> Iterator[XLightCurve]:
        """
        Generator method to yield XLightCurve objects from data chunks.

        Returns
        -------
        Iterator[XLightCurve]
            An iterator of XLightCurve objects.
        """
        content = self.data()
        the_lc = pd.DataFrame(
            columns=["HJD", "MAG_4", "MAG_0", "MAG_1", "MAG_2", "MAG_3", "MER_4", "MER_0", "MER_1", "MER_2", "MER_3",
                     "GRADE", "FRAME"]
        )

        chunks = list(map(lambda x: "#ndata=" + x, content.split("#ndata=")[1:]))
        for chunk in tqdmify(chunks, verbose=self.verbose):
            header = None
            data = []
            for line in tqdmify(chunk.split("\n"), verbose=self.verbose):
                if line.startswith("#"):
                    if "HJD" in line:
                        header = line.replace("#", "").strip().split()
                else:
                    if line:
                        data.append(line_formatter(line.split()))

            the_lc = pd.concat([
                the_lc,
                pd.DataFrame(
                    data=data,
                    columns=header
                )
            ])
        the_lc = the_lc.sort_values(by=["HJD"])
        for extension in tqdmify(range(5), verbose=self.verbose):
            yield XLightCurve(
                Time(the_lc["HJD"], format="jd", scale="utc"),
                magnitude_to_flux(the_lc[f"MAG_{extension}"].to_numpy()),
                magnitude_to_flux(the_lc[f"MER_{extension}"].to_numpy())
            )


class VarAstro(ModelVarAstro):
    def __init__(self, var_astro_id: int, verbose: bool = False, logger: Optional[Logger] = None) -> None:
        self.logger = logger_checker(logger, __name__)
        self.verbose = verbose

        self.var_astro_id = var_astro_id

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(var_astro_id: {self.var_astro_id})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.var_astro_id}')"

    @classmethod
    def from_sky(cls, sky: Sky, max_dist: NAngleType = 20 * units.arcmin, time_out: NAngleType = 3 * units.s,
                 verbose: bool = False, logger: Optional[Logger] = None) -> Self:
        """
            Creates an instance of the class using data fetched from the VarAstro website.

            Parameters
            ----------
            sky: Boundaries, Sky
                An instance of the Sky class containing sky coordinates.
            max_dist: NAngleType, default = 20 * units.arcmin
                Maximum distance around the coordinates to search, defaults to 20 arcminutes.
            time_out: NAngleType, default = 3 * units.s
                The timeout period for waiting for the webpage to load, defaults to 3 seconds.
            verbose: bool, default = False
                Verbosity
            logger: Logger, default = None
                Optional logger for logging, defaults to None.

            Returns
            -------
            Self
                An instance of the class initialized with the fetched VarAstro object ID.

            Raises
            ------
            NoVarAstroObjectError
                If no VarAstro objects are found within the specified coordinates.
            NoLightCurveError
                If there are no entries for the object in VarAstro.
        """
        logger = logger_checker(logger, __name__)

        the_max_dist = degree_checker(max_dist)
        the_time_out = time_checker(time_out)
        options = Options()
        options.add_argument("--headless")
        # driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        driver = webdriver.Chrome(options=options)
        driver.get(
            f"https://var.astro.cz/en/Stars?pageId=1&pageSize=20&ra="
            f"{sky.skycoord.ra.degree}&dec={sky.skycoord.dec.degree}&radiusArcmin="
            f"{int(the_max_dist.to(units.arcmin).value)}")

        WebDriverWait(driver, the_time_out.to(units.s).value).until(
            EC.presence_of_element_located((By.XPATH, VarAstro_XPATHS["result_table_first_row"]))
        )

        rows = driver.find_elements(By.XPATH, VarAstro_XPATHS["result_table_first_row"])

        data = rows[0].find_elements(By.TAG_NAME, "td")
        if data[0].text.strip() == "No data available in table":
            logger.error("There's no such VarAstro object")
            raise NoVarAstroObjectError("There's no such VarAstro object")

        if data[2].text.strip() == "0":
            logger.error("There's no entry for the given object in VarAstro")
            raise NoLightCurveError("There's no entry for the given object in VarAstro")

        var_astro_id = int(data[0].text.strip())

        driver.close()
        return cls(var_astro_id, verbose=verbose, logger=logger)

    def get(self) -> Minima:
        """
        Retrieves minima data from var.astro.cz for the specified astronomical object.

        This method uses Selenium to launch a headless Chrome WebDriver instance,
        navigate to a specific URL based on the object's var_astro_id, and
        extracts data using JavaScript execution. The extracted data is then
        returned as a Minima object containing time and time_error values.


        Returns
        -------
        Minima
            An object containing the retrieved time and time_error data.

        Raises
        -------
        NoDataFoundError
            If no data is found for the specified target.
        NoVarAstroObjectError
            If the VarAstro object is invalid or does not exist.
        """
        options = Options()
        options.add_argument("--headless")
        # driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        driver = webdriver.Chrome(options=options)
        driver.get(
            f"https://var.astro.cz/en/Stars/{self.var_astro_id}"
        )

        csv_data = driver.execute_script(f"""
                return varastro.shared.fetch("/api/charts/ocdiagram/{self.var_astro_id}?v=0").then(function(n) {{
                    if (n.status === 200) return n.json();
                    return n;
                }}).then(function(t) {{
                    return t;
                }})
                """)

        if csv_data is None:
            raise NoDataFoundError("No data found for target.")

        if "dataPrimary" not in csv_data.keys():
            raise NoVarAstroObjectError("There's no such VarAstro object")

        time = []
        time_error = []
        for data in tqdmify(csv_data["dataPrimary"], verbose=self.verbose):
            time.append(float(data["jd"]))
            time_error.append(np.nan)

        return Minima(Time(time, format="jd", scale="utc"), TimeDelta(time_error, format="jd"))


class ETD(ModelVarAstro):
    def __init__(self, etd_id: int, verbose: bool = False, logger: Optional[Logger] = None) -> None:
        self.logger = logger_checker(logger, __name__)
        self.verbose = verbose

        self.etd_id = etd_id

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(etd_id: {self.etd_id})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.etd_id}')"

    @classmethod
    def from_sky(cls, sky: Sky, max_dist: NAngleType = 20 * units.arcmin, time_out: NAngleType = 3 * units.s,
                 verbose: bool = False, logger: Optional[Logger] = None) -> Self:
        """
        Create an instance of the class by fetching data from the VarAstro website.

        This method initializes a headless browser instance to fetch and process data from the VarAstro website using coordinates from the provided Sky object. It waits for the presence of a specific element in the resulting table and retrieves data if available.

        Parameters
        ----------
        sky: Sky
            An instance of the Sky class with RA and Dec coordinates.
        max_dist: NAngleType, default = 20 * units.arcmin
            Maximum distance for the search area, in arcminutes.
        time_out: NAngleType, default = 3 * units.s
            Timeout duration for waiting for the web element, in seconds.
        verbose: bool, default = False
            Verbosity
        logger: Logger, default = None
            Logger instance for logging messages.

        Returns
        -------
        Self
            An instance of the class containing the fetched VarAstro ID.


        Raises
        -------
        NoVarAstroObjectError
            If no data is available for the provided coordinates.
        NoLightCurveError
            If no light curve entry is found for the provided object.
        """
        logger = logger_checker(logger, __name__)

        the_max_dist = degree_checker(max_dist)
        the_time_out = time_checker(time_out)

        options = Options()
        options.add_argument("--headless")
        # driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        driver = webdriver.Chrome(options=options)
        driver.get(
            f"https://var.astro.cz/en/Exoplanets?pageId=1&pageSize=20&ra="
            f"{sky.skycoord.ra.degree}&dec={sky.skycoord.dec.degree}&radiusArcmin="
            f"{int(the_max_dist.to(units.arcmin).value)}"
        )

        WebDriverWait(driver, the_time_out.to(units.s).value).until(
            EC.presence_of_element_located((By.XPATH, VarAstro_XPATHS["result_table_first_row"]))
        )

        rows = driver.find_elements(By.XPATH, VarAstro_XPATHS["result_table_first_row"])

        data = rows[0].find_elements(By.TAG_NAME, "td")
        if data[0].text.strip() == "No data available in table":
            logger.error("There's no such ETD object")
            raise NoVarAstroObjectError("There's no such ETD object")

        if data[1].text.strip() == "0":
            logger.error("There's no entry for the given object in ETD")
            raise NoLightCurveError("There's no entry for the given object in ETD")

        var_astro_id = int(data[0].text.strip())

        driver.close()
        return cls(var_astro_id, verbose=verbose, logger=logger)

    def get(self) -> Minima:
        """
        Fetches data from the var.astro.cz exoplanet database and returns it in a Minima object.

        Returns
        -------
        Minima
            A Minima object containing transposition midtimes and their associated errors.

        Raises
        ------
        NoDataFoundError
            If no data is found for the given target.
        NoVarAstroObjectError
            If there is no ETD object corresponding to self.etd_id.
        """
        options = Options()
        options.add_argument("--headless")
        # driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        driver = webdriver.Chrome(options=options)
        driver.get(
            f"https://var.astro.cz/en/Exoplanets/{self.etd_id}"
        )

        csv_data = driver.execute_script(f"""
                return varastro.shared.fetch("/api/charts/exottvdiagram/{self.etd_id}?v=0").then(function(n) {{
                    if (n.status === 200) return n.json();
                    return n;
                }}).then(function(t) {{
                    return t;
                }})
                """)

        if csv_data is None:
            raise NoDataFoundError("No data found for target.")

        if "data" not in csv_data.keys():
            raise NoVarAstroObjectError("There's no such ETD object")

        time = []
        time_error = []
        for data in tqdmify(csv_data["data"], verbose=self.verbose):
            time.append(float(data["jd"]))
            time_error.append(np.nan)

        return Minima(Time(time, format="jd", scale="utc"), TimeDelta(time_error, format="jd"))
