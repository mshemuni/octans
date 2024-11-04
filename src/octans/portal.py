from logging import getLogger, Logger
from typing import Self, Optional, Literal, Iterator, List, Any

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
from .utils import NAngleType, degree_checker, time_checker, Minima

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


def check_exists_by_xpath(driver, xpath, timeout=3):
    """
    Check if an element does exist in a webpage
    """
    try:
        _ = WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, xpath)))
        return True
    except TimeoutException:
        return False


def line_formatter(line) -> List[Any]:
    """
    Format the lines obtained form ASAS's web site
    """
    the_line = list(map(float, line[:11])) + [line[11], int(line[12])]
    the_line[0] += 2450000
    return the_line


def magnitude_to_flux(magnitude: float, ref_magnitude: float = 0.0, ref_flux: float = 1.0):
    """
    Convert Magnitude to Flux
    """
    flux_density = np.power(
        10,
        (magnitude - ref_magnitude) / - 2.5
    ) * ref_flux / (units.electron * units.second)
    return flux_density.decompose()


class Portal(ModelPortal):
    def __init__(self, sky: Sky, logger: Optional[Logger] = None) -> None:
        if logger is None:
            self.logger = getLogger(__name__)
        else:
            self.logger = logger

        self.__sky = sky

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(sky: {self.sky})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.sky})"

    @classmethod
    def from_name(cls, name: str, logger: Optional[Logger] = None) -> Self:
        return cls(Sky(name), logger=logger)

    @classmethod
    def from_coordinates(cls, ra: NAngleType, dec: NAngleType, radius: NAngleType = 2 * units.arcmin,
                         logger: Optional[Logger] = None) -> Self:
        return cls(Sky.from_coordinates(ra, dec, radius), logger=logger)

    @property
    def sky(self) -> Sky:
        return self.__sky

    @sky.setter
    def sky(self, sky: Sky) -> None:
        self.logger.error("This attribute is immutable and cannot be changed.")
        raise AttributeError("This attribute is immutable and cannot be changed.")

    def kt(self, mission: Optional[Literal['kepler', 'tess']] = None) -> Iterator[XLightCurve]:
        search_result = lk.search_lightcurve(self.sky.name, mission=mission)

        if len(search_result) == 0:
            self.logger.error(f"No data found for target {self.sky.name} at {mission}.")
            raise NoLightCurveError(f"No data found for target {self.sky.name} at {mission}.")

        lc_collection = search_result.download_all()
        for lc in lc_collection:
            yield XLightCurve(time=lc.time.jd, flux=lc.flux.value, flux_err=lc.flux_err.value)

    def kepler(self) -> Iterator[XLightCurve]:
        return self.kt("kepler")

    def tess(self) -> Iterator[XLightCurve]:
        return self.kt("tess")

    def asas(self) -> Iterator[XLightCurve]:
        asas = ASAS.from_sky(self.sky)
        return asas.get()

    def var_astro(self) -> Minima:
        var_astro = VarAstro.from_sky(self.sky)
        return var_astro.get()

    def oc_gateway(self) -> Minima:
        return self.var_astro()

    def etd(self) -> Minima:
        etd = ETD.from_sky(self.sky)
        return etd.get()


class ASAS(ModelASAS):
    def __init__(self, asas_id: str, logger: Optional[Logger] = None) -> None:
        if logger is None:
            self.logger = getLogger(__name__)
        else:
            self.logger = logger

        self.asas_id = asas_id

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(asas_id: {self.asas_id})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.asas_id}')"

    @classmethod
    def from_sky(cls, sky: Sky, max_dist: NAngleType = 20 * units.arcsec, time_out: NAngleType = 3 * units.s,
                 logger: Optional[Logger] = None) -> Self:
        if logger is None:
            logger = getLogger(__name__)

        the_max_dist = degree_checker(max_dist)
        the_time_out = time_checker(time_out)

        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        driver.get("https://www.astrouw.edu.pl/asas/?page=acvs")

        if not check_exists_by_xpath(driver, ASAS_XPATHS["input_coordinate"], timeout=the_time_out.to(units.s).value):
            logger.error("Page Not Found")
            raise PageNotFoundError("Page Not Found")

        # Write the coordinates
        coordinates_box = driver.find_element(By.XPATH, ASAS_XPATHS["input_coordinate"])
        coordinates_box.clear()

        coordinates_box.send_keys(f"{sky.skycoord.ra.to_string('hour', sep=':', precision=0)} "
                                  f"{sky.skycoord.dec.to_string('deg', sep=':', precision=0)}")

        # Clear the radius box and write the actual value
        radius_box = driver.find_element(By.XPATH, ASAS_XPATHS["input_radius"])
        radius_box.clear()
        radius_box.send_keys(f"{int(the_max_dist.to(units.arcsec).value)}")

        # Make sure the radius is in arcseconds
        arcsec_radio = driver.find_element(By.XPATH, ASAS_XPATHS["input_arcsec_radio"])
        arcsec_radio.click()

        # Show all results
        all_radio = driver.find_element(By.XPATH, ASAS_XPATHS["input_show"])
        all_radio.click()

        # Get the results
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

        for tr_soup in trs_soup:
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

        return cls(best_match.text.strip(), logger=logger)

    def data(self) -> str:
        respond = requests.get(f"https://www.astrouw.edu.pl/cgi-asas/asas_cgi_get_data?{self.asas_id},asas3")
        if respond.status_code != 200:
            self.logger.error("There's no such ASAS object")
            raise NoASASObjectError("There's no such ASAS object")

        return str(respond.text)

    def get(self) -> Iterator[XLightCurve]:
        content = self.data()
        the_lc = pd.DataFrame(
            columns=["HJD", "MAG_4", "MAG_0", "MAG_1", "MAG_2", "MAG_3", "MER_4", "MER_0", "MER_1", "MER_2", "MER_3",
                     "GRADE", "FRAME"]
        )

        chunks = list(map(lambda x: "#ndata=" + x, content.split("#ndata=")[1:]))
        for chunk in chunks:
            header = None
            data = []
            for line in chunk.split("\n"):
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
        for extension in range(5):
            yield XLightCurve(
                Time(the_lc["HJD"], format="jd", scale="utc"),
                magnitude_to_flux(the_lc[f"MAG_{extension}"].to_numpy()),
                magnitude_to_flux(the_lc[f"MER_{extension}"].to_numpy())
            )


class VarAstro(ModelVarAstro):
    def __init__(self, var_astro_id: int, logger: Optional[Logger] = None) -> None:
        if logger is None:
            self.logger = getLogger(__name__)
        else:
            self.logger = logger

        self.var_astro_id = var_astro_id

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(var_astro_id: {self.var_astro_id})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.var_astro_id}')"

    @classmethod
    def from_sky(cls, sky: Sky, max_dist: NAngleType = 20 * units.arcmin, time_out: NAngleType = 3 * units.s,
                 logger: Optional[Logger] = None) -> Self:
        if logger is None:
            logger = getLogger(__name__)

        the_max_dist = degree_checker(max_dist)
        the_time_out = time_checker(time_out)

        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
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

        if data[1].text.strip() == "0":
            logger.error("There's no entry for the given object in VarAstro")
            raise NoLightCurveError("There's no entry for the given object in VarAstro")

        var_astro_id = int(data[0].text.strip())

        driver.close()
        return cls(var_astro_id, logger=logger)

    def get(self) -> Minima:
        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
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
        for data in csv_data["dataPrimary"]:
            time.append(float(data["jd"]))
            time_error.append(np.nan)

        return Minima(Time(time, format="jd", scale="utc"), TimeDelta(time_error, format="jd"))


class ETD(ModelVarAstro):
    def __init__(self, etd_id: int, logger: Optional[Logger] = None) -> None:
        if logger is None:
            self.logger = getLogger(__name__)
        else:
            self.logger = logger

        self.etd_id = etd_id

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(etd_id: {self.etd_id})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.etd_id}')"

    @classmethod
    def from_sky(cls, sky: Sky, max_dist: NAngleType = 20 * units.arcmin, time_out: NAngleType = 3 * units.s,
                 logger: Optional[Logger] = None) -> Self:
        if logger is None:
            logger = getLogger(__name__)

        the_max_dist = degree_checker(max_dist)
        the_time_out = time_checker(time_out)

        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
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
        return cls(var_astro_id, logger=logger)

    def get(self) -> Minima:
        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
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
        for data in csv_data["data"]:
            time.append(float(data["jd"]))
            time_error.append(np.nan)

        return Minima(Time(time, format="jd", scale="utc"), TimeDelta(time_error, format="jd"))


class Utilities:
    def __init__(self, sky: Sky, logger: Optional[Logger] = None) -> None:
        if logger is None:
            self.logger = getLogger(__name__)
        else:
            self.logger = logger

        self.sky = sky

    def period(self):
        pass
