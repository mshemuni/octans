from typing import List, Optional, Literal, Tuple

import numpy as np

from .errors import NoLightCurveError, PageNotFound, NoASASObject
from .lc import XLightCurve
from .sky import Sky
from .utils import NAngle, degree_checker, time_checker, NTime
from .rotse import FIELDS as ROTSE_FILEDS

from typing_extensions import Self

import lightkurve as lk

import pandas as pd

from astropy.coordinates import SkyCoord
from astropy import units
from astropy.time import Time

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

# The XPATHs of ASAS's web site.
XPATHS = {
    "input_coordinate": "/html/body/div[2]/form/table/tbody/tr[2]/td[1]/input",
    "input_radius": "/html/body/div[2]/form/table/tbody/tr[2]/td[2]/input",
    "input_arcsec_radio": "/html/body/div[2]/form/table/tbody/tr[2]/td[3]/input[1]",
    "input_show": "/html/body/div[2]/form/table/tbody/tr[2]/td[5]/input[1]",
    "input_results": "/html/body/div[2]/form/table/tbody/tr[1]/th[5]/input",
    "result_table": "/html/body/div[2]/table"
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


def line_formatter(line) -> List:
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


class Portal:
    def __init__(self, sky: Sky) -> None:
        self.__sky = sky

        self.__asas_url = "http://asassn-lb01.ifa.hawaii.edu:9006/lookup_sql/"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(sky: {self.sky})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.sky})"

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(Sky(name))

    @classmethod
    def from_coordinates(cls, ra: NAngle, dec: NAngle, radius: NAngle = 2 * units.arcmin) -> Self:
        return cls(Sky.from_coordinates(ra, dec, radius))

    @property
    def sky(self) -> Sky:
        return self.__sky

    @sky.setter
    def sky(self, sky: Sky) -> None:
        raise AttributeError("This attribute is immutable and cannot be changed.")

    def kkt(self, mission: Optional[Literal['kepler', 'k2', 'tess']] = None) -> List[XLightCurve]:
        search_result = lk.search_lightcurve(self.sky.name, mission=mission)

        if len(search_result) == 0:
            raise NoLightCurveError(f"No data found for target {self.sky.name}.")

        lc_collection = search_result.download_all()
        return [
            XLightCurve.from_lightkurve(lc)
            for lc in lc_collection
        ]

    def kepler(self) -> List[XLightCurve]:
        return self.kkt("kepler")

    def tess(self) -> List[XLightCurve]:
        return self.kkt("tess")

    def k2(self) -> List[XLightCurve]:
        return self.kkt("k2")

    def asas(self) -> XLightCurve:
        asas = ASAS.from_sky(self.sky)
        return asas.get()

    def rotseIII(self, radius=1.8 * units.deg) -> List:
        availables = []
        for term, data in ROTSE_FILEDS.items():
            sky_list = SkyCoord(ra=data[:, 0], dec=data[:, 1], unit=(units.deg, units.deg))
            match = self.sky.skycoord.match_to_catalog_sky(sky_list)

            if any(match[1] <= radius):
                availables.append([term, int(data[match[0]][-1])])

        return availables


class ASAS:
    def __init__(self, asas_id: str) -> None:
        self.asas_id = asas_id

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(asas_id: {self.asas_id})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.asas_id}')"

    @classmethod
    def from_sky(cls, sky: Sky, max_dist: NAngle = 20 * units.arcsec, time_out: NTime = 3 * units.s) -> Self:
        the_max_dist = degree_checker(max_dist)
        the_time_out = time_checker(time_out)

        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        driver.get("https://www.astrouw.edu.pl/asas/?page=acvs")

        if not check_exists_by_xpath(driver, XPATHS["input_coordinate"], timeout=the_time_out.to(units.s).value):
            raise PageNotFound("Page Not Found")

        # Write the coordinates
        coordinates_box = driver.find_element(By.XPATH, XPATHS["input_coordinate"])
        coordinates_box.clear()

        coordinates_box.send_keys(f"{sky.skycoord.ra.to_string('hour', sep=':', precision=0)} "
                                  f"{sky.skycoord.dec.to_string('deg', sep=':', precision=0)}")

        # Clear the radius box and write the actual value
        radius_box = driver.find_element(By.XPATH, XPATHS["input_radius"])
        radius_box.clear()
        radius_box.send_keys(f"{int(the_max_dist.to(units.arcsec).value)}")

        # Make sure the radius is in arcseconds
        arcsec_radio = driver.find_element(By.XPATH, XPATHS["input_arcsec_radio"])
        arcsec_radio.click()

        # Show all results
        all_radio = driver.find_element(By.XPATH, XPATHS["input_show"])
        all_radio.click()

        # Get the results
        result_button = driver.find_element(By.XPATH, XPATHS["input_results"])
        result_button.click()

        if not check_exists_by_xpath(driver, XPATHS["result_table"], timeout=the_time_out.to(units.s).value):
            raise NoASASObject("There's no such ASAS object")

        page_html = driver.page_source
        page_soup = BeautifulSoup(page_html, 'html.parser')

        table_soup = page_soup.select_one("table")
        if len(table_soup) == 0:
            raise NoASASObject("There's no such ASAS object")

        trs_soup = table_soup.find_all("tr", {"align": "center"})

        if len(trs_soup) == 0:
            raise NoASASObject("There's no such ASAS object")

        min_distance = 180 * units.deg
        best_match = None

        for tr_soup in trs_soup:
            tds_soup = tr_soup.find_all("td")
            if len(tds_soup) != 10:
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
            raise NoASASObject("There's no such ASAS object")

        return cls(best_match.text.strip())

    def data(self) -> str:
        respond = requests.get(f"https://www.astrouw.edu.pl/cgi-asas/asas_cgi_get_data?{self.asas_id},asas3")
        if respond.status_code != 200:
            raise NoASASObject("There's no such ASAS object")

        return respond.text

    def get(self) -> XLightCurve:
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

        lcx = XLightCurve(
            Time(the_lc["HJD"], format="jd", scale="utc"),
            magnitude_to_flux(the_lc["MAG_0"].to_numpy()),
            magnitude_to_flux(the_lc["MER_0"].to_numpy())
        )

        return lcx
