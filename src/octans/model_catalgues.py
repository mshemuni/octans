from abc import ABC, abstractmethod

import pandas as pd


class ExoplanetEUModel(ABC):
    @abstractmethod
    def resolve(self) -> pd.DataFrame:
        """Resolves the given sky coordinate to the database"""

    @abstractmethod
    def period(self) -> pd.DataFrame:
        """Returns the period of the exoplanet"""


class NASAExoplanetArchiveModel(ABC):
    @abstractmethod
    def resolve(self) -> pd.DataFrame:
        """Resolves the given sky coordinate to the database"""

    @abstractmethod
    def period(self) -> pd.DataFrame:
        """Returns the period of the exoplanet"""


class PeriodModel:
    @abstractmethod
    def nasa_exoplanet_archive(self) -> pd.DataFrame:
        """Returns the period of the exoplanet from Nasa Exoplanet Archive"""

    @abstractmethod
    def exoplanet_eu(self) -> pd.DataFrame:
        """Returns the period of the exoplanet from Exoplanet EU"""

    def all(self) -> pd.DataFrame:
        """Returns the period of the exoplanet from all available databases"""
