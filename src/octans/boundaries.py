from logging import Logger
from typing import Optional, Iterator
from typing_extensions import Self

from astropy.time import Time

from octans.utils import logger_checker


class Boundaries:
    def __init__(self, lower: Time, upper: Time, logger: Optional[Logger] = None):
        self.logger = logger_checker(logger, __name__)

        self.lower = lower
        self.upper = upper

        self.__verify()

    def __verify(self):
        if self.lower.isscalar:
            if not self.upper.isscalar:
                self.logger.error("Both upper and lower must be either scalar or list")
                raise ValueError("Both upper and lower must be either scalar or list")
        else:
            if self.lower.isscalar:
                self.logger.error("Both upper and lower must be either scalar or list")
                raise ValueError("Both upper and lower must be either scalar or list")

            if len(self.upper) != len(self.lower):
                self.logger.error("Both upper and lower must be of same length")
                raise ValueError("Both upper and lower must be of same length")

    def __str__(self) -> str:
        if self.upper.isscalar:
            return f"{self.lower} {self.upper}"

        return f"lower:{self.lower}, upper:{self.upper}"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        if self.lower.isscalar:
            return 1

        return len(self.lower)

    def __iter__(self) -> Iterator[Self]:
        if self.lower.isscalar:
            yield self.__class__(self.lower, self.upper)
        else:
            for time, error in zip(self.lower, self.upper):
                yield self.__class__(time, error)

    def append(self, other: Self) -> Self:
        """
        Appends two boundary objects

        Parameters
        ----------
        other : Self
            Another instance of boundary

        Returns
        -------
        Self
            A new instance of Boundary
        """
        return self.__class__(
            Time(self.lower.jd.tolist() + other.lower.jd.tolist(), format='jd'),
            Time(self.upper.jd.tolist() + other.upper.jd.tolist(), format='jd'),
            logger=self.logger
        )
