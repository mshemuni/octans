class ObjectNotFoundError(Exception):
    """Raised when an Object not found"""


class NoExoplanetEUObjectError(ObjectNotFoundError):
    """Raised when an ExoplanetEU object not found"""


class NoNASAExoplanetArchiveObjectError(ObjectNotFoundError):
    """Raised when an ExoplanetEU object not found"""


class NoASASObjectError(ObjectNotFoundError):
    """Raised when an ASAS object not found"""


class NoVarAstroObjectError(ObjectNotFoundError):
    """Raised when an ASAS object not found"""


class NoDataFoundError(Exception):
    """Raised when no data was found"""


class PageNotFoundError(Exception):
    """Raised when page not found"""


class NoLightCurveError(NoDataFoundError):
    """Raised when a light curve not found"""
