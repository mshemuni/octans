from unittest import TestCase

from octans import Sky, ExoplanetEU
from octans.errors import NoExoplanetEUObjectError


class TestExoplanetEU(TestCase):
    def test_create(self):
        wasp_3_b = Sky("wasp-3 b")
        _ = ExoplanetEU(wasp_3_b)

    def test_resolve(self):
        wasp_3_b = Sky("wasp-3 b")
        exoplanet = ExoplanetEU(wasp_3_b)
        for i in ['name', 'orbital_period', 'orbital_period_error_min', 'orbital_period_error_max']:
            self.assertIn(i, exoplanet.resolve().columns)

    def test_period(self):
        wasp_3_b = Sky("wasp-3 b")
        exoplanet = ExoplanetEU(wasp_3_b)
        for i in ['name', 'P', 'PEmin', 'PEmax']:
            self.assertIn(i, exoplanet.period().columns)

    def test_create_does_not_Exist(self):
        xy_leo = Sky("xy leo")
        exoplanet = ExoplanetEU(xy_leo)

        with self.assertRaises(NoExoplanetEUObjectError):
            _ = exoplanet.resolve()
