from unittest import TestCase

from octans import Sky, Period
from octans.errors import NoExoplanetEUObjectError, NoNASAExoplanetArchiveObjectError


class TestPeriod(TestCase):
    def test_create(self):
        wasp_3_b = Sky("wasp-3 b")
        _ = Period(wasp_3_b)

    def test_period_all(self):
        wasp_3_b = Sky("wasp-3 b")
        exoplanet = Period(wasp_3_b)
        for i in ["source", "P", "PEmin", "PEmax"]:
            self.assertIn(i, exoplanet.all().columns)

    def test_period_all_does_not_exist(self):
        xy_leo = Sky("xy leo")
        exoplanet = Period(xy_leo)
        with self.assertRaises((NoExoplanetEUObjectError, NoNASAExoplanetArchiveObjectError)):
            _ = exoplanet.all()

    def test_period_exoplanet_eu(self):
        wasp_3_b = Sky("wasp-3 b")
        exoplanet = Period(wasp_3_b)
        for i in ["P", "PEmin", "PEmax"]:
            self.assertIn(i, exoplanet.exoplanet_eu().columns)

    def test_period_exoplanet_eu_does_not_exist(self):
        xy_leo = Sky("xy leo")
        exoplanet = Period(xy_leo)
        with self.assertRaises(NoExoplanetEUObjectError):
            _ = exoplanet.exoplanet_eu()

    def test_period_nasa_exoplanet_archive(self):
        wasp_3_b = Sky("wasp-3 b")
        exoplanet = Period(wasp_3_b)
        for i in ["P", "PEmin", "PEmax"]:
            self.assertIn(i, exoplanet.nasa_exoplanet_archive().columns)

    def test_period_nasa_exoplanet_archive_does_not_exist(self):
        xy_leo = Sky("xy leo")
        exoplanet = Period(xy_leo)
        with self.assertRaises(NoNASAExoplanetArchiveObjectError):
            _ = exoplanet.nasa_exoplanet_archive()
