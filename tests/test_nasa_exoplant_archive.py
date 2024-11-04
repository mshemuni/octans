from unittest import TestCase

from octans import Sky, NASAExoplanetArchive
from octans.errors import NoNASAExoplanetArchiveObjectError


class TestNasaExoplanetArchive(TestCase):
    def test_create(self):
        wasp_3_b = Sky("wasp-3 b")
        _ = NASAExoplanetArchive(wasp_3_b)

    def test_resolve(self):
        wasp_3_b = Sky("wasp-3 b")
        exoplanet = NASAExoplanetArchive(wasp_3_b)
        for i in ["pl_name", "pl_orbper", "pl_orbpererr1", "pl_orbpererr2"]:
            self.assertIn(i, exoplanet.resolve().columns)

    def test_period(self):
        wasp_3_b = Sky("wasp-3 b")
        exoplanet = NASAExoplanetArchive(wasp_3_b)
        for i in ['name', 'P', 'PEmin', 'PEmax']:
            self.assertIn(i, exoplanet.period().columns)

    def test_create_does_not_Exist(self):
        xy_leo = Sky("xy leo")
        exoplanet = NASAExoplanetArchive(xy_leo)

        with self.assertRaises(NoNASAExoplanetArchiveObjectError):
            _ = exoplanet.resolve()
