import unittest

import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord

from octans import Sky

from astropy.coordinates import name_resolve


class TestFitsArray(unittest.TestCase):
    def setUp(self):
        self.NAME = "XY LEO"
        self.SAMPLE = Sky(self.NAME)

    def test___str__(self):
        string = str(self.SAMPLE)

        self.assertTrue(string.endswith(")"))
        self.assertTrue(string.startswith(f"{self.SAMPLE.__class__.__name__}"))

    def test___repr__(self):
        string = repr(self.SAMPLE)

        self.assertTrue(string.endswith(")"))
        self.assertTrue(string.startswith(f"{self.SAMPLE.__class__.__name__}"))

    def test_name(self):
        self.assertEqual(self.SAMPLE.name, self.NAME)

    def test_name_update(self):
        self.SAMPLE.name = "AU SER"
        self.assertEqual(self.SAMPLE.name, "AU SER")

    def test_name_update_wrong(self):
        with self.assertRaises(name_resolve.NameResolveError):
            self.SAMPLE.name = "YX LEO"

    def test_skycoord(self):
        self.assertAlmostEqual(
            self.SAMPLE.skycoord.ra.deg, 150.41840437
        )
        self.assertAlmostEqual(
            self.SAMPLE.skycoord.dec.deg, 17.40905616
        )

    def test_skycoord_update(self):
        skycoord = SkyCoord(ra=150.41840437, dec=17.40905616, unit=units.deg)
        with self.assertRaises(AttributeError):
            self.SAMPLE.skycoord = skycoord

    def test_from_coordinates(self):
        sample = Sky.from_coordinates(150.41840437, 17.40905616)
        self.assertEqual(sample.name, "V* XY Leo")

    def test_from_coordinates_radian(self):
        sample = Sky.from_coordinates(np.radians(150.41840435) * units.rad, np.radians(17.40905616) * units.rad)
        self.assertEqual(sample.name, "V* XY Leo")

    def test_from_coordinates_degree(self):
        sample = Sky.from_coordinates(150.41840435 * units.deg, 17.40905616 * units.deg)
        self.assertEqual(sample.name, "V* XY Leo")

    def test_from_coordinates_angle_both(self):
        sample = Sky.from_coordinates(150.41840435 * units.deg, np.radians(17.40905616) * units.rad)
        self.assertEqual(sample.name, "V* XY Leo")

    def test_from_coordinates_angle_htob(self):
        sample = Sky.from_coordinates(np.radians(150.41840435) * units.rad, 17.40905616 * units.deg)
        self.assertEqual(sample.name, "V* XY Leo")

    def test_from_coordinates_not_found(self):
        with self.assertRaises(ValueError):
            _ = Sky.from_coordinates(0.001, 0.02, 0)

    def test_from_coordinates_bad_unit(self):
        with self.assertRaises(ValueError):
            _ = Sky.from_coordinates(150.41840435 * units.m, 17.40905616 * units.deg)


if __name__ == '__main__':
    unittest.main()
