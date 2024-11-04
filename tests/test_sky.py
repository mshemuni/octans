from astropy import units
from astropy.coordinates import SkyCoord
from astropy.coordinates.name_resolve import NameResolveError

from octans import Sky
from unittest import TestCase

from octans.errors import ObjectNotFoundError


class TestSky(TestCase):

    def test_create(self):
        xy_leo = Sky("xy leo")
        sky_coord = SkyCoord.from_name("xy Leo")
        self.assertEqual(xy_leo.skycoord, sky_coord)

    def test_create_from_coordinates(self):
        sky_coord = SkyCoord.from_name("xy Leo")
        xy_leo = Sky.from_coordinates(sky_coord.ra, sky_coord.dec)
        self.assertIn("xy leo", xy_leo.name.lower())

    def test_create_does_not_exist(self):
        with self.assertRaises(NameResolveError):
            _ = Sky("WA Dra")

    def test_create_from_coordinates_does_not_exist(self):
        with self.assertRaises(ObjectNotFoundError):
            _ = Sky.from_coordinates(1, 0, radius=1 * units.arcsec)

    def test_change_name(self):
        xy_leo = Sky("xy leo")
        sky_coord = SkyCoord.from_name("xy Leo")
        self.assertEqual(xy_leo.skycoord.ra, sky_coord.ra)
        self.assertEqual(xy_leo.skycoord.dec, sky_coord.dec)

        xy_leo.name = "AW Dra"
        sky_coord_aw_dra = SkyCoord.from_name("AW DRA")
        self.assertEqual(xy_leo.skycoord.ra, sky_coord_aw_dra.ra)
        self.assertEqual(xy_leo.skycoord.dec, sky_coord_aw_dra.dec)

    def test_change_sky_coord(self):
        xy_leo = Sky("xy leo")
        sky_coord = SkyCoord.from_name("xy Leo")
        self.assertEqual(xy_leo.skycoord.ra, sky_coord.ra)
        self.assertEqual(xy_leo.skycoord.dec, sky_coord.dec)

        sky_coord_aw_dra = SkyCoord.from_name("AW DRA")

        with self.assertRaises(AttributeError):
            xy_leo.skycoord = sky_coord_aw_dra
