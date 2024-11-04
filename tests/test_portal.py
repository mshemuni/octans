from astropy.coordinates import SkyCoord
from astropy.coordinates.name_resolve import NameResolveError

from octans import Sky, Portal
from unittest import TestCase

from octans.errors import NoLightCurveError, NoASASObjectError, NoVarAstroObjectError


class TestSky(TestCase):

    def test_create(self):
        sky = Sky("xy leo")
        portal = Portal(sky)
        sky_coord = SkyCoord.from_name("xy Leo")
        self.assertEqual(portal.sky.skycoord, sky_coord)

    def test_create_from_name(self):
        sky = Sky("xy leo")
        portal = Portal.from_name("xy leo")
        self.assertEqual(sky.skycoord, portal.sky.skycoord)

    def test_create_from_name_does_not_exist(self):
        with self.assertRaises(NameResolveError):
            _ = Portal.from_name("wa dra")

    def test_change_sky_coords(self):
        portal = Portal.from_name("xy leo")
        new_sky = Sky("aw dra")
        with self.assertRaises(AttributeError):
            portal.sky = new_sky

    def test_kepler(self):
        portal = Portal.from_name("kepler9")
        kepler = portal.kepler()
        self.assertEqual(len(list(kepler)), 50)

    def test_kepler_does_not_exist(self):
        portal = Portal.from_name("wasp 3")
        kepler = portal.kepler()
        with self.assertRaises(NoLightCurveError):
            _ = list(kepler)

    def test_tess(self):
        portal = Portal.from_name("kepler9")
        tess = portal.tess()
        self.assertEqual(len(list(tess)), 14)

    def test_tess_does_not_exist(self):
        portal = Portal.from_name("Alpha pegasi")
        tess = portal.tess()
        with self.assertRaises(NoLightCurveError):
            _ = list(tess)

    def test_asas(self):
        portal = Portal.from_name("xy leo")
        asas = portal.asas()
        self.assertEqual(len(list(asas)), 5)

    def test_asas_does_not_exist(self):
        portal = Portal.from_name("aw dra")
        with self.assertRaises(NoASASObjectError):
            _ = portal.asas()

    def test_var_astro(self):
        portal = Portal.from_name("xy leo")
        var_astro = portal.var_astro()
        self.assertEqual(len(var_astro), 317)

    def test_var_astro_does_not_exist(self):
        portal = Portal.from_name("aw dra")
        with self.assertRaises(NoLightCurveError):
            _ = portal.var_astro()

    def test_etd(self):
        portal = Portal.from_name("wasp 2")
        etd = portal.etd()
        self.assertEqual(len(etd), 199)

    def test_etd_does_not_exist(self):
        portal = Portal.from_name("xy leo")
        with self.assertRaises(NoVarAstroObjectError):
            _ = portal.etd()
