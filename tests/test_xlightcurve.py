from unittest import TestCase

from octans import XLightCurve
import numpy as np

np.random.seed(42)


class TestXLightCurve(TestCase):
    def setUp(self):
        self.time = np.linspace(2460619.0, 2460619.30, 100)
        self.flux = np.sin(self.time * np.pi * 20)
        self.flux[self.flux > 0.5] = self.flux.max()
        self.flux += 1.1

        self.mean = self.flux.mean() / 2
        self.std = self.flux.std() / 10

        noise = np.random.normal(self.mean, self.std, self.flux.shape)
        self.flux += noise

    def test_create(self):
        _ = XLightCurve(time=self.time.tolist(), flux=self.flux.tolist())

    def test_boundaries(self):
        xlc = XLightCurve(time=self.time.tolist(), flux=self.flux.tolist())
        boundaries = xlc.boundaries_extrema()
        self.assertEqual(len(boundaries), 3)
        boundaries_to_check = [
            [2460619.0636363635, 2460619.084848485],
            [2460619.1666666665, 2460619.187878788],
            [2460619.2636363637, 2460619.2878787876]
        ]

        for each_boundary, boundary_to_check in zip(boundaries, boundaries_to_check):
            self.assertAlmostEquals(each_boundary.lower.jd, boundary_to_check[0])
            self.assertAlmostEquals(each_boundary.upper.jd, boundary_to_check[1])

    def test_boundaries_period(self):
        xlc = XLightCurve(time=self.time.tolist(), flux=self.flux.tolist())
        minimas = xlc.minima_mean()
        period = np.diff(minimas.time.jd).mean()
        t0 = minimas[0]
        boundaries = xlc.boundaries_period(t0.time, period)
        self.assertEqual(len(boundaries), 4)
        boundaries_to_check = [
            [2460619.0666340576, 2460619.086482542],
            [2460619.1658764817, 2460619.185724966],
            [2460619.265118906, 2460619.2849673904]
        ]
        for each_boundary, boundary_to_check in zip(boundaries, boundaries_to_check):
            self.assertAlmostEquals(each_boundary.lower.jd, boundary_to_check[0])
            self.assertAlmostEquals(each_boundary.upper.jd, boundary_to_check[1])

    def test_chop(self):
        xlc = XLightCurve(time=self.time.tolist(), flux=self.flux.tolist())
        chopped_xlc = xlc.chop()
        self.assertEqual(len(list(chopped_xlc)), 3)

    def test_smooth_b_spline(self):
        xlc = XLightCurve(time=self.time.tolist(), flux=self.flux.tolist())
        smooth_xlc = xlc.smooth_b_spline()
        self.assertNotEqual((smooth_xlc.flux - self.flux).value.mean(), 0)

    def test_smooth_savgol_filter(self):
        xlc = XLightCurve(time=self.time.tolist(), flux=self.flux.tolist())
        smooth_xlc = xlc.smooth_b_spline()
        self.assertNotEqual((smooth_xlc.flux - self.flux).value.mean(), 0)

    def test_smooth_butterworth_filter(self):
        xlc = XLightCurve(time=self.time.tolist(), flux=self.flux.tolist())
        smooth_xlc = xlc.smooth_butterworth_filter()
        self.assertNotEqual((smooth_xlc.flux - self.flux).value.mean(), 0)
