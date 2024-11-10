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

    def test_boundaries_period_large(self):
        xlc = XLightCurve(time=self.time.tolist(), flux=self.flux.tolist())
        with self.assertRaises(ValueError):
            _ = xlc.boundaries_period(self.time.min(), 1, 0.75)

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
