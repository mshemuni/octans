from logging import Logger, getLogger
from typing import Iterator, Optional, Callable, Any

import numpy as np
from numpy import polyfit
from scipy.interpolate import splrep, splev, UnivariateSpline
from scipy.signal import argrelextrema, savgol_filter, butter, filtfilt, medfilt
from typing_extensions import Self

from lightkurve import LightCurve
from astropy.time import Time, TimeDelta

from .kvw import kvw
from .lc_model import ModelXLightCurve
from .utils import Boundaries, Minima, interpolate, normalize, neighbor, TimeType


class XLightCurve(ModelXLightCurve, LightCurve):
    def __init__(self, *args, logger: Optional[Logger] = None, **kwargs):
        if logger is None:
            self.logger = getLogger(__name__)
        else:
            self.logger = logger

        super().__init__(*args, **kwargs)

    @classmethod
    def from_lightcurve(cls, lightcurve: LightCurve, logger: Optional[Logger] = None):
        return cls(
            time=lightcurve.time, flux=lightcurve.flux, flux_err=lightcurve.flux_err, logger=logger
        )

    def boundaries_extrema(self, sigma_multiplier: float = 1.0) -> Boundaries:
        flux_std = self.flux.std()
        flux_mean = self.flux.mean()
        times = self.time[self.flux < flux_mean - sigma_multiplier * flux_std]
        right_diffs = np.diff(times.jd)
        right_indices = np.concatenate([
            np.where(right_diffs > right_diffs.mean())[0],
            np.array([len(times) - 1])
        ])

        left_diffs = abs(np.diff(times.jd[::-1]))
        left_indices = len(times) - 1 - np.concatenate([
            np.array([len(times) - 1]),
            np.where(left_diffs > left_diffs.mean())[0]
        ])

        right_indices.sort()
        right_time = Time(
            [
                times[ri].jd
                for ri in right_indices
            ],
            format='jd'
        )

        left_indices.sort()
        left_time = Time(
            [
                times[li].jd
                for li in left_indices
            ],
            format='jd'
        )

        return Boundaries(
            left_time, right_time,
        )

    def boundaries_period(self, t0: Time, period: float, gap: float = 0.1) -> Boundaries:
        if gap >= 0.5:
            raise ValueError("Gap must be greater than 0.5")

        time_min = self.time.min()
        time_max = self.time.max()
        diff = (t0 - time_min).jd % period
        minimas = np.arange(time_min.jd + diff, time_max.jd + diff, period)

        left_time = Time(
            [
                minima - gap * period
                for minima in minimas
            ],
            format='jd'
        )

        right_time = Time(
            [
                minima + gap * period
                for minima in minimas
            ],
            format='jd'
        )

        return Boundaries(left_time, right_time)

    def chop(self, boundaries: Optional[Boundaries] = None) -> Iterator[Self]:

        if boundaries is None:
            boundaries_to_use = self.boundaries_extrema()
        else:
            boundaries_to_use = boundaries

        for boundary in boundaries_to_use:
            yield self[np.logical_and(self.time.jd > boundary.lower.jd, self.time.jd < boundary.upper.jd)]

    def smooth_b_spline(self, s: float = 1.0) -> Self:
        fluxes = np.array(self.flux.value)
        flux_mask = np.isnan(fluxes)

        bspl = splrep(self.time.jd[~flux_mask], np.array(self.flux.value)[~flux_mask], s=s)

        xlc = self.__class__(
            time=self.time[~flux_mask],
            flux=splev(self.time.jd[~flux_mask], bspl) * self.flux.unit,
            flux_err=self.flux_err[~flux_mask]
        )
        return xlc

    def smooth_savgol_filter(self, window: int = 21, order: int = 2) -> Self:
        fluxes = np.array(self.flux.value)
        flux_mask = np.isnan(fluxes)

        xlc = self.__class__(
            time=self.time[~flux_mask],
            flux=savgol_filter(np.array(self.flux.value)[~flux_mask], window, order) * self.flux.unit,
            flux_err=self.flux_err[~flux_mask]
        )

        return xlc

    # https://www.delftstack.com/howto/python/smooth-data-in-python/
    def smooth_butterworth_filter(self, cutoff_freq: float = 0.5, sampling_rate: float = 10.0,
                                  order: int = 4) -> Self:
        fluxes = np.array(self.flux.value)
        flux_mask = np.isnan(fluxes)

        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        ba = butter(order, normal_cutoff, btype="low", analog=False)

        xlc = self.__class__(
            time=self.time[~flux_mask],
            flux=filtfilt(*ba, np.array(self.flux.value)[~flux_mask]) * self.flux.unit,
            flux_err=self.flux_err[~flux_mask]
        )
        return xlc

    def minima_mean(self, boundaries: Optional[Boundaries] = None) -> Minima:
        times = []
        time_error = []
        for chopped in self.chop(boundaries):
            try:
                mean_time = np.mean(chopped.time.jd)
                std_time = np.std(chopped.time.jd)
                if not np.isnan(mean_time):
                    times.append(mean_time)
                    time_error.append(std_time)
            except Exception as e:
                self.logger.warning(e)

        times = Time(
            times,
            format='jd'
        )

        error = TimeDelta(
            time_error, format='jd'
        )
        return Minima(times, error)

    def minima_median(self, boundaries: Optional[Boundaries] = None) -> Minima:
        times = []
        time_error = []
        for chopped in self.chop(boundaries):
            try:
                median_time = np.median(chopped.time.jd)
                if not np.isnan(median_time):
                    times.append(median_time)
                    q1 = np.percentile(chopped.time.jd, 25)
                    q3 = np.percentile(chopped.time.jd, 75)
                    time_error.append(max(abs(q1 - median_time), abs(q3 - median_time)))
            except Exception as e:
                self.logger.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return Minima(times, error)

    def minima_local(self, boundaries: Optional[Boundaries] = None) -> Minima:
        times = []
        time_error = []
        for chopped in self.chop(boundaries):
            try:
                local_minimas = argrelextrema(chopped.flux.value, np.less)
                if len(local_minimas) == 0:
                    self.logger.warning("No local minima was found")
                    continue

                local_minima = local_minimas[0]

                if len(local_minima) == 0:
                    self.logger.warning("No local minima was found")
                    continue

                times.append(chopped.time[local_minimas].jd[0])
                time_error.append(np.nan)
            except Exception as e:
                self.logger.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return Minima(times, error)

    def minima_fit(self, boundaries: Optional[Boundaries] = None, deg: int = 2) -> Minima:
        times = []
        time_error = []

        for chopped in self.chop(boundaries):
            try:
                if len(chopped) <= 2:
                    continue
                coefficients, (res,), *_ = polyfit(chopped.time.jd, chopped.flux.value, deg=deg, full=True)
                derivative_coefficients = np.polyder(coefficients)
                roots = np.roots(derivative_coefficients)

                times.append(roots[0])
                time_error.append(res)
            except Exception as e:
                self.logger.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return Minima(times, error)

    def minima_periodogram(self) -> Minima:

        pg = self.normalize(unit='ppm').to_periodogram()
        folded_xlc = self.fold(pg.period_at_max_power)

        first_minimum = folded_xlc.time[folded_xlc.flux == folded_xlc.flux.min()][0].jd + self.time.min().jd
        times = []
        time_error = []
        while first_minimum < self.time.max().jd:
            try:
                first_minimum += pg.period_at_max_power.value
                times.append(first_minimum)
                time_error.append(np.nan)
            except Exception as e:
                self.logger.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return Minima(times, error)

    # https://github.com/hdeeg/KvW
    def minima_kwee_van_woerden(self, number_of_folds: int = 5, time_off: bool = True, init_min_flux: bool = False,
                                boundaries: Optional[Boundaries] = None) -> Minima:
        times = []
        time_error = []

        for chopped in self.chop(boundaries):
            try:
                minima, minima_err, kvw_err, flag = kvw(
                    chopped.time.jd, chopped.flux.value,
                    nfold=number_of_folds, notimeoff=not time_off,
                    init_minflux=init_min_flux, rms=self.flux_err.value.mean()
                )
                times.append(minima)
                time_error.append(kvw_err)

            except Exception as e:
                self.logger.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return Minima(times, error)

    def minima_bisector(self, middle_selector: Optional[Callable[[Any], float]] = None, number_of_chords: int = 5,
                        sigma_multiplier: float = 0.1, boundaries: Optional[Boundaries] = None) -> Minima:
        """Find minima times of each boundary using the bisector method"""
        if middle_selector is None:
            middle_selector = np.mean

        if number_of_chords < 3:
            raise ValueError("number of chord must be greater than 3")

        times = []
        time_error = []
        for chopped in self.chop(boundaries):
            try:
                std_time = np.std(chopped.time.jd)

                arg_middle = neighbor(chopped.time.jd, middle_selector(chopped.time.jd))

                flux_argmin = np.argmin(chopped.flux)
                flux_min = chopped.flux[flux_argmin]
                flux_max = min(chopped[:arg_middle].flux.max(), chopped[arg_middle:].flux.max())
                flux_std = chopped.flux.std()
                flux_lines = np.linspace(flux_min + flux_std * sigma_multiplier,
                                         flux_max - flux_std * sigma_multiplier, number_of_chords + 2)[1:-1]
                points = []
                for the_flux in flux_lines:
                    left_signs = np.sign(chopped[:arg_middle].flux - the_flux)
                    left_sign_change_index = ((np.roll(left_signs, 1) - left_signs) != 0).astype(int)[1:].argmax()

                    right_signs = np.sign(chopped[arg_middle:].flux - the_flux)
                    right_sign_change_index = ((np.roll(right_signs, 1) - right_signs) != 0).astype(int)[1:].argmax()

                    left_point_1_flux = chopped[:arg_middle].flux[left_sign_change_index]
                    left_point_2_flux = chopped[:arg_middle].flux[left_sign_change_index + 1]
                    right_point_1_flux = chopped[arg_middle:].flux[right_sign_change_index]
                    right_point_2_flux = chopped[arg_middle:].flux[right_sign_change_index + 1]
                    left_flux_propagation = normalize(the_flux, left_point_1_flux, left_point_2_flux)
                    right_flux_propagation = normalize(the_flux, right_point_1_flux, right_point_2_flux)

                    left_point_1_time = chopped[:arg_middle].time[left_sign_change_index].jd
                    left_point_2_time = chopped[:arg_middle].time[left_sign_change_index + 1].jd
                    right_point_1_time = chopped[arg_middle:].time[right_sign_change_index].jd
                    right_point_2_time = chopped[arg_middle:].time[right_sign_change_index + 1].jd

                    left_time = interpolate(left_flux_propagation, left_point_1_time, left_point_2_time)
                    right_time = interpolate(right_flux_propagation, right_point_1_time, right_point_2_time)

                    points.append((left_time + right_time) / 2)

                times.append(np.mean(points))
                time_error.append(std_time)
            except Exception as e:
                self.logger.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return Minima(times, error)

    def mid_egress_fielder(self, boundaries: Optional[Boundaries] = None) -> Minima:
        """
        First method of estimating egress of the WD is the spline fitting derivative method [e.g.,
            Fiedler et al., 1997]. This method is depicted in Figure 2 and individual steps are:
            1. Points of light curve are smoothed by an approximative cubic spline function.
            2. Derivation of the approximative cubic spline function.
            3. The time of the mid-egress is time of maximum of the derivative.
        """
        times = []
        time_error = []

        for chopped in self.chop(boundaries):
            try:
                spline = UnivariateSpline(chopped.time.jd, chopped.flux.value, s=0, k=3)
                derivative = spline.derivative()
                derivative_times = np.linspace(chopped.time.jd.min(), chopped.time.jd.max(), 1000)
                derivative_values = derivative(derivative_times)
                egress_time = derivative_times[np.argmax(derivative_values)]

                times.append(egress_time)
                time_error.append(np.nan)
            except Exception as e:
                self.logger.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return Minima(times, error)

    def mid_ingress_fielder(self, boundaries: Optional[Boundaries] = None) -> Minima:
        """
        First method of estimating egress of the WD is the spline fitting derivative method [e.g.,
            Fiedler et al., 1997]. This method is depicted in Figure 2 and individual steps are:
            1. Points of light curve are smoothed by an approximative cubic spline function.
            2. Derivation of the approximative cubic spline function.
            3. The time of the mid-egress is time of minimum of the derivative.
        """
        times = []
        time_error = []

        for chopped in self.chop(boundaries):
            try:
                spline = UnivariateSpline(chopped.time.jd, chopped.flux.value, s=0, k=3)
                derivative = spline.derivative()
                derivative_times = np.linspace(chopped.time.jd.min(), chopped.time.jd.max(), 1000)
                derivative_values = derivative(derivative_times)
                ingress_time = derivative_times[np.argmin(derivative_values)]

                times.append(ingress_time)
                time_error.append(np.nan)
            except Exception as e:
                self.logger.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return Minima(times, error)

    def minima_fielder(self, boundaries: Optional[Boundaries] = None) -> Minima:
        times = []
        time_error = []

        for chopped in self.chop(boundaries):
            try:
                spline = UnivariateSpline(chopped.time.jd, chopped.flux.value, s=0, k=3)
                derivative = spline.derivative()
                derivative_times = np.linspace(chopped.time.jd.min(), chopped.time.jd.max(), 1000)
                derivative_values = derivative(derivative_times)
                egress_time = derivative_times[np.argmin(derivative_values)]
                ingress_time = derivative_times[np.argmax(derivative_values)]

                times.append((egress_time + ingress_time) / 2)
                time_error.append(np.nan)
            except Exception as e:
                self.logger.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return Minima(times, error)

    def mid_egress_wood(self, boundaries: Optional[Boundaries] = None, median_width: int = 5,
                        avg_width: int = 5, egress_width: float = 10.0) -> Minima:
        """
        Second method was proposed by Wood et al. [1985]. Method uses median and average filter
            for smoothing of the points of the light curve. Median filter of a given width changes each point
            of the light curve to a median of a surrounding points of the given width and average filter
            changes each point on a average value of the points in a surrounding of the given width. This
            method is depicted in Figure 3 and individual steps are:
            1. Points of light curve are smoothed by median filter. This step can be omitted, if the
            measurements have small uncertainties.
            2. Derivative of smoothed light curve.
            3. Using average filter of width equal to the expected duration of egress.
            4. A spline function is fitted to points that are not lying within expected egress.
            5. Times of the beginning and the end of the egress of the WD are the same as times where
            the derivative differs greatly from the spline function.
            6. The time of the mid-egress of the WD is average value of these two values.
        """
        times = []
        time_error = []

        for chopped in self.chop(boundaries):
            try:
                smoothed_curve = medfilt(chopped.flux.value, kernel_size=median_width)
                if len(smoothed_curve) < 3:
                    self.logger.warning("Could not find enough points for egress estimation")
                    continue

                derivative = np.gradient(smoothed_curve, chopped.time.jd)
                smoothed_derivative = savgol_filter(derivative, window_length=avg_width, polyorder=2)
                egress_indices = np.where(np.abs(smoothed_derivative) < egress_width)[0]
                spline_fit = UnivariateSpline(
                    chopped.time.jd[egress_indices], smoothed_derivative[egress_indices], s=0
                )
                diff = np.abs(smoothed_derivative - spline_fit(chopped.time.jd))
                significant_diff_indices = np.where(diff > np.percentile(diff, 95))[0]

                if len(significant_diff_indices) < 2:
                    self.logger.warning("Could not find enough significant points for egress estimation")
                    continue

                start_egress_time = chopped.time.jd[significant_diff_indices[0]]
                end_egress_time = chopped.time.jd[significant_diff_indices[-1]]

                mid_egress_time = (start_egress_time + end_egress_time) / 2

                times.append(mid_egress_time)
                time_error.append(np.nan)
            except Exception as e:
                self.logger.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return Minima(times, error)

    def mid_ingress_wood(self, boundaries: Optional[Boundaries] = None, median_width: int = 5,
                         avg_width: int = 5, egress_width: float = 10.0) -> Minima:
        """
        Second method was proposed by Wood et al. [1985]. Method uses median and average filter
            for smoothing of the points of the light curve. Median filter of a given width changes each point
            of the light curve to a median of a surrounding points of the given width and average filter
            changes each point on a average value of the points in a surrounding of the given width. This
            method is depicted in Figure 3 and individual steps are:
            1. Points of light curve are smoothed by median filter. This step can be omitted, if the
            measurements have small uncertainties.
            2. Derivative of smoothed light curve.
            3. Using average filter of width equal to the expected duration of egress.
            4. A spline function is fitted to points that are not lying within expected egress.
            5. Times of the beginning and the end of the egress of the WD are the same as times where
            the derivative differs greatly from the spline function.
            6. The time of the mid-egress of the WD is average value of these two values.
        """
        times = []
        time_error = []

        for chopped in self.chop(boundaries):
            try:
                smoothed_curve = medfilt(chopped.flux.value, kernel_size=median_width)
                if len(smoothed_curve) < 3:
                    self.logger.warning("Could not find enough points for egress estimation")
                    continue

                derivative = np.gradient(smoothed_curve, chopped.time.jd)
                smoothed_derivative = savgol_filter(derivative, window_length=avg_width, polyorder=2)
                ingress_indices = np.where(np.abs(smoothed_derivative) > egress_width)[0]
                spline_fit = UnivariateSpline(
                    chopped.time.jd[ingress_indices], smoothed_derivative[ingress_indices], s=0
                )
                diff = np.abs(smoothed_derivative - spline_fit(chopped.time.jd))
                significant_diff_indices = np.where(diff > np.percentile(diff, 95))[0]

                if len(significant_diff_indices) < 2:
                    self.logger.warning("Could not find enough significant points for ingress estimation")
                    continue

                start_ingress_time = chopped.time.jd[significant_diff_indices[0]]
                end_ingress_time = chopped.time.jd[significant_diff_indices[-1]]

                mid_ingress_time = (start_ingress_time + end_ingress_time) / 2

                times.append(mid_ingress_time)
                time_error.append(np.nan)
            except Exception as e:
                self.logger.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return Minima(times, error)

    def minima_thoroughgood(self, boundaries: Optional[Boundaries] = None) -> Minima:
        """
        The accretion disc outshines the WD and the bright spot. The egress of the WD can be
            hardly identified and therefore derivative methods are not usable. Precise time of the mid-
            eclipse is hard to estimate because of asymmetrical shape of the eclipse. Many authors use a
            parabola fitting method [e.g. Thoroughgood et al., 2004] to the eclipse light curve or to the
            lower half of the eclipse curve. It is important to use same method for all estimations of the
            mid-eclipse times of each nova or nova-like variable to ensure smaller systematic errors. The
            parabola fitting method is depicted on the left part of Figure 4 and individual steps are:
            1. Fitting parabola to the eclipse part of the light curve (or to the lower half of the eclipse
            part of the light curve).
            2. The time of the mid-eclipse is minimum of the parabola.
        """
        times = []
        time_error = []

        for chopped in self.chop(boundaries):
            try:
                coefficients = np.polyfit(chopped.time.jd, chopped.flux.value, 2)
                a, b, _ = coefficients
                x_min = -b / (2 * a)
                times.append(x_min)
                time_error.append(np.nan)
            except Exception as e:
                self.logger.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return Minima(times, error)

    def period(self) -> TimeType:
        pg = self.to_periodogram()
        return pg.period_at_max_power
