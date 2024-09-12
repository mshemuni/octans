import matplotlib.pyplot as plt
from astropy import units

from .models import ModelXLightCurve
from .kvw import kvw
from .utils import normalize, interpolate, neighbor, TimeError

import logging
from typing import Optional, Union, List, Literal, Dict, Callable

import numpy as np
from astropy.time import Time, TimeDelta
from numpy import polyfit
from scipy.signal import butter, filtfilt, argrelextrema, medfilt
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from scipy.interpolate import splrep, splev, UnivariateSpline
from scipy.signal import savgol_filter
from typing_extensions import Self
import copy
from lightkurve import LightCurve

from .utils import time_checker

log = logging.getLogger(__name__)


class XLightCurve(ModelXLightCurve, LightCurve):
    """
    XLightCurve object
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return f"{self.__class__.__name__}(@: {id(self)}, n: {len(self)})"

    @classmethod
    def from_lightkurve(cls, lightkurve: LightCurve) -> Self:
        """
        Create a XLightCurve from a LightCurve
        """
        instance = super().__new__(cls)
        instance.__dict__ = copy.deepcopy(lightkurve.__dict__)
        return instance

    def from_indices(self, indices: List[int]) -> Self:
        """
        Returns a new XLightCurve from given indices.
        """
        times, fluxes, flux_errors = [], [], []

        for ind in indices:
            times.append(self.time[ind].jd)
            fluxes.append(self.flux[ind].value)
            flux_errors.append(self.flux_err[ind].value)

        return self.__class__(times, fluxes, flux_errors)

    def set_time(self, times: Time) -> None:
        """
        Sets new Time
        """

        time_checker(times, unit=units.day)

        self.time.writeable = True
        self.time = times
        self.time.writeable = False

    def update_time(self, amount: TimeDelta) -> None:
        """
        Updates time
        """

        time_checker(amount, unit=units.day)

        self.time.writeable = True
        self.time += amount
        self.time.writeable = False

    def to_hjd(self, sky: SkyCoord, loc: EarthLocation) -> Self:
        """Convert time to HJD"""
        ltt_heli = self.time.light_travel_time(sky, location=loc, kind="heliocentric")
        xlc = self.__class__.from_lightkurve(self)
        xlc.update_time(ltt_heli)
        return xlc

    def to_bjd(self, sky: SkyCoord, loc: EarthLocation) -> Self:
        """Convert time to BJD"""
        ltt_bary = self.time.light_travel_time(sky, location=loc, kind="barycentric")
        xlc = self.__class__.from_lightkurve(self)
        xlc.update_time(ltt_bary)
        return xlc

    def boundaries_period(self, t0: Time, period: float, gap: float = 0.1):
        if gap >= 0.5:
            raise ValueError("Gap must be greater than 0.5")

        time_min = self.time.min()
        time_max = self.time.max()
        diff = (t0 - time_min).jd % period
        minimas = np.arange(time_min.jd + diff, time_max.jd - diff, period)
        boundaries = np.array([
            [minima - gap * period, minima + gap * period]
            for minima in minimas
        ])
        return boundaries

    def boundaries_extrema(self, sigma_multiplier: float = 1.0) -> List[List[int]]:
        """Find boundaries of each eclipse using extrema method"""
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
        left_indices.sort()

        return [
            [times[li].jd, times[ri].jd]
            for li, ri in zip(left_indices, right_indices)
        ]

    def minima_mean(self, boundaries: Optional[Union[np.ndarray, List]] = None) -> TimeError:
        """Find minima times of each boundary using the mean value of the time"""
        if boundaries is None:
            bound = self.boundaries_extrema()
        else:
            bound = boundaries

        times = []
        time_error = []
        for lower, upper in bound:
            try:
                chopped = self.time[np.logical_and(self.time.jd > lower, self.time.jd < upper)].jd
                mean_time = np.mean(chopped)
                std_time = np.std(chopped)
                if not np.isnan(mean_time):
                    times.append(mean_time)
                    time_error.append(std_time)
            except Exception as e:
                log.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return TimeError(times, error)

    def minima_median(self, boundaries: Optional[Union[np.ndarray, List]] = None) -> TimeError:
        """Find minima times of each boundary using the median value of the time"""
        if boundaries is None:
            bound = self.boundaries_extrema()
        else:
            bound = boundaries

        times = []
        time_error = []
        for lower, upper in bound:
            try:
                chopped = self.time[np.logical_and(self.time.jd > lower, self.time.jd < upper)].jd
                median_time = np.median(chopped)
                if not np.isnan(median_time):
                    times.append(median_time)
                    q1 = np.percentile(chopped, 25)
                    q3 = np.percentile(chopped, 75)
                    time_error.append(max(abs(q1 - median_time), abs(q3 - median_time)))
            except Exception as e:
                log.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return TimeError(times, error)

    def minima_local(self, boundaries: Optional[Union[np.ndarray, List]] = None) -> TimeError:
        """Find minima times of each boundary using the local minima of the time"""
        if boundaries is None:
            bound = self.boundaries_extrema()
        else:
            bound = boundaries

        times = []
        time_error = []
        for lower, upper in bound:
            try:
                chopped_data = self[np.logical_and(self.time.jd > lower, self.time.jd < upper)]

                local_minimas = argrelextrema(chopped_data.flux.value, np.less)
                if len(local_minimas) == 0:
                    log.warning("No local minima was found")
                    continue

                local_minima = local_minimas[0]

                if len(local_minima) == 0:
                    log.warning("No local minima was found")
                    continue

                times.append(chopped_data.time[local_minimas].jd[0])
                time_error.append(np.nan)
            except Exception as e:
                log.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return TimeError(times, error)

    # https://chat.openai.com/
    def minima_fit(self, boundaries: Optional[Union[np.ndarray, List]] = None, deg: int = 2) -> TimeError:
        """Find minima times of each boundary using the derivative of the fitted polynomial of the time"""
        if boundaries is None:
            bound = self.boundaries_extrema()
        else:
            bound = boundaries

        times = []
        time_error = []

        for lower, upper in bound:
            try:
                chopped_data = self[np.logical_and(self.time.jd > lower, self.time.jd < upper)]
                if len(chopped_data) <= 2:
                    continue
                coefficients, (res,), *_ = polyfit(chopped_data.time.jd, chopped_data.flux.value, deg=deg, full=True)
                derivative_coefficients = np.polyder(coefficients)
                roots = np.roots(derivative_coefficients)

                times.append(roots[0])
                time_error.append(res)
            except Exception as e:
                log.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return TimeError(times, error)

    # http://docs.lightkurve.org/tutorials/3-science-examples/periodograms-creating-periodograms.html?highlight=lomb%20scargle%20periodogram
    def minima_periodogram(self) -> TimeError:
        """Find minima times of each boundary using the periodogram of the lightcurve"""

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
                log.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return TimeError(times, error)

    # https://github.com/hdeeg/KvW
    def minima_kwee_van_woerden(self, number_of_folds: int = 5, time_off: bool = True, init_min_flux: bool = False,
                                boundaries: Optional[Union[np.ndarray, List]] = None) -> TimeError:
        """Find minima times of each boundary using the kwee van woerden method"""
        # Formal hata. Chi^2 olmalı.
        if boundaries is None:
            bound = self.boundaries_extrema()
        else:
            bound = boundaries

        times = []
        time_error = []

        for lower, upper in bound:
            try:
                chopped_data = self[np.logical_and(self.time.jd > lower, self.time.jd < upper)]
                minima, minima_err, kvw_err, flag = kvw(
                    chopped_data.time.value, chopped_data.flux.value,
                    nfold=number_of_folds, notimeoff=not time_off,
                    init_minflux=init_min_flux, rms=self.flux_err.value.mean()
                )
                times.append(minima)
                time_error.append(kvw_err)

            except Exception as e:
                log.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return TimeError(times, error)

    def minima_bisector(self, middle_selector: Optional[Callable] = None, number_of_chords: int = 5,
                        sigma_multiplier: float = 0.1, boundaries: Optional[Union[np.ndarray, List]] = None,
                        fit_degree: int = 2) -> TimeError:
        """Find minima times of each boundary using the bisector method"""
        if middle_selector is None:
            middle_selector = np.mean

        if number_of_chords < 3:
            raise ValueError("number of chord must be greater than 3")

        if boundaries is None:
            bound = self.boundaries_extrema()
        else:
            bound = boundaries

        times = []
        time_error = []
        for lower, upper in bound:
            try:
                chopped_data = self[np.logical_and(self.time.jd > lower, self.time.jd < upper)].smooth_b_spline()

                std_time = np.std(chopped_data.time.jd)

                arg_middle = neighbor(chopped_data.time.jd, middle_selector(chopped_data.time.jd))

                flux_argmin = np.argmin(chopped_data.flux)
                flux_min = chopped_data.flux[flux_argmin]
                flux_max = min(chopped_data[:arg_middle].flux.max(), chopped_data[arg_middle:].flux.max())
                flux_std = chopped_data.flux.std()
                flux_lines = np.linspace(flux_min + flux_std * sigma_multiplier,
                                         flux_max - flux_std * sigma_multiplier, number_of_chords + 2)[1:-1]
                points = []
                for the_flux in flux_lines:
                    left_signs = np.sign(chopped_data[:arg_middle].flux - the_flux)
                    left_sign_change_index = ((np.roll(left_signs, 1) - left_signs) != 0).astype(int)[1:].argmax()

                    right_signs = np.sign(chopped_data[arg_middle:].flux - the_flux)
                    right_sign_change_index = ((np.roll(right_signs, 1) - right_signs) != 0).astype(int)[1:].argmax()

                    left_point_1_flux = chopped_data[:arg_middle].flux[left_sign_change_index]
                    left_point_2_flux = chopped_data[:arg_middle].flux[left_sign_change_index + 1]
                    right_point_1_flux = chopped_data[arg_middle:].flux[right_sign_change_index]
                    right_point_2_flux = chopped_data[arg_middle:].flux[right_sign_change_index + 1]
                    left_flux_propagation = normalize(the_flux, left_point_1_flux, left_point_2_flux)
                    right_flux_propagation = normalize(the_flux, right_point_1_flux, right_point_2_flux)

                    left_point_1_time = chopped_data[:arg_middle].time[left_sign_change_index].jd
                    left_point_2_time = chopped_data[:arg_middle].time[left_sign_change_index + 1].jd
                    right_point_1_time = chopped_data[arg_middle:].time[right_sign_change_index].jd
                    right_point_2_time = chopped_data[arg_middle:].time[right_sign_change_index + 1].jd

                    left_time = interpolate(left_flux_propagation, left_point_1_time, left_point_2_time)
                    right_time = interpolate(right_flux_propagation, right_point_1_time, right_point_2_time)

                    points.append((left_time + right_time) / 2)

                times.append(np.mean(points))
                time_error.append(std_time)
            except Exception as e:
                log.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return TimeError(times, error)

    def mid_egress_fielder(self, boundaries: Optional[Union[np.ndarray, List]] = None) -> TimeError:
        """
        First method of estimating egress of the WD is the spline fitting derivative method [e.g.,
            Fiedler et al., 1997]. This method is depicted in Figure 2 and individual steps are:
            1. Points of light curve are smoothed by an approximative cubic spline function.
            2. Derivation of the approximative cubic spline function.
            3. The time of the mid-egress is time of maximum of the derivative.
        """
        if boundaries is None:
            bound = self.boundaries_extrema()
        else:
            bound = boundaries

        times = []
        time_error = []

        for lower, upper in bound:
            try:
                chopped_data = self[np.logical_and(self.time.jd > lower, self.time.jd < upper)]
                spline = UnivariateSpline(chopped_data.time.jd, chopped_data.flux.value, s=0, k=3)
                derivative = spline.derivative()
                derivative_times = np.linspace(chopped_data.time.jd.min(), chopped_data.time.jd.max(), 1000)
                derivative_values = derivative(derivative_times)
                egress_time = derivative_times[np.argmax(derivative_values)]

                times.append(egress_time)
                time_error.append(np.nan)
            except Exception as e:
                log.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return TimeError(times, error)

    def mid_ingress_fielder(self, boundaries: Optional[Union[np.ndarray, List]] = None) -> TimeError:
        """
        First method of estimating egress of the WD is the spline fitting derivative method [e.g.,
            Fiedler et al., 1997]. This method is depicted in Figure 2 and individual steps are:
            1. Points of light curve are smoothed by an approximative cubic spline function.
            2. Derivation of the approximative cubic spline function.
            3. The time of the mid-egress is time of minimum of the derivative.
        """
        if boundaries is None:
            bound = self.boundaries_extrema()
        else:
            bound = boundaries

        times = []
        time_error = []

        for lower, upper in bound:
            try:
                chopped_data = self[np.logical_and(self.time.jd > lower, self.time.jd < upper)]
                spline = UnivariateSpline(chopped_data.time.jd, chopped_data.flux.value, s=0, k=3)
                derivative = spline.derivative()
                derivative_times = np.linspace(chopped_data.time.jd.min(), chopped_data.time.jd.max(), 1000)
                derivative_values = derivative(derivative_times)
                ingress_time = derivative_times[np.argmin(derivative_values)]

                times.append(ingress_time)
                time_error.append(np.nan)
            except Exception as e:
                log.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return TimeError(times, error)

    def minima_fielder(self, boundaries: Optional[Union[np.ndarray, List]] = None) -> TimeError:
        if boundaries is None:
            bound = self.boundaries_extrema()
        else:
            bound = boundaries

        times = []
        time_error = []

        for lower, upper in bound:
            try:
                chopped_data = self[np.logical_and(self.time.jd > lower, self.time.jd < upper)]
                spline = UnivariateSpline(chopped_data.time.jd, chopped_data.flux.value, s=0, k=3)
                derivative = spline.derivative()
                derivative_times = np.linspace(chopped_data.time.jd.min(), chopped_data.time.jd.max(), 1000)
                derivative_values = derivative(derivative_times)
                egress_time = derivative_times[np.argmin(derivative_values)]
                ingress_time = derivative_times[np.argmax(derivative_values)]

                times.append((egress_time + ingress_time) / 2)
                time_error.append(np.nan)
            except Exception as e:
                log.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return TimeError(times, error)

    def mid_egress_wood(self, boundaries: Optional[Union[np.ndarray, List]] = None, median_width: int = 5,
                        avg_width: int = 5, egress_width: float = 10.0) -> TimeError:
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
        if boundaries is None:
            bound = self.boundaries_extrema()
        else:
            bound = boundaries

        times = []
        time_error = []

        for lower, upper in bound:
            try:
                chopped_data = self[np.logical_and(self.time.jd > lower, self.time.jd < upper)]
                smoothed_curve = medfilt(chopped_data.flux.value, kernel_size=median_width)
                if len(smoothed_curve) < 3:
                    log.warning("Could not find enough points for egress estimation")
                    continue

                derivative = np.gradient(smoothed_curve, chopped_data.time.jd)
                smoothed_derivative = savgol_filter(derivative, window_length=avg_width, polyorder=2)
                egress_indices = np.where(np.abs(smoothed_derivative) < egress_width)[0]
                spline_fit = UnivariateSpline(
                    chopped_data.time.jd[egress_indices], smoothed_derivative[egress_indices], s=0
                )
                diff = np.abs(smoothed_derivative - spline_fit(chopped_data.time.jd))
                significant_diff_indices = np.where(diff > np.percentile(diff, 95))[0]

                if len(significant_diff_indices) < 2:
                    log.warning("Could not find enough significant points for egress estimation")
                    continue

                start_egress_time = chopped_data.time.jd[significant_diff_indices[0]]
                end_egress_time = chopped_data.time.jd[significant_diff_indices[-1]]

                mid_egress_time = (start_egress_time + end_egress_time) / 2

                times.append(mid_egress_time)
                time_error.append(np.nan)
            except Exception as e:
                log.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return TimeError(times, error)

    def minima_thoroughgood(self, boundaries: Optional[Union[np.ndarray, List]] = None) -> TimeError:
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
        if boundaries is None:
            bound = self.boundaries_extrema()
        else:
            bound = boundaries

        times = []
        time_error = []

        for lower, upper in bound:
            try:
                chopped_data = self[np.logical_and(self.time.jd > lower, self.time.jd < upper)]
                coefficients = np.polyfit(chopped_data.time.jd, chopped_data.flux.value, 2)
                a, b, _ = coefficients
                x_min = -b / (2 * a)
                times.append(x_min)
                time_error.append(np.nan)
            except Exception as e:
                log.warning(e)

        times = Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

        error = TimeDelta(
            time_error, format='jd'
        )

        return TimeError(times, error)

    def minima(self,
               minima_type: Literal["local", "mean", "median", "fit", "periodogram", "all", "kvw", "bisector"] = "fit",
               boundaries: Optional[Union[np.ndarray, List]] = None) -> Union[Dict[str, TimeError], TimeError]:
        """Find minimas of all available methods"""
        if minima_type.lower() == "periodogram":
            return self.minima_periodogram()
        elif minima_type.lower() == "local":
            return self.minima_local(boundaries=boundaries)
        elif minima_type.lower() == "mean":
            return self.minima_mean(boundaries=boundaries)
        elif minima_type.lower() == "median":
            return self.minima_median(boundaries=boundaries)
        elif minima_type.lower() == "fit":
            return self.minima_fit(boundaries=boundaries)
        elif minima_type.lower() == "kvw":
            return self.minima_kwee_van_woerden(boundaries=boundaries)
        elif minima_type.lower() == "bisector":
            return self.minima_bisector(boundaries=boundaries)
        elif minima_type.lower() == "all":
            return {
                "periodogram": self.minima_periodogram(),
                "local": self.minima_local(boundaries=boundaries),
                "mean": self.minima_mean(boundaries=boundaries),
                "median": self.minima_median(boundaries=boundaries),
                "fit": self.minima_fit(boundaries=boundaries),
                "kvw": self.minima_kwee_van_woerden(boundaries=boundaries),
                "bisector": self.minima_bisector(boundaries=boundaries),
            }

        raise ValueError("Invalid minima type")

    # https://stackoverflow.com/questions/46633544/smoothing-out-a-curve
    def smooth_b_spline(self, s: float = 1.0) -> Self:
        """Smooth the lightcurve using B-Spline filter"""
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
        """Smooth the lightcurve using Savitzky-Golay filter"""
        fluxes = np.array(self.flux.value)
        flux_mask = np.isnan(fluxes)

        xlc = self.__class__(
            time=self.time[~flux_mask],
            flux=savgol_filter(np.array(self.flux.value)[~flux_mask], window, order) * self.flux.unit,
            flux_err=self.flux_err[~flux_mask]
        )

        return xlc

    # https://www.delftstack.com/howto/python/smooth-data-in-python/
    def smooth_butterworth_filter(self, cutoff_freq: float = 0.5, sampling_rate: float = 10.0, order: int = 4) -> Self:
        """Smooth the lightcurve using Butterworth filter"""
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

    def fold_periodogram(self, unit: Literal["ppm", "ppt"] = "ppm") -> Self:
        """Fold the lightcurve using periodogram"""
        pg = self.normalize(unit=unit).to_periodogram()
        folded_xlc = self.fold(pg.period_at_max_power)
        return self.__class__(folded_xlc.time, folded_xlc.flux, folded_xlc.flux_err)

    def fold_phase(self, minimum_time: Time, period: float) -> Self:
        """Fold the lightcurve using Phase"""
        phase = ((self.time.jd - minimum_time.jd - period.to("day").value / 2) / period.to("day").value) % 1
        return self.__class__(phase, self.flux, self.flux_err)

