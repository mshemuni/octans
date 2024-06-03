from astropy import units

from .models import ModelXLightCurve
from .kvw import kvw
from .utils import normalize, interpolate, neighbor

import logging
from typing import Optional, Union, List, Literal, Dict, Callable, Iterable

import numpy as np
from astropy.time import Time, TimeDelta
from numpy import polyfit
from scipy.signal import butter, filtfilt, argrelextrema
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from scipy.interpolate import splrep, splev
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

    def from_indices(self, indices: List[int]):
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

    def minima_mean(self, boundaries: Optional[Union[np.ndarray, List]] = None) -> Time:
        """Find minima times of each boundary using the mean value of the time"""
        if boundaries is None:
            bound = self.boundaries_extrema()
        else:
            bound = boundaries

        time = []
        for lower, upper in bound:
            mean_time = np.mean(self.time[np.logical_and(self.time.jd > lower, self.time.jd < upper)].jd)
            if not np.isnan(mean_time):
                time.append(mean_time)

        return Time(
            time,
            scale=self.time.scale, format=self.time.format
        )

    def minima_median(self, boundaries: Optional[Union[np.ndarray, List]] = None) -> Time:
        """Find minima times of each boundary using the median value of the time"""
        if boundaries is None:
            bound = self.boundaries_extrema()
        else:
            bound = boundaries

        time = []
        for lower, upper in bound:
            median_time = np.median(self.time[np.logical_and(self.time.jd > lower, self.time.jd < upper)].jd)
            if not np.isnan(median_time):
                time.append(median_time)

        return Time(
            time,
            scale=self.time.scale, format=self.time.format
        )

    def minima_local(self, boundaries: Optional[Union[np.ndarray, List]] = None) -> Time:
        """Find minima times of each boundary using the local minima of the time"""
        if boundaries is None:
            bound = self.boundaries_extrema()
        else:
            bound = boundaries

        time = []
        for lower, upper in bound:
            chopped_data = self[np.logical_and(self.time.jd > lower, self.time.jd < upper)]

            local_minimas = argrelextrema(chopped_data.flux.value, np.less)[0]
            if len(local_minimas) == 0:
                continue
            time.append(chopped_data.time[local_minimas].jd[0])

        return Time(
            time,
            scale=self.time.scale, format=self.time.format
        )

    # https://chat.openai.com/
    def minima_fit(self, boundaries: Optional[Union[np.ndarray, List]] = None, deg: int = 2) -> Time:
        """Find minima times of each boundary using the derivative of the fitted polynomial of the time"""
        if boundaries is None:
            bound = self.boundaries_extrema()
        else:
            bound = boundaries

        times = []

        for lower, upper in bound:
            chopped_data = self[np.logical_and(self.time.jd > lower, self.time.jd < upper)]
            if len(chopped_data) <= 2:
                continue
            coefficients = polyfit(chopped_data.time.jd, chopped_data.flux.value, deg=deg)
            derivative_coefficients = np.polyder(coefficients)
            roots = np.roots(derivative_coefficients)

            times.append(roots[0])

        return Time(
            times,
            scale=self.time.scale, format=self.time.format
        )

    # http://docs.lightkurve.org/tutorials/3-science-examples/periodograms-creating-periodograms.html?highlight=lomb%20scargle%20periodogram
    def minima_periodogram(self) -> Time:
        """Find minima times of each boundary using the periodogram of the lightcurve"""
        pg = self.normalize(unit='ppm').to_periodogram()
        folded_xlc = self.fold(pg.period_at_max_power)

        first_minimum = folded_xlc.time[folded_xlc.flux == folded_xlc.flux.min()][0].jd + self.time.min().jd
        times = []
        while first_minimum < self.time.max().jd:
            first_minimum += pg.period_at_max_power.value
            times.append(first_minimum)

        return Time(
            times[:-1],
            scale=self.time.scale, format=self.time.format
        )

    # https://github.com/hdeeg/KvW
    def minima_kwee_van_woerden(self, number_of_folds: int = 5, time_off: bool = True, init_min_flux: bool = False,
                                boundaries: Optional[Union[np.ndarray, List]] = None) -> Time:
        """Find minima times of each boundary using the kwee van woerden method"""
        if boundaries is None:
            bound = self.boundaries_extrema()
        else:
            bound = boundaries

        times = []

        for lower, upper in bound:
            chopped_data = self[np.logical_and(self.time.jd > lower, self.time.jd < upper)]

            try:
                minima, minima_err, kvw_err, flag = kvw(
                    chopped_data.time.value, chopped_data.flux.value,
                    nfold=number_of_folds, notimeoff=not time_off,
                    init_minflux=init_min_flux, rms=self.flux_err.value.mean()
                )
                times.append(minima)
            except Exception as e:
                log.warning(e)

        return Time(times, scale=self.time.scale, format=self.time.format)

    def minima_chord(self, middle_selector: Optional[Callable] = None, number_of_chords: int = 5,
                     sigma_multiplier: float = 0.1, boundaries: Optional[Union[np.ndarray, List]] = None,
                     fit_degree: int = 2) -> Time:
        """Find minima times of each boundary using the chord method"""

        if middle_selector is None:
            middle_selector = np.mean

        if number_of_chords < 3:
            raise ValueError("number of chord must be greater than 3")

        if boundaries is None:
            bound = self.boundaries_extrema()
        else:
            bound = boundaries

        times = []

        for lower, upper in bound:
            try:
                chopped_data = self[np.logical_and(self.time.jd > lower, self.time.jd < upper)].smooth_b_spline()

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
            except Exception as e:
                log.warning(e)
        return Time(times, scale=self.time.scale, format=self.time.format)

    def minima(self, minima_type: Literal["local", "mean", "median", "fit", "periodogram", "all", "kvw"] = "fit",
               boundaries: Optional[Union[np.ndarray, List]] = None) -> Union[Dict[str, Time], Time]:
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
        elif minima_type.lower() == "chord":
            return self.minima_chord(boundaries=boundaries)
        elif minima_type.lower() == "all":
            return {
                "periodogram": self.minima_periodogram(),
                "local": self.minima_local(boundaries=boundaries),
                "mean": self.minima_mean(boundaries=boundaries),
                "median": self.minima_median(boundaries=boundaries),
                "fit": self.minima_fit(boundaries=boundaries),
                "kvw": self.minima_kwee_van_woerden(boundaries=boundaries),
                "chord": self.minima_chord(boundaries=boundaries),
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
        phase = ((self.time.jd - minimum_time.jd - period / 2) / period) % 1
        return self.__class__(phase, self.flux, self.flux_err)
