from logging import Logger
from typing import Iterator, Optional, Callable, Any, Union

import numpy as np
from astropy import units
from matplotlib import pyplot as plt
from numpy import polyfit
from scipy.interpolate import splrep, splev, UnivariateSpline
from scipy.signal import argrelextrema, savgol_filter, butter, filtfilt, medfilt
from typing_extensions import Self

from lightkurve import LightCurve
from astropy.time import Time, TimeDelta

from .kvw import kvw
from .lc_model import ModelXLightCurve
from .utils import interpolate, normalize, neighbor, Numeric, logger_checker, tqdmify
from .minima import Minima
from .boundaries import Boundaries


class XLightCurve(ModelXLightCurve, LightCurve):
    def __init__(self, *args, verbose: bool = False, logger: Optional[Logger] = None, **kwargs):
        self.logger = logger_checker(logger, __name__)
        self.verbose = verbose

        super().__init__(*args, **kwargs)

    @classmethod
    def from_lightcurve(cls, lightcurve: LightCurve, verbose: bool = False, logger: Optional[Logger] = None) -> Self:
        """
        Constructs an instance of the class from a LightCurve object.

        Parameters
        ----------
        lightcurve: LightCurve
            An object containing the time, flux, and flux_err data.
        logger: Logger, default = None
            An optional logger for logging information.

         Returns
        -------
        XLightCurve
            An instance of the class populated with data from the LightCurve object.
        """
        return cls(
            time=lightcurve.time, flux=lightcurve.flux, flux_err=lightcurve.flux_err,
            verbose=verbose, logger=logger
        )

    def boundaries_extrema(self, sigma_multiplier: float = 1.0) -> Boundaries:
        """
        Calculates the extrema (boundaries) of time values where the flux is a certain number of standard deviations below the mean flux.

        Parameters
        ----------
        sigma_multiplier: float
            A multiplier for the standard deviation of the flux. Default is 1.0.

        Returns
        -------
        Boundaries
            An object containing left and right boundary times.
        """
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
                for ri in tqdmify(right_indices, verbose=self.verbose)
            ],
            format='jd'
        )

        left_indices.sort()
        left_time = Time(
            [
                times[li].jd
                for li in tqdmify(left_indices, verbose=self.verbose)
            ],
            format='jd'
        )

        return Boundaries(
            left_time, right_time,
        )

    def boundaries_period(self, minimum_time: Union[Time, Numeric], period: Union[units.Quantity["time"], Numeric],
                          gap: Optional[Union[units.Quantity["time"], Numeric]] = None) -> Boundaries:
        """
        Calculates the time boundaries based on a given period and minimum time.

        Notes
        -----
        The method computes the boundaries by first determining the epochs when the events recur,
        then calculating the left and right boundaries around each epoch based on the specified gap.

        Parameters
        ----------
        minimum_time : Union[Time, Numeric]
            The reference minimum time. Can be an instance of `Time` or a numeric value representing Julian Date.
        period : Union[units.Quantity["time"], Numeric]
            The period after which events recur. Can be an instance of a time quantity or a numeric value (in days).
        gap : Union[units.Quantity["time"], Numeric], default = None
            The time gap on either side of the calculated events. Defaults to half of the period.

        Returns
        -------
        Boundaries
            A Boundaries instance containing the calculated left and right time boundaries for each event.

        Raises
        ------
        ValueError
            If the provided gap is larger than half of the period.
        """
        if isinstance(minimum_time, (float, int)):
            the_minimum_time = Time(minimum_time, format='jd')
        else:
            the_minimum_time = minimum_time

        if isinstance(period, (float, int)):
            the_period = period * units.day
        else:
            the_period = period

        if gap is None:
            the_gap = the_period / 2
        else:
            if isinstance(gap, (float, int)):
                the_gap = gap * units.day
            else:
                the_gap = gap

        if the_gap > the_period / 2:
            raise ValueError("gap must be smaller than half of the period")

        epochs = np.round((self.time - the_minimum_time) / the_period)
        minimas = the_minimum_time + epochs * the_period

        left_time = Time(
            [
                minima - the_gap.to("day").value * the_period.to("day").value
                for minima in tqdmify(minimas, verbose=self.verbose)
            ],
            format='jd'
        )

        right_time = Time(
            [
                minima + the_gap.to("day").value * the_period.to("day").value
                for minima in tqdmify(minimas, verbose=self.verbose)
            ],
            format='jd'
        )

        return Boundaries(left_time, right_time)

    def boundaries_clicker(self) -> Boundaries:

        lower = []
        upper = []

        fig, ax = plt.subplots()
        ax.plot(self.time.jd, self.flux.value)
        ax.set_title('Click to add Ctrl+click to remove')

        def find_closest_line(lines, x_click):
            if not lines:
                return None
            distances = [abs(line.get_xdata()[0] - x_click) for line in lines]
            closest_index = np.argmin(distances)
            return lines[closest_index] if distances[closest_index] < 0.1 else None

        def onclick(event):
            # Check if no tool (e.g., zoom or pan) is active in the toolbar
            if fig.canvas.manager.toolbar.mode == '' and event.inaxes:
                x = event.xdata  # x-coordinate of the click

                if event.key == 'control':  # If Ctrl key is held
                    if event.button == 1:  # Left-click
                        closest_line = find_closest_line(lower, x)
                        if closest_line:
                            closest_line.remove()  # Remove the line from the plot
                            lower.remove(closest_line)  # Remove the line from our list
                    elif event.button == 3:  # Right-click
                        closest_line = find_closest_line(upper, x)
                        if closest_line:
                            closest_line.remove()
                            upper.remove(closest_line)
                else:
                    # Draw a red line for left-click, blue for right-click
                    if event.button == 1:  # Left mouse button
                        line = ax.axvline(x=x, color='red', linestyle='--')
                        lower.append(line)
                    elif event.button == 3:  # Right mouse button
                        line = ax.axvline(x=x, color='blue', linestyle='--')
                        upper.append(line)

                # Redraw the plot to update the display
                fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        return Boundaries(
            Time([float(eac.get_xdata()[0]) for eac in lower], format="jd"),
            Time([float(eac.get_xdata()[0]) for eac in upper], format="jd"),
            logger=self.logger
        )

    def chop(self, boundaries: Optional[Boundaries] = None) -> Iterator[Self]:
        """
        Iterator that yields chopped segments of the time series data based on the given boundaries.

        Parameters
        ----------
        boundaries: Boundaries, default = None
            Optional parameter specifying the boundaries within which to chop the time series. If not provided, the extrema boundaries are used.

        Yields
        ------
        Self
            Sliced segments of the time series data within the specified boundaries.
        """
        if boundaries is None:
            boundaries_to_use = self.boundaries_extrema()
        else:
            boundaries_to_use = boundaries

        for boundary in boundaries_to_use:
            yield self[np.logical_and(self.time.jd > boundary.lower.jd, self.time.jd < boundary.upper.jd)]

    def smooth_b_spline(self, s: float = 1.0) -> Self:
        """
        Generates a smoothed version of the light curve using B-spline fitting.

        Notes
        -----
        The method applies B-spline fitting (using `splrep` and `splev` from `scipy.interpolate`) to the flux data
        points of the light curve, excluding any masked (NaN) values. The smoothed flux values are then used to create
        a new instance of the light curve class.

        Parameters
        ----------
        s: float, default = 1.0
            Smoothing factor used in the B-spline fitting. Default is 1.0.

        Returns
        -------
        Self
            An instance of the light curve class with smoothed flux values.
        """
        fluxes = np.array(self.flux.value)
        flux_mask = np.isnan(fluxes)

        bspl = splrep(self.time.jd[~flux_mask], np.array(self.flux.value)[~flux_mask], s=s)

        xlc = self.__class__(
            time=self.time[~flux_mask],
            flux=splev(self.time.jd[~flux_mask], bspl) * self.flux.unit,
            flux_err=self.flux_err[~flux_mask],
            logger=self.logger
        )
        return xlc

    def smooth_savgol_filter(self, window: int = 21, order: int = 2) -> Self:
        """
        Smooths the flux using a Savitzky-Golay filter.

        Notes
        -----
        The method applies a Savitzky-Golay filter to the flux values of the object, effectively smoothing the data.
        The filtered data excludes any NaN values in the flux.

        Parameters
        ----------
        window: float, int = 21
            The length of the filter window (must be a positive odd integer).
        order: float, int = 2
            The order of the polynomial used to fit the samples. Must be less than the window length.

        Returns
        -------
        Self
            A new instance of the class with the smoothed flux values.
        """
        fluxes = np.array(self.flux.value)
        flux_mask = np.isnan(fluxes)

        xlc = self.__class__(
            time=self.time[~flux_mask],
            flux=savgol_filter(np.array(self.flux.value)[~flux_mask], window, order) * self.flux.unit,
            flux_err=self.flux_err[~flux_mask],
            logger=self.logger
        )

        return xlc

    # https://www.delftstack.com/howto/python/smooth-data-in-python/
    def smooth_butterworth_filter(self, cutoff_freq: float = 0.5, sampling_rate: float = 10.0, order: int = 4) -> Self:
        """
        Applies a Butterworth low-pass filter to the flux data of the light curve, smoothing it.

        Parameters
        ----------
        cutoff_freq: float, default = 0.5
            The cutoff frequency for the filter.
        sampling_rate: float, default = 10.0
            The sampling rate of the data.
        order: int, default = 4
            The order of the Butterworth filter.

        Returns
        -------
        Self
            A new instance of the current class with the smoothed flux data.
        """
        fluxes = np.array(self.flux.value)
        flux_mask = np.isnan(fluxes)

        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        ba = butter(order, normal_cutoff, btype="low", analog=False)

        xlc = self.__class__(
            time=self.time[~flux_mask],
            flux=filtfilt(*ba, np.array(self.flux.value)[~flux_mask]) * self.flux.unit,
            flux_err=self.flux_err[~flux_mask],
            logger=self.logger
        )
        return xlc

    def minima_mean(self, boundaries: Optional[Boundaries] = None) -> Minima:
        """
        Calculates the mean of minima times from a chopped time series within optional boundaries.

        Parameters
        ----------
        boundaries: Boundaries, default = None
            Optional boundaries to constrain the time series data for calculating each minima.

        Returns
        -------
        Minima
            A data structure containing the times of the minima and their associated errors.
        """
        times = []
        time_error = []
        for chopped in tqdmify(self.chop(boundaries), verbose=self.verbose):
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
        return Minima(times, error, logger=self.logger)

    def minima_median(self, boundaries: Optional[Boundaries] = None) -> Minima:
        """
        Computes the median times of chopped time intervals and their associated errors.

        Parameters
        ----------
        boundaries: Boundaries, default = None
            Optional boundaries to constrain the time series data for calculating each minima.

        Returns
        -------
        Minima
            A data structure containing the times of the minima and their associated errors.
        """
        times = []
        time_error = []
        for chopped in tqdmify(self.chop(boundaries), verbose=self.verbose):
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

        return Minima(times, error, logger=self.logger)

    def minima_local(self, boundaries: Optional[Boundaries] = None) -> Minima:
        """
        Finds local minima of the flux values within specified boundaries.

        Parameters
        ----------
        boundaries: Boundaries, default = None
            Optional boundaries to constrain the time series data for calculating each minima.

        Returns
        -------
        Minima
            A data structure containing the times of the minima and their associated errors.
        """
        times = []
        time_error = []
        for chopped in tqdmify(self.chop(boundaries), verbose=self.verbose):
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

        return Minima(times, error, logger=self.logger)

    def minima_fit(self, boundaries: Optional[Boundaries] = None, deg: int = 2) -> Minima:
        """
        Finds the time of the minima in the data using polynomial fitting.

        Notes
        -----
        This method iterates over the chopped time series data within the specified boundaries, fits a polynomial of the
        given degree to the flux values, and calculates the roots of the polynomial's derivative to find the minima.
        It collects the times of these minima and their associated residuals (errors).
        The result is encapsulated in a Minima object containing the times and their corresponding errors.

        Parameters
        ----------
        boundaries: Boundaries, default = None
            Optional boundaries to constrain the time series data for calculating each minima.

        Returns
        -------
        Minima
            A data structure containing the times of the minima and their associated errors.
        """
        times = []
        time_error = []

        for chopped in tqdmify(self.chop(boundaries), verbose=self.verbose):
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

        return Minima(times, error, logger=self.logger)

    def minima_periodogram(self) -> Minima:
        """
        Determines the minima points in the periodogram of the light curve.

        Notes
        -----
        This method normalizes the light curve, converts it to a periodogram, and folds it at the period where the periodogram's power is maximized.
        It then identifies the first minimum point in the folded light curve and iterates to determine subsequent minima points using the period where
        the periodogram's power is maximized. Errors in the minima timings are currently set to NaN.

        Returns
        -------
        Minima
            An object containing the times of the minima points and their respective errors.

        """
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

        return Minima(times, error, logger=self.logger)

    # https://github.com/hdeeg/KvW
    def minima_kwee_van_woerden(self, number_of_folds: int = 5, time_off: bool = True, init_min_flux: bool = False,
                                boundaries: Optional[Boundaries] = None) -> Minima:
        """
        Determines the minima in time series data using the Kwee-van Woerden method.

        Notes
        -----
        https://github.com/hdeeg/KvW

        Parameters
        ----------
        number_of_folds: int, default = 5
            The number of folds to apply in phase space
        time_off: bool, default = True
            Whether to apply a time offset correction
        init_min_flux: bool, default = False
            Whether to use the initial minimum flux to approximate the first minimum
        boundaries: Boundaries, default = None
            Optional boundaries to constrain the time series data for calculating each minima.

        Returns
        -------
        Minima
            A data structure containing the times of the minima and their associated errors.
        """
        times = []
        time_error = []

        for chopped in tqdmify(self.chop(boundaries), verbose=self.verbose):
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

        return Minima(times, error, logger=self.logger)

    def minima_bisector(self, middle_selector: Optional[Callable[[Any], Numeric]] = None, number_of_chords: int = 5,
                        sigma_multiplier: float = 0.1, boundaries: Optional[Boundaries] = None) -> Minima:
        """
        Calculates the minima bisector of the chopped flux data.

        This method uses a specified number of chords to find the bisector of the flux minima within the given boundaries.
        It calculates the bisector by selecting points on each chord and interpolating them to find the central time.

        Parameters
        ----------
        middle_selector: Callable[[Any], Numeric], default = None
            function to determine the middle value. Defaults to numpy.mean if not provided.
        number_of_chords: number_of_chords, default = 5
            Number of chords to be used for the bisector calculation. Must be greater than 3.
        sigma_multiplier: float, default = 0.1
            Multiplier used in calculating the flux lines for generating the chords.
        boundaries: Boundaries, default = None
            Optional boundaries to constrain the time series data for calculating each minima.

        Returns
        -------
        Minima
            A data structure containing the times of the minima and their associated errors.

        Raises
        -------
        ValueError
            If the number of chords is less than 3.
        """
        if middle_selector is None:
            middle_selector = np.mean

        if number_of_chords < 3:
            raise ValueError("number of chord must be greater than 3")

        times = []
        time_error = []
        for chopped in tqdmify(self.chop(boundaries), verbose=self.verbose):
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
                for the_flux in tqdmify(flux_lines, verbose=self.verbose):
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

        return Minima(times, error, logger=self.logger)

    def mid_egress_fielder(self, boundaries: Optional[Boundaries] = None) -> Minima:
        """
        Calculates the approximate egress time based on the derivative of the flux data within given boundaries.

        Notes
        -----
        First method of estimating egress of the WD is the spline fitting derivative method [e.g.,
        Fiedler et al., 1997]. This method is depicted in Figure 2 and individual steps are:
        1. Points of light curve are smoothed by an approximative cubic spline function.
        2. Derivation of the approximative cubic spline function.
        3. The time of the mid-egress is time of maximum of the derivative.

        Parameters
        ----------
        boundaries: Boundaries, default = None
            Optional boundaries to constrain the time series data for calculating each minima.

        Returns
        -------
        Minima
            An object that contains the calculated egress times and their associated errors.
        """
        times = []
        time_error = []

        for chopped in tqdmify(self.chop(boundaries), verbose=self.verbose):
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

        return Minima(times, error, logger=self.logger)

    def mid_ingress_fielder(self, boundaries: Optional[Boundaries] = None) -> Minima:
        """
        Finds the mid-ingress times of a celestial object using spline interpolation.

        Notes
        _____
        First method of estimating egress of the WD is the spline fitting derivative method [e.g.,
        Fiedler et al., 1997]. This method is depicted in Figure 2 and individual steps are:
        1. Points of light curve are smoothed by an approximative cubic spline function.
        2. Derivation of the approximative cubic spline function.
        3. The time of the mid-egress is time of minimum of the derivative.

        Parameters
        ----------
        boundaries: Boundaries, default = None
            Optional boundaries to constrain the time series data for calculating each minima.

        Returns
        -------
        Minima
            An object containing the calculated mid-ingress times and their associated errors.
        """
        times = []
        time_error = []

        for chopped in tqdmify(self.chop(boundaries), verbose=self.verbose):
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

        return Minima(times, error, logger=self.logger)

    def minima_fielder(self, boundaries: Optional[Boundaries] = None) -> Minima:
        """
        Finds the minima in the light curve data within the given boundaries.

        Parameters
        ----------
        boundaries: Boundaries, default = None
            Optional boundaries to constrain the time series data for calculating each minima.

        Returns
        -------
        Minima
            A data structure containing the times of the minima and their associated errors.
        """
        times = []
        time_error = []

        for chopped in tqdmify(self.chop(boundaries), verbose=self.verbose):
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

        return Minima(times, error, logger=self.logger)

    def mid_egress_wood(self, boundaries: Optional[Boundaries] = None, median_width: int = 5,
                        avg_width: int = 5, egress_width: float = 10.0) -> Minima:
        """
        Estimates the mid-egress time for each chopped portion of the light curve data and computes the associated error.

        Notes
        -----
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

        Parameters
        ----------
        boundaries: Boundaries, default = None
            Optional boundaries to constrain the time series data for calculating each minima.
        median_width: int, default = 5
            Kernel size for the median filter applied to the flux values.
        avg_width: int, default = 5
            Window length for the Savitzky-Golay filter applied to the derivative of the smoothed flux.
        egress_width: float, default = 10.0
            Threshold for identifying significant changes in the gradient of the smoothed flux values.

        Returns
        -------
        Minima
            Object containing the computed mid-egress times and their associated errors.
        """
        times = []
        time_error = []

        for chopped in tqdmify(self.chop(boundaries), verbose=self.verbose):
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

        return Minima(times, error, logger=self.logger)

    def mid_ingress_wood(self, boundaries: Optional[Boundaries] = None, median_width: int = 5,
                         avg_width: int = 5, egress_width: float = 10.0) -> Minima:
        """
        Estimates the mid-ingress time for the given boundaries using signal processing techniques.

        Notes
        -----
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

        Parameters
        ----------
        boundaries: Boundaries, default = None
            Optional boundaries to constrain the time series data for calculating each minima.
        median_width: int, default = 5
            The window size to use for the median filter to smooth the flux data.
        avg_width: int, default = 5
            The window size to use for the Savitzky-Golay filter to smooth the derivative of the flux data.
        egress_width: float, default = 10.0
            The threshold value to identify the egress points based on the smoothed derivative.

        Returns
        -------
        Minima
            An instance of Minima containing the calculated mid-ingress times and their corresponding uncertainties.
        """
        times = []
        time_error = []

        for chopped in tqdmify(self.chop(boundaries), verbose=self.verbose):
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

        return Minima(times, error, logger=self.logger)

    def minima_thoroughgood(self, boundaries: Optional[Boundaries] = None) -> Minima:
        """
        Calculates the minima using a parabola fit to the chopped flux data.

        Notes
        -----
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

        Parameters
        ----------
        boundaries: Boundaries, default = None
            Optional boundaries to constrain the time series data for calculating each minima.

        Returns
        -------
        Minima
            An object containing the calculated times of minima and their associated errors.
        """
        times = []
        time_error = []

        for chopped in tqdmify(self.chop(boundaries), verbose=self.verbose):
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

        return Minima(times, error, logger=self.logger)

    def period(self) -> Time:
        """
        Calculates the period corresponding to the maximum power in the periodogram.

        Returns
        -------
        Time
            The period at which the periodogram has its maximum power.
        """
        pg = self.to_periodogram()
        return pg.period_at_max_power
