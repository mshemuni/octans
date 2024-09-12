# import numpy as np
# from scipy.interpolate import UnivariateSpline
# import matplotlib.pyplot as plt
# from octans import XLightCurve
# import pandas as pd
# from matplotlib import pyplot as plt
#
# from astropy.time import Time
#
#
# def find_minimum_time_spline(light_curve_times, light_curve_magnitudes):
#     """
#     Finds the minimum time (mid-egress of WD) using the spline fitting derivative method.
#
#     Parameters:
#     - light_curve_times: Array of time values from the light curve observations.
#     - light_curve_magnitudes: Array of corresponding magnitude values.
#
#     Returns:
#     - min_time: The time of the minimum (mid-egress of WD).
#     """
#
#     # Fit a cubic spline to the light curve data
#     spline = UnivariateSpline(light_curve_times, light_curve_magnitudes, s=0, k=3)
#
#     # Calculate the derivative of the spline
#     derivative = spline.derivative()
#
#     # Find the time of the maximum of the derivative
#     # (mid-egress time corresponds to max derivative point)
#     derivative_times = np.linspace(light_curve_times.min(), light_curve_times.max(), 1000)
#     derivative_values = derivative(derivative_times)
#     min_time = derivative_times[np.argmax(derivative_values)]
#
#     # Plotting for visualization (optional)
#     plt.figure(figsize=(10, 5))
#     plt.plot(light_curve_times, light_curve_magnitudes, 'o', label='Observed Data')
#     plt.plot(derivative_times, spline(derivative_times), label='Cubic Spline Fit')
#     plt.plot(derivative_times, derivative_values, label='Derivative of Spline')
#     plt.axvline(min_time, color='r', linestyle='--', label='Min Time (Mid-egress)')
#     plt.xlabel('Time')
#     plt.ylabel('Magnitude')
#     plt.legend()
#     plt.title('Spline Fitting Derivative Method')
#     plt.show()
#
#     return min_time
#
#
# import numpy as np
# from scipy.signal import medfilt, savgol_filter
# from scipy.interpolate import UnivariateSpline
#
#
# def find_minimum_time(light_curve, time, median_width=5, avg_width=5, egress_width=10):
#     """
#     Find the minimum time of a light curve using the method proposed by Wood et al. [1985].
#
#     Parameters:
#     - light_curve: numpy array of light intensity values.
#     - time: numpy array of time values corresponding to the light curve.
#     - median_width: width of the median filter for smoothing (optional).
#     - avg_width: width of the average filter for smoothing the derivative (optional).
#     - egress_width: expected duration of egress used for filtering and spline fitting.
#
#     Returns:
#     - mid_egress_time: estimated time of mid-egress.
#     """
#     # Step 1: Smooth the light curve using a median filter (optional based on uncertainties)
#     smoothed_curve = medfilt(light_curve, kernel_size=median_width)
#
#     # Step 2: Calculate the derivative of the smoothed light curve
#     derivative = np.gradient(smoothed_curve, time)
#
#     # Step 3: Smooth the derivative using an average filter (Savitzky-Golay filter)
#     smoothed_derivative = savgol_filter(derivative, window_length=avg_width, polyorder=2)
#
#     # Step 4: Fit a spline function to points outside the expected egress region
#     egress_indices = np.where(np.abs(smoothed_derivative) < egress_width)[0]
#     spline_fit = UnivariateSpline(time[egress_indices], smoothed_derivative[egress_indices], s=0)
#
#     # Step 5: Find points where derivative differs significantly from the spline fit
#     diff = np.abs(smoothed_derivative - spline_fit(time))
#     significant_diff_indices = np.where(diff > np.percentile(diff, 95))[0]  # 95th percentile threshold
#
#     if len(significant_diff_indices) < 2:
#         raise ValueError("Could not find enough significant points for egress estimation.")
#
#     # Times at the beginning and end of egress
#     start_egress_time = time[significant_diff_indices[0]]
#     end_egress_time = time[significant_diff_indices[-1]]
#
#     # Step 6: Calculate the mid-egress time
#     mid_egress_time = (start_egress_time + end_egress_time) / 2
#
#     return mid_egress_time
#
#
# df = pd.read_csv("data/CHSS_3072_S56_120__final.csv")
# xlc = XLightCurve(df["TIME"], df["NORM_DT_SAP_FLUX"], df["NORM_DT_SAP_FLUX_ERR"])
#
# b = xlc.boundaries_extrema(sigma_multiplier=0.2)
# ch1 = xlc[(xlc.time.jd < b[0][1]) & (xlc.time.jd > b[0][0])]
# ch1.plot()
# plt.show()
#
# # Example usage
# light_curve_times = np.array(ch1.time.jd)  # Replace with actual time data
# light_curve_magnitudes = np.array(ch1.flux)  # Replace with actual magnitude data
#
# min_time = find_minimum_time_spline(light_curve_times, light_curve_magnitudes)
# print(f"The minimum time (mid-egress of WD) is: {min_time}")
# #
# mt = find_minimum_time(light_curve_magnitudes, light_curve_times)
# print(f"mt is: {mt}")
from astropy.coordinates import SkyCoord

print(SkyCoord.from_name("kepler-8"))