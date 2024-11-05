from octans import Portal

# Create a portal for "kepler-8"
portal = Portal.from_name("kepler-8")

# Retrieve light curves
light_curves = portal.kepler()

# Get the first one
light_curve = next(light_curves)

# smooth and flatten
xlc = light_curve.smooth_butterworth_filter().flatten()

# Calculate minima times with different methods
minimas = {
    "Mean": xlc.minima_mean(),
    "Fit": xlc.minima_fit(),
    "KVW": xlc.minima_kwee_van_woerden(),
    "Bisector": xlc.minima_bisector(),
    "fielder": xlc.minima_fielder(),
}