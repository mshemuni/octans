import matplotlib.pyplot as plt
from octans import XLightCurve
import pandas as pd

# Get data
data = pd.read_csv("data/CHSS_3072_S56_120__final.csv")

# Create XLightCurve object
xlc = XLightCurve(
    data["TIME"],
    data["NORM_DT_SAP_FLUX"],
    data["NORM_DT_SAP_FLUX_ERR"]
)

# Smooth the lightcurve
folded_xlc = xlc.fold_periodogram()

# plot
folded_xlc.plot()
plt.show()
