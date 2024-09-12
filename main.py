import numpy as np
from matplotlib import pyplot as plt

from octans import Portal

# portal = Portal.from_coordinates(281.28812083, 42.45108092)
portal = Portal.from_name("kepler-8")
xlcs = portal.kepler()

xlc = xlcs[0]

minimas = xlc.minima_mean()

fig, ax = plt.subplots()
xlc.plot(ax=ax, color="blue")

for minima in minimas:
    ax.axvline(minima.time.jd, color='red')
# 2455105.5860413993 ± 0.028896315427280064
# 2455109.1106923865 ± 0.02284451533079593

plt.show()
