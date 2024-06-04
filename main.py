from octans import Portal

from matplotlib import pyplot as plt

p = Portal.from_name("Kepler-9")

tess_xlcs = p.tess()

fig, ax = plt.subplots()

for i, tess_xlc in enumerate(tess_xlcs):
    tess_xlc.plot(ax=ax)

ax.get_legend().remove()
plt.show()
