from octans import Portal, Period
from matplotlib import pyplot as plt


name = "OO Aql"

# Create a Portal and Period Object
portal = Portal.from_name(name)
period = Period.from_name(name)

# Retrieve Periods from all available sources
period_values = period.all()
var_astro_period_value = period_values[period_values["source"] == "VAS"].iloc[0].P

# Retrieve minima times from VarAstro
var_astro_minimas = portal.var_astro()

# Create O-C
var_astro_oc = var_astro_minimas.oc(period=var_astro_period_value)
var_astro_oc.plot(marker=".", color="red", linestyle="None")
plt.title(f"O-C Diagram of {name}")
plt.xlabel("Time (JD)")
plt.ylabel("O-C (JD)")
plt.show()
