from octans import Minima
from astropy.time import Time, TimeDelta

mnm = Minima(Time.now(), TimeDelta(0.02, format="jd"))
print(mnm)
for i in mnm:
    print(i)