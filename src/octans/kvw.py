from logging import Logger

import numpy as np
from logging import getLogger
from typing import Optional
from scipy.optimize import curve_fit


# https://github.com/hdeeg/KvW
def kvw(time, flux, init_minflux=False, rms=None, nfold=5, notimeoff=False, debug=0, logger: Optional[Logger] = None):
    if logger is None:
        logger = getLogger(__name__)

    errflag = 0
    npts = len(time)
    minid = npts // 2

    time0 = 0
    if not notimeoff:
        time0 = np.floor(time[minid] * 10) / 10
        time -= time0

    if debug >= 3:
        for i in range(npts):
            logger.info(i, flux[i])

    difftime = time[1:] - time[:-1]
    mediandiff = np.median(difftime)
    if np.max(difftime) / mediandiff >= 1.01 or mediandiff / np.min(difftime) >= 1.01:
        logger.warning('KVW: WARN: Time-points not equidistant within 1%. Raised errorflag')
        errflag += 1

    if init_minflux:
        minid = int(np.argmin(flux))
        if debug >= 2:
            logger.info('minid from init_minflux', minid)

    if debug >= 1:
        logger.info('minid ', minid)

    noffr = (nfold - 1) / 4.0
    noff = int(noffr)
    minfoldid = minid - noff
    maxfoldid = minid + noff
    nleft = minfoldid
    nright = npts - maxfoldid - 1
    z = min(nleft, nright)
    minpid = minfoldid - z
    maxpid = maxfoldid + z

    if debug >= 2:
        logger.info('points considered in pairings:', minpid, ' to ', maxpid)

    if debug >= 1:
        logger.info('Z: ', z)

    if z < 3:
        raise ValueError(
            'Error: Less than 3 points in in/egress can be paired. Decrease nfold parameter or provide more datapoints for in- or egress')

    s = np.zeros(nfold)
    foldidf = np.zeros(nfold)
    for segid in range(nfold):
        foldidf[segid] = minid - noffr + segid / 2.0
        foldidi = int(foldidf[segid] + 0.0001)
        if abs(foldidf[segid] - foldidi) <= 0.01:
            for i in range(1, z + 1):
                idlo = foldidi - i
                idhi = foldidi + i
                s[segid] += (flux[idlo] - flux[idhi]) ** 2
        else:
            for i in range(1, z + 1):
                idlo = foldidi - i + 1
                idhi = foldidi + i
                s[segid] += (flux[idlo] - flux[idhi]) ** 2

    minidf = int(np.floor(foldidf[0]))
    maxidf = int(np.floor(foldidf[-1] + 0.50001))
    popt, _ = curve_fit(lambda x, a, b: a + b * x, np.arange(minidf, maxidf + 1), time[minidf:maxidf + 1])
    if debug >= 2:
        logger.info('popt: ', popt)

    timef = popt[0] + popt[1] * foldidf
    if debug >= 2:
        logger.info('timef (orig): ', timef)

    if nfold >= 5:
        minSid = np.argmin(s)
        nSleft = minSid
        nSright = nfold - minSid - 1

        if nSleft - nSright >= 2:  # More than 2 more points on left than on right, cut left points
            timef = timef[nSleft - nSright - 1:]
            s = s[nSleft - nSright - 1:]

        if nSright - nSleft >= 2:  # More than 2 more points on right than on left, cut right points
            timef = timef[:nSleft - nSright + 1]
            s = s[:nSleft - nSright + 1]
        if debug >= 2:
            logger.info('nSleft: ', nSleft, ' nSright: ', nSright)

    if debug >= 2:
        logger.info('foldidf ', foldidf)
        logger.info('timef ', timef)
        logger.info('     S ', s)

    cp = np.polyfit(timef, s, 2, full=False)
    if debug >= 1:
        logger.info('cp (kvw): ', cp)

    mintim = -cp[1] / (2.0 * cp[0]) + time0
    kvwminerr = np.sqrt((4 * cp[0] * cp[2] - cp[1] ** 2) / (4 * cp[0] ** 2 * (z - 1)))

    if rms is None:
        rms = np.sqrt(np.min(s) / (2 * (z - 1)))

    cp[2] = (z - 1) * 2 * rms ** 2 + cp[1] ** 2 / (4 * cp[0])
    minerr = np.sqrt((4 * cp[0] * cp[2] - cp[1] ** 2) / (4 * cp[0] ** 2 * (z - 1)))
    if debug >= 1:
        logger.info('cp (revised): ', cp)
        logger.info('type of cp:', type(cp))
        logger.info('rms:', rms)
        logger.info('minerr: ', minerr)

    return mintim, minerr, kvwminerr, errflag
