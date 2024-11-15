from logging import Logger

import numpy as np
from typing import Optional
from scipy.optimize import curve_fit

from octans.utils import logger_checker


# https://github.com/hdeeg/KvW
def kvw(time, flux, init_minflux=False, rms=None, nfold=5, notimeoff=False, debug=0, logger: Optional[Logger] = None):
    """
    Returns eclipse mid (minimum)-time using the Kwee Van-Woerden (1956) method with revised timing error, following Deeg 2021, (Galaxies, vol. 9, issue 1, p. 1)
    Data-points that are equidistant in time are required and the lightcurve should only contain the eclipse (no off-eclipse data).
    For the initial guess of the minimum time, two options are given, controlled by the keyword init_minflux : By default (init_minflux=0), the middle of
    the lightcurve is assumed. Else (init_minflux=1), the time of the flux-minimum is used. This option is fine or preferential for low-noise lightcurves showing
    a clear flux-mimimum not extending over more than 2-3 points.
    If the rms (point-to-point noise) of the off-eclipse flux is known, it should be supplied by keyword rms. Else, rms is estimated from S of the best
     flux-pairing, which is assumed to be dominated by flux measurement errors. For details on calculation of that rms, see Deeg 2021.
    By default, the time of minimum is derived from 5 pairings (nfold paramter) that fold at i-1,i-0.5,i,i+0.5,i+1, where i is the index of the point of minimum
     flux. The original KvW algorithm uses nfold=3.

    input
        time   vector with time values in ascending order
        flux   vector with corresponding flux values
    output
        tuple with 4 values:
        time of mimimum, error of minimum (method by Deeg 2021), error of minimum (KvW's orignal method), and error-code.
        error-codes: 0: OK, data are equidistant within 1% against medium spacing.  1: data not equidistantly spaced

    keywords
        init_minflux input:  By default, the middle of the lightcurve is used as intial guess of the minimum time. If 1, uses as initial min-time
                             estimate the value of lowest flux.
        rms     input: average measurement error of individual flux values (in units of the flux values). rms should usually be supplied.
        nfold   input: number of foldings on which to perform pairings of flux values, default=5
        notimeoff input: if set to 1, disables internal offsetting of time values. This is save to use if eclipse min-time is within +-0.1 of time of the intial min-time estimate)
        noplot  input: if set to 1, supresses plots
        noprint input: if set to 1, supresses all text-output
        debug   input: debugging flag. If 1, prints some, if 2, prints lots, if 3 even more intermediate values

      HJD 16nov2023:  First version of kvw.py, translated from kvw.pro version 15nov2023 using Chatgpt4.
        21nov2023: Revised code that delivers identical numerical results as IDL code kvw.pro (ecxept for graphics, which are simpler)
        23nov2023  fixed the citation of Deeg 2021 paper

    CITING this code: Deeg, H.J. 2021, "A Modified Kwee-Van Woerden Method for Eclipse Minimum
                         Timing with Reliable Error Estimates"Galaxies, vol. 9, issue 1, p. 1

    COPYRIGHT (C) 2020, 2023 Hans J. Deeg
        This program is free software you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation either version 2 of the License, or
        any later version.

        As a special exception, this program may be compiled with the
        Interactive Data Language (IDL) and be linked to libaries
        pertaining to IDL.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program in file 'licence.txt' if not, write to
        the Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
        Boston, MA 02110-1301 USA
    ----------------------------------
        """
    logger = logger_checker(logger, __name__)

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
