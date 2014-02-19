#!/usr/bin/env python
import tempfile
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from obspy import UTCDateTime
from obspy.core import AttribDict
import obspy.signal.array_analysis as AA
import scipy.interpolate as spi
import matplotlib.cm as cm


KM_PER_DEG = 111.1949


def array_analysis_helper(stream, inventory, method, frqlow, frqhigh,
                          filter=True, baz_plot=True, static3D=False,
                          vel_corr=4.8, wlen=None, slx=(-10, 10),
                          sly=(-10, 10), sls=0.2, array_response=True):
    """
    Array analysis wrapper routine for MESS 2014.

    :param stream: Waveforms for the array processing.
    :type stream: :class:`obspy.core.stream.Stream`
    :param inventory: Station metadata for waveforms
    :type inventory: :class:`obspy.station.inventory.Inventory`
    :param method: Method used for the array analysis (one of "FK", "DLS")
    :type method: str
    :param filter: Whether to bandpass data to selected frequency range
    :type filter: bool
    :param frqlow: Low corner of frequency range for array analysis
    :type frqlow: float
    :param frqhigh: High corner of frequency range for array analysis
    :type frqhigh: float
    :param baz_plot: Whether to show backazimuth-slowness map (True) or
        slowness x-y map (False).
    :type baz_plot: str
    :param static3D: static correction of topography using `vel_corr` as
        velocity (slow!)
    :type static3D: bool
    :param vel_corr: Correction velocity for static topography correction in
        km/s.
    :type vel_corr: float
    :param wlen: sliding window for analysis in seconds, 0 to use the whole
        trace without windowing.
    :type wlen: float
    :param slx: Min/Max slowness for analysis in x direction.
    :type slx: (float, float)
    :param sly: Min/Max slowness for analysis in y direction.
    :type sly: (float, float)
    :param sls: step width of slowness grid
    :type sls: float
    :param array_response: superimpose array reponse function in plot (slow!)
    :type array_response: bool
    """
    sllx, slmx = slx
    slly, slmy = sly

    starttime = max([tr.stats.starttime for tr in stream])
    endtime = min([tr.stats.endtime for tr in stream])
    stream.trim(starttime, endtime)

    #stream.attach_response(inventory)
    stream.merge()
    for tr in stream:
        for station in inventory[0].stations:
            if tr.stats.station == station.code:
                tr.stats.coordinates = \
                    AttribDict(dict(latitude=station.latitude,
                                    longitude=station.longitude,
                                    elevation=station.elevation))
                break

    if filter:
        stream.filter('bandpass', freqmin=frqlow, freqmax=frqhigh,
                      zerophase=True)

    print stream

    tmpdir = tempfile.mkdtemp(prefix="obspy-")
    filename_patterns = (os.path.join(tmpdir, 'pow_map_%03d.npy'),
                         os.path.join(tmpdir, 'apow_map_%03d.npy'))

    def dump(pow_map, apow_map, i):
        np.save(filename_patterns[0] % i, pow_map)
        np.save(filename_patterns[1] % i, apow_map)

    try:
        # next step would be needed if the correction velocity needs to be
        # estimated
        # XXX
        velgrid = np.arange(4.8, 4.9, 0.2)
        velgrid = np.arange([vel_corr])
        for vc in velgrid:
            print vc
            sllx /= KM_PER_DEG
            slmx /= KM_PER_DEG
            slly /= KM_PER_DEG
            slmy /= KM_PER_DEG
            sls /= KM_PER_DEG
            if method == 'FK':
                kwargs = dict(
                    #slowness grid: X min, X max, Y min, Y max, Slow Step
                    sll_x=sllx, slm_x=slmx, sll_y=slly, slm_y=slmy, sl_s=sls,
                    # sliding window properties
                    win_len=wlen, win_frac=0.8,
                    # frequency properties
                    frqlow=frqlow, frqhigh=frqhigh, prewhiten=0,
                    # restrict output
                    store=dump,
                    semb_thres=-1e9, vel_thres=-1e9, verbose=False,
                    timestamp='julsec', stime=starttime, etime=endtime,
                    method=0, correct_3dplane=False, vel_cor=vc,
                    static_3D=static3D)

                # here we do the array processing
                start = UTCDateTime()
                out = AA.array_processing(stream, **kwargs)
                print "Total time in routine: %f\n" % (UTCDateTime() - start)

                # make output human readable, adjust backazimuth to values
                # between 0 and 360
                t, rel_power, abs_power, baz, slow = out.T

            else:
                kwargs = dict(
                    # slowness grid: X min, X max, Y min, Y max, Slow Step
                    sll_x=sllx, slm_x=slmx, sll_y=slly, slm_y=slmy, sl_s=sls,
                    # sliding window properties
                    # frequency properties
                    frqlow=frqlow, frqhigh=frqhigh,
                    # restrict output
                    store=dump,
                    win_len=wlen, win_frac=0.5,
                    nthroot=4, method='DLS',
                    verbose=False, timestamp='julsec',
                    stime=starttime, etime=endtime, vel_cor=vc,
                    static_3D=False)

                # here we do the array processing
                start = UTCDateTime()
                out = AA.beamforming(stream, **kwargs)
                print "Total time in routine: %f\n" % (UTCDateTime() - start)

                # make output human readable, adjust backazimuth to values
                # between 0 and 360
                t, rel_power, baz, slow_x, slow_y, slow = out.T

            # calculating array response
            if array_response:
                stepsfreq = (frqhigh - frqlow) / 10.
                tf_slx = sllx
                tf_smx = slmx
                tf_sly = slly
                tf_smy = slmy
                transff = AA.array_transff_freqslowness(
                    stream, (tf_slx, tf_smx, tf_sly, tf_smy), sls, frqlow,
                    frqhigh, stepsfreq, coordsys='lonlat',
                    correct_3dplane=False, static_3D=False, vel_cor=vc)

            # now let's do the plotting
            cmap = cm.rainbow

            #
            # we will plot everything in s/deg
            slow *= KM_PER_DEG
            sllx *= KM_PER_DEG
            slmx *= KM_PER_DEG
            slly *= KM_PER_DEG
            slmy *= KM_PER_DEG
            sls *= KM_PER_DEG

            if method == 'FK':
                spl = stream.copy()
                spl.trim(starttime, endtime)
            else:
                if wlen <= 0.:
                    spl = stream.copy()
                    spl[0].data = max_beam[0]
                    spl.trim(starttime, endtime)

            numslice = len(t)
            powmap = []
            slx = np.arange(sllx-sls, slmx, sls)
            sly = np.arange(slly-sls, slmy, sls)
            if baz_plot:
                maxslowg = np.sqrt(slmx*slmx + slmy*slmy)
                bzs = np.arctan2(sls, np.sqrt(slmx*slmx + slmy*slmy))*180/np.pi
                xi = np.arange(0., maxslowg, sls)
                yi = np.arange(-180., 180., bzs)
                grid_x, grid_y = np.meshgrid(xi, yi)

            for i in xrange(numslice):
                powmap.append(np.load(filename_patterns[0] % i))
                if method != 'FK':
                    trace.append(np.load(filename_patterns[1] % i))

            npts = spl[0].stats.npts
            df = spl[0].stats.sampling_rate
            T = np.arange(0, npts / df, 1 / df)

            for i in xrange(numslice):
                slow_x = np.sin((baz[i]+180.)*np.pi/180.)*slow[i]
                slow_y = np.cos((baz[i]+180.)*np.pi/180.)*slow[i]
                st = UTCDateTime(t[i]) - starttime
                if wlen <= 0:
                    en = endtime
                else:
                    en = st + wlen
                print UTCDateTime(t[i])
                # add polar and colorbar axes
                fig = plt.figure(figsize=(8, 8))
                ax1 = fig.add_axes([0.1, 0.87, 0.7, 0.10])
                if method == 'FK':
                    ax1.plot(T, spl[0].data, 'k')
                    try:
                        ax1.axvspan(st, en, facecolor='g', alpha=0.3)
                    except IndexError:
                        pass
                else:
                    ax1.plot(T, trace[i], 'k')

                ax1.yaxis.set_major_locator(MaxNLocator(3))
                l, u = ax1.get_ylim()

                ax = fig.add_axes([0.10, 0.1, 0.70, 0.7])

                if baz_plot:
                    slowgrid = []
                    transgrid = []
                    pow = np.asarray(powmap[i])
                    for ix, sx in enumerate(slx):
                        for iy, sy in enumerate(sly):
                            bbaz = np.arctan2(sx, sy)*180/np.pi+180.
                            if bbaz > 180.:
                                bbaz = -180. + (bbaz-180.)
                            slowgrid.append((np.sqrt(sx*sx+sy*sy), bbaz,
                                             pow[ix, iy]))
                            if array_response:
                                tslow = (np.sqrt((sx+slow_x) *
                                         (sx+slow_x)+(sy+slow_y) *
                                         (sy+slow_y)))
                                tbaz = (np.arctan2(sx+slow_x, sy+slow_y) *
                                        180 / np.pi + 180.)
                                if tbaz > 180.:
                                    tbaz = -180. + (tbaz-180.)
                                transgrid.append((tslow, tbaz,
                                                  transff[ix, iy]))

                    slowgrid = np.asarray(slowgrid)
                    sl = slowgrid[:, 0]
                    bz = slowgrid[:, 1]
                    slowg = slowgrid[:, 2]
                    grid = spi.griddata((sl, bz), slowg, (grid_x, grid_y),
                                        method='nearest')
                    ax.pcolormesh(xi, yi, grid, cmap=cmap)

                    if array_response:
                        level = np.arange(0.1, 0.5, 0.1)
                        transgrid = np.asarray(transgrid)
                        tsl = transgrid[:, 0]
                        tbz = transgrid[:, 1]
                        transg = transgrid[:, 2]
                        trans = spi.griddata((tsl, tbz), transg,
                                             (grid_x, grid_y),
                                             method='nearest')
                        ax.contour(xi, yi, trans, level, colors='k',
                                   linewidth=0.2)

                    ax.set_xlabel('slowness [s/deg]')
                    ax.set_ylabel('backazimuth [deg]')
                    ax.set_xlim(xi[0], xi[-1])
                    ax.set_ylim(yi[0], yi[-1])
                else:
                    ax.set_xlabel('slowness [s/deg]')
                    ax.set_ylabel('slowness [s/deg]')
                    slow_x = np.cos((baz[i]+180.)*np.pi/180.)*slow[i]
                    slow_y = np.sin((baz[i]+180.)*np.pi/180.)*slow[i]
                    ax.pcolormesh(slx, sly, powmap[i].T)
                    ax.arrow(0, 0, slow_y, slow_x, head_width=0.005,
                             head_length=0.01, fc='k', ec='k')
                    if array_response:
                        tslx = np.arange(sllx+slow_x, slmx+slow_x+sls, sls)
                        tsly = np.arange(slly+slow_y, slmy+slow_y+sls, sls)
                        try:
                            ax.contour(tsly, tslx, transff.T, 5, colors='k',
                                       linewidth=0.5)
                        except:
                            pass
                    ax.set_ylim(slx[0], slx[-1])
                    ax.set_xlim(sly[0], sly[-1])
                new_time = t[i]

                result = "BAZ: %.2f, Slow: %.2f s/deg, Time %s" % (
                    baz[i], slow[i], UTCDateTime(new_time))
                ax.set_title(result)

                plt.show()
    finally:
        shutil.rmtree(tmpdir)
