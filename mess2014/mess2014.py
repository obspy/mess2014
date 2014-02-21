#!/usr/bin/env python
from collections import defaultdict
import tempfile
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from obspy import UTCDateTime, Stream
from obspy.core import AttribDict
from obspy.core.util.geodetics import locations2degrees, gps2DistAzimuth, \
    kilometer2degrees
import obspy.signal.array_analysis as AA
from obspy.taup import getTravelTimes
import scipy.interpolate as spi
import matplotlib.cm as cm


KM_PER_DEG = 111.1949


def array_analysis_helper(stream, inventory, method, frqlow, frqhigh,
                          filter=True, baz_plot=True, static3D=False,
                          vel_corr=4.8, wlen=-1, slx=(-10, 10),
                          sly=(-10, 10), sls=0.5, array_response=True):
    """
    Array analysis wrapper routine for MESS 2014.

    :param stream: Waveforms for the array processing.
    :type stream: :class:`obspy.core.stream.Stream`
    :param inventory: Station metadata for waveforms
    :type inventory: :class:`obspy.station.inventory.Inventory`
    :param method: Method used for the array analysis
        (one of "FK", "DLS", "PWS", "SWP").
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
    :param wlen: sliding window for analysis in seconds, use -1 to use the
        whole trace without windowing.
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

    if method not in ("FK", "DLS", "PWS", "SWP"):
        raise ValueError("Invalid method: ''" % method)

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
        velgrid = np.array([vel_corr])
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
                    nthroot=4, method=method,
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
                    #spl[0].data = max_beam[0]
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
                    pass
                    #trace.append(np.load(filename_patterns[1] % i))

            T = spl[0].times()

            for i in xrange(numslice):
                slow_x = np.sin((baz[i]+180.)*np.pi/180.)*slow[i]
                slow_y = np.cos((baz[i]+180.)*np.pi/180.)*slow[i]
                st = UTCDateTime(t[i]) - starttime
                if wlen <= 0:
                    en = T[-1]
                else:
                    en = st + wlen
                print UTCDateTime(t[i])
                # add polar and colorbar axes
                fig = plt.figure(figsize=(8, 8))
                ax1 = fig.add_axes([0.1, 0.87, 0.7, 0.10])
                if method == 'FK':
                    ax1.plot(T, spl[0].data, 'k')
                    ax1.axvspan(st, en, facecolor='g', alpha=0.3)
                else:
                    pass
                    #ax1.plot(T, trace[i], 'k')

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


def attach_coordinates_to_traces(stream, inventory, event=None):
    """
    If event is given, the event distance in degree will also be attached.
    """
    # Get the coordinates for all stations
    coords = {}
    for network in inventory:
        for station in network:
            coords["%s.%s" % (network.code, station.code)] = \
                {"latitude": station.latitude,
                 "longitude": station.longitude,
                 "elevation": station.elevation}

    # Calculate the event-station distances.
    if event:
        event_lat = event.origins[0].latitude
        event_lng = event.origins[0].longitude
        for value in coords.values():
            value["distance"] = locations2degrees(
                value["latitude"], value["longitude"], event_lat, event_lng)

    # Attach the information to the traces.
    for trace in stream:
        station = ".".join(trace.id.split(".")[:2])
        value = coords[station]
        trace.stats.coordinates = AttribDict()
        trace.stats.coordinates.latitude = value["latitude"]
        trace.stats.coordinates.longitude = value["longitude"]
        trace.stats.coordinates.elevation = value["elevation"]
        if event:
            trace.stats.distance = value["distance"]


def show_distance_plot(stream, event, inventory, starttime, endtime,
                       plot_travel_times=True):
    """
    Plots distance dependent waveforms.
    """
    stream = stream.slice(starttime=starttime, endtime=endtime).copy()
    event_depth_in_km = event.origins[0].depth / 1000.0
    event_time = event.origins[0].time

    attach_coordinates_to_traces(stream, inventory, event=event)

    cm = plt.cm.jet

    stream.traces = sorted(stream.traces, key=lambda x: x.stats.distance)[::-1]

    # One color for each trace.
    colors = [cm(_i) for _i in np.linspace(0, 1, len(stream))]

    # Relative event times.
    times_array = stream[0].times() + (stream[0].stats.starttime - event_time)

    distances = [tr.stats.distance for tr in stream]
    min_distance = min(distances)
    max_distance = max(distances)
    distance_range = max_distance - min_distance
    stream_range = distance_range / 10.0

    # Normalize data and "shift to distance".
    stream.normalize()
    for tr in stream:
        tr.data *= stream_range
        tr.data += tr.stats.distance

    plt.figure(figsize=(20, 12))
    for _i, tr in enumerate(stream):
        plt.plot(times_array, tr.data, label="%s.%s" % (tr.stats.network,
                 tr.stats.station), color=colors[_i])
    plt.grid()
    plt.ylabel("Distance in degree to event")
    plt.xlabel("Time in seconds since event")
    plt.legend()

    dist_min, dist_max = plt.ylim()

    if plot_travel_times:

        distances = defaultdict(list)
        ttimes = defaultdict(list)

        for i in np.linspace(dist_min, dist_max, 1000):
            tts = getTravelTimes(i, event_depth_in_km, "ak135")
            for phase in tts:
                name = phase["phase_name"]
                distances[name].append(i)
                ttimes[name].append(phase["time"])

        for key in distances.iterkeys():
            min_distance = min(distances[key])
            max_distance = max(distances[key])
            min_tt_time = min(ttimes[key])
            max_tt_time = max(ttimes[key])

            if min_tt_time >= times_array[-1] or \
                    max_tt_time <= times_array[0] or \
                    (max_distance - min_distance) < 0.8 * (dist_max - dist_min):
                continue
            ttime = ttimes[key]
            dist = distances[key]
            if max(ttime) > times_array[0] + 0.9 * times_array.ptp():
                continue
            plt.scatter(ttime, dist, s=0.5, zorder=-10, color="black", alpha=0.8)
            plt.text(max(ttime) + 0.005 * times_array.ptp(),
                     dist_max - 0.02 * (dist_max - dist_min),
                     key)

    plt.ylim(dist_min, dist_max)
    plt.xlim(times_array[0], times_array[-1])

    plt.title(event.short_str())

    plt.show()


def align_phases(stream, event, inventory, phase_name, method="simple"):
    """
    Method either 'simple' or 'fft'. Simple will just shift the starttime of
    Trace, while 'fft' will do the shift in the frequency domain.
    """
    method = method.lower()
    if method not in ['simple', 'fft']:
        msg = "method must be 'simple' or 'fft'"
        raise ValueError(msg)

    stream = stream.copy()
    attach_coordinates_to_traces(stream, inventory, event)

    stream.traces = sorted(stream.traces, key=lambda x: x.stats.distance)[::-1]

    tr_1 = stream[-1]
    tt_1 = getTravelTimes(tr_1.stats.distance, event.origins[0].depth / 1000.0, "ak135")

    for tt in tt_1:
        if tt["phase_name"] != phase_name:
            continue
        tt_1 = tt["time"]
        break

    for tr in stream:
        tt = getTravelTimes(tr.stats.distance, event.origins[0].depth / 1000.0, "ak135")
        for t in tt:
            if t["phase_name"] != phase_name:
                continue
            tt = t["time"]
            break
        if method == "simple":
            tr.stats.starttime -= (tt - tt_1)
        else:
            AA.shifttrace_freq(Stream(traces=[tr]), [- ((tt - tt_1))])
    return stream


def vespagram(stream, ev, inv, method, frqlow, frqhigh, baz, scale, nthroot=4,
              filter=True, static3D=False, vel_corr=4.8, sl=(0.0, 10.0, 0.5),
              align=False, align_phase=['P', 'Pdiff'], plot_trace=True):

    starttime = max([tr.stats.starttime for tr in stream])
    endtime = min([tr.stats.endtime for tr in stream])
    stream.trim(starttime, endtime)

    org = ev.preferred_origin() or ev.origins[0]
    ev_lat = org.latitude
    ev_lon = org.longitude
    ev_depth = org.depth/1000.  # in km
    ev_otime = org.time

    sll, slm, sls = sl
    sll /= KM_PER_DEG
    slm /= KM_PER_DEG
    sls /= KM_PER_DEG
    center_lon = 0.
    center_lat = 0.
    center_elv = 0.
    seismo = stream
    seismo.attach_response(inv)
    seismo.merge()
    sz = Stream()
    i = 0
    for tr in seismo:
        for station in inv[0].stations:
            if tr.stats.station == station.code:
                tr.stats.coordinates = \
                    AttribDict({'latitude': station.latitude,
                                'longitude': station.longitude,
                                'elevation': station.elevation})
                center_lon += station.longitude
                center_lat += station.latitude
                center_elv += station.elevation
                i += 1
        sz.append(tr)

    center_lon /= float(i)
    center_lat /= float(i)
    center_elv /= float(i)

    starttime = max([tr.stats.starttime for tr in stream])
    stt = starttime
    endtime = min([tr.stats.endtime for tr in stream])
    e = endtime
    stream.trim(starttime, endtime)

    #nut = 0
    max_amp = 0.
    sz.trim(stt, e)
    sz.detrend('simple')

    print sz
    fl, fh = frqlow, frqhigh
    if filter:
        sz.filter('bandpass', freqmin=fl, freqmax=fh, zerophase=True)

    if align:
        deg = []
        shift = []
        res = gps2DistAzimuth(center_lat, center_lon, ev_lat, ev_lon)
        deg.append(kilometer2degrees(res[0]/1000.))
        tt = getTravelTimes(deg[0], ev_depth, model='ak135')
        for item in tt:
            phase = item['phase_name']
            if phase in align_phase:
                try:
                    travel = item['time']
                    travel = ev_otime.timestamp + travel
                    dtime = travel - stt.timestamp
                    shift.append(dtime)
                except:
                    break
        for i, tr in enumerate(sz):
            res = gps2DistAzimuth(tr.stats.coordinates['latitude'],
                                  tr.stats.coordinates['longitude'],
                                  ev_lat, ev_lon)
            deg.append(kilometer2degrees(res[0]/1000.))
            tt = getTravelTimes(deg[i+1], ev_depth, model='ak135')
            for item in tt:
                phase = item['phase_name']
                if phase in align_phase:
                    try:
                        travel = item['time']
                        travel = ev_otime.timestamp + travel
                        dtime = travel - stt.timestamp
                        shift.append(dtime)
                    except:
                        break
        shift = np.asarray(shift)
        shift -= shift[0]
        AA.shifttrace_freq(sz, -shift)

    baz += 180.
    nbeam = int((slm - sll)/sls + 0.5) + 1
    kwargs = dict(
        # slowness grid: X min, X max, Y min, Y max, Slow Step
        sll=sll, slm=slm, sls=sls, baz=baz, stime=stt, method=method,
        nthroot=nthroot, etime=e, correct_3dplane=False, static_3D=static3D,
        vel_cor=vel_corr)

    start = UTCDateTime()
    slow, beams, max_beam, beam_max = AA.vespagram_baz(sz, **kwargs)
    print "Total time in routine: %f\n" % (UTCDateTime() - start)

    df = sz[0].stats.sampling_rate
    # Plot the seismograms
    npts = len(beams[0])
    print npts
    T = np.arange(0, npts/df, 1/df)
    sll *= KM_PER_DEG
    slm *= KM_PER_DEG
    sls *= KM_PER_DEG
    slow = np.arange(sll, slm, sls)
    max_amp = np.max(beams[:, :])
    #min_amp = np.min(beams[:, :])
    scale *= sls

    fig = plt.figure()

    if plot_trace:
        ax1 = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        for i in xrange(nbeam):
            if i == max_beam:
                ax1.plot(T, sll + scale*beams[i]/max_amp + i*sls, 'r',
                         zorder=1)
            else:
                ax1.plot(T, sll + scale*beams[i]/max_amp + i*sls, 'k',
                         zorder=-1)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('slowness [s/deg]')
        ax1.set_xlim(T[0], T[-1])
        data_minmax = ax1.yaxis.get_data_interval()
        minmax = [min(slow[0], data_minmax[0]), max(slow[-1], data_minmax[1])]
        ax1.set_ylim(*minmax)
    #####
    else:
        #step = (max_amp - min_amp)/100.
        #level = np.arange(min_amp, max_amp, step)
        #beams = beams.transpose()
        #cmap = cm.hot_r
        cmap = cm.rainbow

        ax1 = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        #ax1.contour(slow,T,beams,level)
        #extent = (slow[0], slow[-1], \
        #               T[0], T[-1])
        extent = (T[0], T[-1], slow[0], slow[-1])

        ax1.set_ylabel('slowness [s/deg]')
        ax1.set_xlabel('T [s]')
        beams = np.flipud(beams)
        ax1.imshow(beams, cmap=cmap, interpolation="nearest",
                   extent=extent, aspect='auto')

    ####
    result = "BAZ: %.2f Time %s" % (baz-180., stt)
    ax1.set_title(result)

    plt.show()
    return slow, beams, max_beam, beam_max
