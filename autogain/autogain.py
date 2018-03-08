#!/use/bin/env python

import sys
import glob
import os
from collections import defaultdict, OrderedDict
from pyrocko import io
from pyrocko import cake
from pyrocko import util
from pyrocko import trace
from pyrocko import model
from pyrocko import catalog as pyrocko_catalog
from pyrocko import pile
from pyrocko import gui_util
from pyrocko.orthodrome import distance_accurate50m
from pyrocko.gf import LocalEngine
import logging
import numpy as num
# import matplotlib
# matplotlib.use('GTK')
import matplotlib.pyplot as plt

from autogain.util_optic import Optics

from autogain.gains import Gains

pjoin = os.path.join
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
km = 1000.


def printl(l):
    for i in l:
        print(i)


class PhasePie():
    '''Calculates and caches phase arrivals'''
    def __init__(self, mod='prem-no-ocean.f'):
        ''':param mod: Name of the model to be used.
           (see earthmodels in pyrocko)'''
        self.model = cake.load_model(mod)
        self.arrivals = defaultdict(dict)
        self.which = None

    def t(self, phase_selection, z_dist):
        ''':param phase_selection: phase names speparated by vertical bars
        :param z_dist: tuple with (depth, distance)
        '''
        if 'first' in phase_selection:
            self.which = 'first'

        if 'last' in phase_selection:
            self.which = 'last'

        if self.which:
            phase_selection = self.strip(phase_selection)

        z, dist = z_dist
        if (phase_selection, dist, z) in self.arrivals.keys():
            return self.arrivals[(phase_selection, dist, z)]

        phases = [cake.PhaseDef(pid) for pid in phase_selection.split('|')]
        arrivals = self.model.arrivals(distances=[dist*cake.m2d],
                                       phases=phases,
                                       zstart=z)
        if arrivals == []:
            logger.info('none of defined phases at d=%s, z=%s' % (dist, z))
            return
        else:
            want = self.phase_selector(arrivals)
            self.arrivals[(phase_selection, dist, z)] = want
            return want

    def phase_selector(self, _list):
        if self.which == 'first':
            return min(_list, key=lambda x: x.t).t
        if self.which == 'last':
            return max(_list, key=lambda x: x.t).t

    def strip(self, ps):
        ps = ps.replace(self.which, '')
        ps = ps.rstrip(')')
        ps = ps.lstrip('(')
        return ps

    def __str__(self):
        s = 'PhasePie'
        for k1, kv in self.arrivals.items():
            s += '%s %s \n' % (str(k1), kv)
        return s


class StaticWindow():
    def __init__(self, tmin, static_length):
        '''Rigid, static window starting at *tmin*
           with a duration of *static_length*.'''
        self.tmin = tmin
        self.static_length = static_length

    def get_tmin_tmax(self):
        return self.tmin, self.tmin+self.static_length


class StaticLengthWindow():
    def __init__(self, static_length, phase_position):
        '''
        phase_position: 0-> start ... 0.5 -> center ... 1.0 -> end'''
        self.phase_position = phase_position
        self.static_length = static_length

    def t(self):
        return self.static_length*self.phase_position,\
               self.static_length*1.0-self.phase_position


def guess_nsl_template(code):
    if len(code) == 1 or isinstance(code, str):
        return '*.%s.*.*' % (code)
    elif len(code) == 2:
        return '*.%s.%s.*' % (code)
    elif len(code) == 3:
        return '%s.%s.%s.*' % (code)


class EventSelector():
    '''subclasses need to implement *get_events()*'''
    def __init__(self, *args, **kwargs):
        pass

    def save_events(self, fn='candidates.pf'):
        if not self.events:
            raise Exception('Need to run processing first.')
        else:
            events = self.events

        event.Event.dump_catalog(events, fn) # ???
        # not sure, but pyrocko.model.Event moved to pyrocko.model.event.Event


class EventSelectorCatalog(EventSelector):
    def __init__(self, magmin=None, distmin=None, distmax=None, depthmin=None,
                 depthmax=None, catalog=None):
        '''
        Automatically download events from *catalog*.

        :param magmin: minimum magnitude
        :param distmin: minimum distance
        :param distamx: maximum distance
        :param depthmin: minimum depth
        :param depthmax: maximum depth
        :param catalog: instance of :py:class:`pyrocko.catalog.Earthquake`
            subclass. Default is Geofon catalog.
        '''
        EventSelector.__init__(self)
        self.catalog = catalog or pyrocko_catalog.Geofon()
        self.magmin = magmin
        self.distmin = distmin
        self.distmax = distmax
        self.depthmin = depthmin
        self.depthmax = depthmax

        self.events = None

    def get_events(self, data_pile=None, stations=None):
        event_names = self.catalog.get_event_names(time_range=(data_pile.tmin,
                                                               data_pile.tmax),
                                                   magmin=self.magmin)

        events = [self.catalog.get_event(en) for en in event_names]
        events = filter(lambda x: min_dist(x, stations) >=
                        self.distmin, events)
        if self.distmax:
            events = filter(lambda x: max_dist(x, stations) <=
                            self.distmax, events)
        if self.depthmin:
            events = filter(lambda x: x.depth >= self.depthmin, events)
        if self.depthmax:
            events = filter(lambda x: x.depth <= self.depthmax, events)
        self.events = events
        return events


class EventCollection(EventSelector):
    '''A container for events.

    Instantiate using one of the two following kwargs:
    :param events: list of :py:class:`pyrocko.model.Event` instances
    :param markers: list of :py:class:`pyrocko.gui_util.EventMarker` instances
    '''
    def __init__(self, *args, **kwargs):
        if 'events' in kwargs.keys():
            self.events = kwargs.pop('events')
        elif 'markers' in kwargs.keys():
            markers = kwargs.pop('markers')
            self.events = [m.get_event() for m in markers]
        else:
            raise Exception('requires either "events" or "markers" kwargs.')
        EventSelector(self, args, kwargs)

    def get_events(self, *args, **kwargs):
        return self.events


class Section():
    ''' Related to one event. All traces scale relative to average mean abs
    max'''
    def __init__(self, event, stations):
        self.stations = stations
        self.event = event
        self.traces = []
        self.reference_scale = None

        self.max_tr = {}
        self.relative_scalings = {}
        self.finished = False

    def finish(self, reference_nsl, fband, taper):
        for tr in self.traces:
            tr_backup = tr.copy()
            tr_backup.set_location('B')
            tr.ydata -= num.int32(tr.get_ydata().mean())  # otherwise dtype error!
            tr.highpass(fband['order'], fband['corner_hp'])
            tr.taper(taper, chop=False)
            tr.lowpass(fband['order'], fband['corner_lp'])
            self.max_tr[tr.nslc_id] = num.max(num.abs(tr.get_ydata()))

        if reference_nsl is not True:
            reference_nslc = list(filter(
                lambda x: util.match_nslc(guess_nsl_template(reference_nsl), x),
                                          self.max_tr.keys()))
            print(reference_nslc)
            self.____reference_nslc = reference_nslc
            if not len(reference_nslc) == 1:
                logger.info('no reference trace available. ' +
                            'remains unfinished: %s' % self.event)
                self.reference_scale = 1.
                # self.set_relative_scalings()
                self.finished = False
            else:
                self.reference_scale = self.max_tr[reference_nslc[0]]
                self.set_relative_scalings()
                self.finished = True
        else:
            self.reference_scale = 1.
            self.set_relative_scalings()
            self.finished = True

    def set_relative_scalings(self):
        for nslc_id, maxs in self.max_tr.items():
            self.relative_scalings[nslc_id] = self.reference_scale/maxs

    def extend(self, tr):
        self.traces.extend(tr)

    def get_gained_traces(self):
        gained = []
        for tr in self.traces:
            tr = tr.copy()
            tr.ydata *= self.relative_scalings[tr.nslc_id]
            tr.set_location('G')
            gained.append(tr)
        return gained

    def get_ungained_traces(self):
        return self.traces

    def snuffle(self):
        trace.snuffle(self.traces, events=[self.event], stations=self.stations)

    def iter_scalings(self):
        for nslc_id, scaling in self.relative_scalings.items():
            yield (nslc_id, scaling)


class AutoGain():
    def __init__(self, data_pile, stations, event_selector,
                 component='Z', reference_nsl=None, scale_one=False,
                 phase_selection='first(p|P)'):
        ''':param phase_selection:
            follows the logic of fomosto's Store phase definitions'''

        self.reference_nsl = reference_nsl or scale_one
        #self.references = filter(lambda x:
        #                          util.match_nslc(guess_nsl_template(
        #                         self.reference_nsl), x.nslc_id) , self.traces)

        self.component = component
        self.data_pile = data_pile
        self.stations = stations
        self.candidates = event_selector.get_events(data_pile, stations)
        self.phaser = None
        self.phase_selection = phase_selection
        self.all_nslc_ids = set()
        self.minmax = {}
        self.scaling_factors = {}
        self.sections = []
        self.results = None
        self._mean = None

    def process(self, fband, taper):
        for event in self.candidates:
            section = Section(event, self.stations)
            skipped = 0
            unskipped = 0
            for i_s, s in enumerate(self.stations):
                dist = distance_accurate50m(event, s)
                arrival = self.phaser.t(self.phase_selection,
                                        (event.depth, dist))
                if arrival is None:
                    skipped += 1
                    logger.debug('skipping event %s at stations %s. ' +
                                 'Reason no phase arrival'
                                 % (event, s))
                    continue
                else:
                    unskipped += 1
                selector = lambda tr: util.match_nslc('%s.*%s' %
                                                      (s.nsl_string(),
                                                       self.component),
                                                      tr.nslc_id)
                window_min, window_max = self.window.t()
                tr = self.data_pile.chopper(tmin=event.time+arrival -
                                            window_min,
                                            tmax=event.time+arrival +
                                            window_max,
                                            trace_selector=selector)
                try:
                    _tr = next(tr)    # does not work without try and except?!
                except StopIteration:
                    continue
                try:
                    assert len(_tr) in (0, 1)
                    self.all_nslc_ids.add(_tr[0].nslc_id)
                    section.extend(_tr)
                except IndexError:
                    continue
                except AssertionError:
                    continue
                try:
                    next(tr)
                    raise Exception('More than one trace returned')
                except StopIteration:
                    continue

            logger.debug('skipped %s/%s' % (skipped, unskipped))

            section.finish(self.reference_nsl, fband, taper)
            self.sections.append(section)

    def get_sections(self):
        return self.sections

    def congreate(self):
        indx = dict(zip(self.all_nslc_ids, num.arange(0, len(self.all_nslc_ids))))
        self.results = num.empty((len(self.sections), len(self.all_nslc_ids)))
        self.results[:] = num.nan
        for i_sec, section in enumerate(self.sections):
            for nslc_id, scaling in section.iter_scalings():
                self.results[i_sec, indx[nslc_id]] = scaling

    def get_results(self):
        return self.sections

    def set_phaser(self, phaser):
        self.phaser = phaser

    def set_window(self, window):
        self.window = window

    @property
    def mean(self):
        if self.results is None:
            self.congreate()
        if self._mean is None:
            self._mean = dict(zip(map(lambda x: '.'.join(x),
                                      self.all_nslc_ids),
                                  num.nanmean(self.results, axis=0)))
        return self._mean

    def save_mean(self, fn):
        g = Gains()
        #tmp = {}
        g.trace_gains = self.mean
        #g.gains = zip(ids, mean_section)
        #for i in xrange(len(ids)):
        #    g.trace_gains[ids[i]] = mean_section[i]
        #print(g)
        g.regularize()
        g.validate()
        g.dump(filename=fn)


def min_dist(event, stations):
    dists = [distance_accurate50m(event, s) for s in stations]
    return min(dists)


def max_dist(event, stations):
    dists = [distance_accurate50m(event, s) for s in stations]
    return max(dists)


def is_reference(tr, reference_station):
    return reference_station in tr.nslc_id


class Checker():
    def __init__(self):
        pass

    def check_snr(self):
        pass


if __name__ == '__main__':
    '''usage example'''
    km = 1000.
    #fbands = []
    #fbands.apppend([1.0, 2.0])
    #fbands.apppend([2.0, 6.0])
    #fbands.apppend([4.0, 10.])

    phases = LocalEngine(store_superdirs=['/data/stores'],
                         default_store_id='globalttt').get_store()

    #filenames = glob.glob('data/*.mseed')
    #filenames = glob.glob('/data/webnet/waveform_R/2008/*.mseed')
    #datapath = '/data/webnet/mseed/2008'
    #datapath = '/data/webnet/waveform_R/2008'
    #datapath = '/data/share/Res_all_NKC'
    datapath = '/media/usb0/Res_all_NKC_taper'
    #datapath = '/media/usb0/restituted_pyrocko'
    stations = model.load_stations('../data/stations.pf')
    reference_id = 'NKC'
    references = {}
    data_pile = pile.make_pile(datapath, selector='rest_*')

    fband = {'order': 4, 'corner_hp': 1.0, 'corner_lp': 4.}
    window = StaticLengthWindow(static_length=30.,
                                phase_position=0.5)

    taper = trace.CosFader(xfrac=0.25)

    #event_selector = EventSelector(distmin=1000*km,
    #                               distmax=20000*km,
    #                               depthmin=2*km,
    #                               depthmax=600*km,
    #                               magmin=4.9)

    candidate_fn = '../candidates2013.pf'
    candidates = [m.get_event() for m in gui_util.Marker.load_markers(candidate_fn)]
    event_selector = EventCollection(events=candidates)

    ag = AutoGain(data_pile,
                  stations=stations,
                  reference_nsl=reference_id,
                  event_selector=event_selector,
                  component='Z')

    ag.set_phaser(phases)
    ag.set_window(window)
    ag.process(fband, taper)
    ag.save_mean(candidate_fn.replace('candidates', 'gains'))

    optics = Optics(ag)
    optics.plot()
    plt.show()
    for s in ag.get_sections():
        scaled_traces = s.get_gained_traces()
        unscaled_traces = s.get_ungained_traces()

        scaled_traces.extend(unscaled_traces)
        trace.snuffle(scaled_traces, events=candidates)
