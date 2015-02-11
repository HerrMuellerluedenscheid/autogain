import sys
import glob
import os
from collections import defaultdict, OrderedDict
from pyrocko import io
from pyrocko import cake
from pyrocko import util
from pyrocko import trace
from pyrocko import model
from pyrocko import catalog
from pyrocko import pile
from pyrocko.orthodrome import distance_accurate50m
from pyrocko.gf import LocalEngine
import logging
import numpy as num
import matplotlib
matplotlib.use('GTK')
import matplotlib.cm as cm
import matplotlib.pyplot as plt

pjoin = os.path.join
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
km = 1000.

def printl(l):
    for i in l:
        print i

class PhasePie():
    def __init__(self, which='first'):
        self.which = which
        self.model = cake.load_model('prem-no-ocean.m')
        self.arrivals = defaultdict(dict)
    
    def t(self, phase_ids, z_dist):
        z, dist = z_dist 
        for phase_id in phase_ids.split('|'):
            if phase_id in self.arrivals.keys() and (dist, z) in self.arrivals[phase_id].keys():
                return phase_selector(self.arrivals[phase_id][(dist,z)], self.which)
            else:
                self.add_arrival(dist, z, phase_id)
                continue

    def add_arrival(self, dist, z, phase_id):
        Phase = cake.PhaseDef(phase_id)
        arrivals = self.model.arrivals(dist, phases=Phase, zstart=z)
        if arrivals==[]:
            logger.debug('no phase %s at d=%s, z=%s' %(phase_id, dist, z))
            return 
        else:
            self.arrivals[phase_id][(dist, z)] = arrivals
            return self.phase_selector(arrivals)
    
    def phase_selector(self, _list):
        if self.which=='first':
            return min(_list, key=lambda x: x.tmin)
        if self.which=='last':
            return max(_list, key=lambda x: x.tmin)
        
class StaticWindow():
    def __init__(self, tmin, static_length):
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
        return self.static_length*self.phase_position, self.static_length*1.0-self.phase_position

def guess_nsl_template(code):
    if len(code)==1 or isinstance(code, str):
        return '*.%s.*.*'%(code)
    elif len(code)==2:
        return '*.%s.%s.*'%(code)
    elif len(code)==3:
        return '%s.%s.%s.*'%(code)

class EventSelector():
    def __init__(self, magmin, distmin, distmax, depthmin=None, depthmax=None):
        self.magmin = magmin
        self.distmin = distmin
        self.distmax = distmax
        self.depthmin = depthmin
        self.depthmax = depthmax

    def get_events(self, data_pile, stations):
        cat = catalog.Geofon()
        event_names = cat.get_event_names(time_range=(data_pile.tmin, 
                                                      data_pile.tmax),
                                          magmin=self.magmin)
        
        events = [cat.get_event(en) for en in event_names]
        events = filter(lambda x: min_dist(x, stations)>=self.distmin, events)
        if self.distmax:
            events = filter(lambda x: max_dist(x, stations)<=self.distmax, events)
        if self.depthmin:
            events = filter(lambda x: x.depth>=self.depthmin, events)
        if self.depthmax:
            events = filter(lambda x: x.depth<=self.depthmax, events)
        return events

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
            tr.ydata -= tr.get_ydata().mean()
            tr.taper(taper)
            tr.bandpass(**fband)
            self.max_tr[tr.nslc_id] = num.max(num.abs(tr.get_ydata()))
        
        reference_nslc = filter(
            lambda x: util.match_nslc(guess_nsl_template(reference_nsl), x), self.max_tr.keys())
        self.____reference_nslc = reference_nslc
        if not len(reference_nslc)==1:
            logger.info('no reference trace available. remains unfinished: %s' % self.event)
            self.reference_scale = 1.
            self.set_relative_scalings()
        else:
            self.reference_scale = self.max_tr[reference_nslc[0]]
            self.set_relative_scalings()
            self.finished = True

    def set_relative_scalings(self):
        for nslc_id, maxs in self.max_tr.iteritems():
            self.relative_scalings[nslc_id] = self.reference_scale/maxs

    def extend(self, tr):
        self.traces.extend(tr)

    def get_gained_traces(self):
        gained = []
        for tr in self.traces:
            tr = tr.copy()
            tr.ydata *= self.relative_scalings[tr.nslc_id]
            gained.append(tr)
        return gained

    def snuffle(self):
        trace.snuffle(self.traces, events=[self.event], stations=self.stations)

    def iter_scalings(self):
        for nslc_id, scaling in self.relative_scalings.iteritems():
            yield (nslc_id, scaling)

class AutoGain():
    def __init__(self, reference_nsl, data_pile, stations, event_selector, component='Z'):
        self.reference_nsl = reference_nsl
        #self.references = filter(lambda x: util.match_nslc(guess_nsl_template(
        #                                                   self.reference_nsl), x.nslc_id) , self.traces)

        self.component = component
        self.data_pile = data_pile
        self.stations = stations
        self.candidates = event_selector.get_events(data_pile, stations)
        self.phaser = PhasePie()
        self.all_nslc_ids = set()
        self.minmax = {}
        self.scaling_factors = {}
        self.sections = []
        self.results = None

    def process(self, fband, taper):
        for event in self.candidates:
            section = Section(event, self.stations)
            skipped = 0
            unskipped = 0
            for i_s, s in enumerate(self.stations):
                dist = distance_accurate50m(event, s)
                arrival = self.phaser.t('begin', (event.depth, dist))
                if arrival==None:
                    skipped +=1
                    logger.debug('skipping event %s at stations %s. Reason no phase arrival'
                                % (event, s))
                    continue
                else:
                    unskipped +=1
                selector = lambda tr: util.match_nslc('%s.*%s'%(s.nsl_string(),
                                                                self.component),
                                                      tr.nslc_id)
                
                
                window_min, window_max = window.t()
                tr = self.data_pile.chopper(tmin=event.time+arrival - window_min, 
                                            tmax=event.time+arrival + window_max,
                                            trace_selector=selector)
                _tr = tr.next()
                try:
                    assert len(_tr) in (0, 1)
                    self.all_nslc_ids.add(_tr[0].nslc_id)
                    section.extend(_tr)
                except IndexError:
                    continue
                try:
                    tr.next()
                    raise Exception('More than one trace returned')
                except StopIteration:
                    continue

            logger.debug('skipped %s/%s'%(skipped, unskipped))

            section.finish(self.reference_nsl, fband, taper)
            self.sections.append(section)


    def congreate(self):
        indx = dict(zip(self.all_nslc_ids, num.arange(0,len(self.all_nslc_ids))))
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
        if self.results==None:
            self.congreate()

        return (self.all_nslc_ids, num.nanmean(self.results, axis=0))

class Optics():
    def __init__(self, autogain):
        self.autogain = autogain
        
    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        event_count = 0
        for section in sorted(self.autogain.get_results(), key=lambda x:
                              x.event.magnitude):
            event_count += 1
            for nslc_id in self.autogain.all_nslc_ids:
                try:
                    scale = section.relative_scalings[nslc_id]
                except KeyError:
                    continue
                ax.plot(event_count, 
                        scale, 
                        'o',
                        c=self.color(nslc_id), 
                        ms=2+1.5**section.event.magnitude,
                        label=nslc_id)

        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        self.add_mean_lines(ax)

    def add_mean_lines(self, ax):
        ids, _means = self.autogain.mean
        for i in range(len(_means)):
            m = _means[i]
            nslc_id = ids.pop()
            ax.axhline(y=m, c=self.color(nslc_id), label=nslc_id)

    def set_color(self):
        want = self.autogain.all_nslc_ids
        self._color_dict = dict(zip(want, num.linspace(0, 1, len(want))))
    
    def color(self, nslc_id):
        try:
            return cm.gist_rainbow(self._color_dict[nslc_id])
        except AttributeError: 
            self.set_color()
            return cm.gist_rainbow(self._color_dict[nslc_id])

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
    km = 1000.
    #fbands = []
    #fbands.apppend([1.0, 2.0])
    #fbands.apppend([2.0, 6.0])
    #fbands.apppend([4.0, 10.])

    phases = LocalEngine(store_superdirs=['/data/stores'],
                         default_store_id='global_2s').get_store()
    
    methods = ['minmax', 'match']

    #filenames = glob.glob('data/*.mseed')
    #filenames = glob.glob('/data/webnet/waveform_R/2008/*.mseed')
    datapath = '/data/webnet/mseed/2008'
    stations = model.load_stations('data/stations.pf')
    reference_id ='KRC'
    references = {}
    data_pile = pile.make_pile(datapath)
    
    
    fband = {'order':4, 'corner_hp':1.0, 'corner_lp':4.}
    window = StaticLengthWindow(static_length=20., 
                                phase_position=0.4)
    
    taper = trace.CosFader(xfrac=0.15)
    
    event_selector = EventSelector(distmin=1000*km,
                                   distmax=20000*km,
                                   depthmin=1*km,
                                   depthmax=600*km,
                                   magmin=4.9)

    ag = AutoGain(reference_id, data_pile, stations=stations,
                  event_selector=event_selector, component='Z')
    ag.set_phaser(phases)
    ag.set_window(window)
    ag.process(fband, taper)
    ag.congreate()
    optics = Optics(ag)
    optics.plot()
    plt.show()
    #trace.snuffle(scaled_traces, events=candidates)
