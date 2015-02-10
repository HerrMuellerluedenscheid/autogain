import glob
import os
from collections import defaultdict
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

pjoin = os.path.join
logger = logging.getLogger()
km = 1000.

class PhasePie():
    def __init__(self, which='first'):
        self.which = which
        self.model = cake.load_model('prem-no-ocean.m')
        self.arrivals = defaultdict(dict)
    
    def t(self, phase_ids, dist_z):
        dist, z = dist_z
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
        
#class Window():
#    def __init__(self, phase_pie, static_length=20.):
#        self.phase_pie = phase_pie
#        self.static_length = static_length
#
#    def get_tmin_tmax(self, t):
#        return t-0.5*seld.static_length, t+0.5*self.static_length

class StaticWindow():
    def __init__(self, tmin, static_length):
        self.tmin = tmin
        self.static_length = static_length

    def get_tmin_tmax(self):
        return self.tmin, self.tmin+self.static_length

class StaticLengthWindow():
    def __init__(self, static_length):
        self.static_length = static_length

    def t(self):
        return self.static_length

class ChopperConfig():
    def __init__(init, phaseids, window, stations):
        self.phaseids = phaseids
        self.window = window
        self.stations = stations

    def station_of_trace(self, tr):
        for s in self.stations:
            if util.match_nslc('%s.%s.%s.*'%s.nsl(), tr.nslc_id):
                return s

class Chopper():
    def __self__(init, config):
        self.config = config

    def chop(self, tr, event):
        for tr in traces:
            station = self.config.station_of_trace(tr)
            dist = distance_accurate50m(station, event)
            self.config.window.get_tmin_tmax(self, tr)
            tr.chop()

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


class AutoGain():
    def __init__(self, reference_nsl, data_pile, stations, event_selector, component=None):
        self.reference_nsl = reference_nsl
        #self.references = filter(lambda x: util.match_nslc(guess_nsl_template(
        #                                                   self.reference_nsl), x.nslc_id) , self.traces)

        self.component = 'Z'
        self.data_pile = data_pile
        self.stations = stations
        self.candidates = event_selector.get_events(data_pile, stations)
        self.component = component
        self.phaser = PhasePie()
    
        self.minmax = {}
        self.scaling_factors = {}

    def process(self):
        for event in self.candidates:
            self.process_event(event)

    def process_event(self, event):
        for s in self.stations:
            self.process_station(s, event)
    
    def process_station(self, event, station):
        dist = distance_accurate50m(event, station)
        arrival = self.phaser.t('first(P|p)', (dist, event.depth*km))
        selector = lambda tr: util.match_nslc('%s.%s'%(station.nsl_string(), self.component),
                                              '.'.join(tr.nslc_id))

        tr = self.data_pile.chopper(tmin=arrival, 
                                    tmax=arrival + window.t(),
                                    trace_selector=selector)

        tr.snuffle()

    def set_phaser(self, phaser):
        self.phaser = phaser

    def set_window(self, window):
        self.window = window

    def set_scaling_factors(self, fband, window=None):
        for tr in self.references:
            tr = tr.copy()
            tr.bandpass(**fband )
            if window:
                tr.chop(window.get_tmin_tmax())
            self.minmax[tr.channel] = trace.minmax([tr]).values()
            
        for tr in self.traces:
            tr = tr.copy()
            tr.bandpass(**fband )
            if window:
                tr.chop(window.get_tmin_tmax())
            mm = trace.minmax([tr])
            self.scaling_factors[tr.nslc_id] = num.max(num.abs(self.minmax[tr.channel]))/num.max(num.abs(mm.values()))

    def get_scaled(self):
        if self.scaling_factors == {}:
            self.set_scaling_factors()
        
        scaled = []
        for tr in self.traces:
            tr = tr.copy()
            tr.ydata *= self.scaling_factors[tr.nslc_id]
            scaled.append(tr) 
        
        return scaled

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
                         default_store_id='globalttt').get_store()
    #phases = PhasePie()

    methods = ['minmax', 'match']

    #filenames = glob.glob('data/*.mseed')
    #filenames = glob.glob('/data/webnet/waveform_R/2008/*.mseed')
    datapath = '/data/webnet/waveform_R/2008'
    #filenames = os.listdir(datapath)

    #logger.info(filenames)
    stations = model.load_stations('data/stations.pf')
    reference_id ='KRC'
    references = {}
    data_pile = pile.make_pile(datapath)
    
    #for fn in filenames:
    #    trs = io.load(pjoin(datapath, fn))
    #    for t in trs:
    #        if is_reference(t, reference_id):
    #            references[t.channel] = t
    #        else:
    #            traces.append(t)

    #traces = filter(lambda tr: tr.channel[-1]=='Z', traces)
    
    fband = {'order':4, 'corner_hp':1.0, 'corner_lp':4.}
    #window = Window(phases)
    #ChopperConfig('P', stations)
    #window = StaticWindow(tmin=util.str_to_time('2013-03-11 15:02:41.000'), 
    #                      static_length=30.)
    window = StaticLengthWindow(static_length=30.)
    #ag = AutoGain(reference_id, traces)
    event_selector = EventSelector(distmin=1000*km,
                                   distmax=20000*km,
                                   depthmin=1*km,
                                   depthmax=600*km,
                                   magmin=1.5)

    ag = AutoGain(reference_id, data_pile, stations=stations,
                  event_selector=event_selector, component='Z')
    ag.set_phaser(phases)
    ag.set_window(window)
    ag.process()
    #ag.set_scaling_factors(fband, window)
    scaled_traces = ag.get_scaled()
    trace.snuffle(scaled_traces, events=candidates)
