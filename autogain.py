import glob
from collections import defaultdict
from pyrocko import io
from pyrocko import cake
from pyrocko import util
from pyrocko.orthodrome import distance_accurate50m
import numpy as num


km = 1000.

class PhasePie():
    def __init__(self):
        self.model = cake.load_model('prem-no-ocean.m')
        self.arrivals = defaultdict(dict)
    
    def get_arrival(dist, z, phase_id, which='first'):
        if phase_id in self.arrivals.keys() and (dist, z) in self.arrivals[phase_id].keys():
            return phase_selector(self.arrivals[phase_id][(dist,z)], which)
        else:
            return self.add_arrival(dist, z, phase_id, which)

    def add_arrival(dist, z, phase_id, which):
        Phase = cake.PhaseDef(phase_id)
        arrivals = model.arrivals(dist, phases=Phase, zstart=z)
        self.arrivals[phase_id][(dist, z)] = arrivals
        return phase_selector(arrivals, which)
    
    def phase_selector(self, _list, which):
        if which=='first':
            return min(_list, key=lambda x: x.tmin)
        if which=='last':
            return max(_list, key=lambda x: x.tmin)
        
class Window():
    def __init__(self, phase_pie, static_length=20.):
        self.phase_pie = phase_pie
        self.static_length = static_length

    def get_tmin_tmax(self, t):
        return t-0.5*seld.static_length, t+0.5*self.static_length


class ChopperConfig():
    def __init__(init, phaseids, window, stations):
        self.phaseids = phaseids
        self.window = window
        self.stations = stations

    def station_of_trace(self, tr):
        for s in self.stations:
            if util.match_nslc('%s.%s.%s.*'%s.nsl(), tr.nslc_id)
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


class AutoGain():
    def __init__(self, reference):
        self.reference = reference


def is_reference(tr):
    reference_station = 'KAC'
    return reference_station in tr.nslc_id

if '__name__' == '__main__':

    fbands = []
    fbands.apppend([1.0, 2.0])
    fbands.apppend([2.0, 6.0])
    fbands.apppend([4.0, 10.])

    methods = ['minmax', 'match']

    filenames = glob.glob('data/*.mseed')
    stations = model.load_stations('data/stations.pf')
    traces = []
    references = {}
    for fn in filenames:
        trs = io.load(fn)
        for t in trs:
            if is_reference(t):
                references[t.channel] = t
            else:
                traces.append(t)
    phases = PhasePie()
    window = Window(phases)
    ChopperConfig('P', stations)
    if method=='minmax':
        for fband in fbands:
            for tr in traces:
                
