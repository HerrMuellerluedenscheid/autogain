import glob
from collections import defaultdict
from pyrocko import io
from pyrocko import cake
from pyrocko import util
from pyrocko import trace
from pyrocko import model
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

class AutoGain():
    def __init__(self, reference_nsl, traces):
        self.reference_nsl = reference_nsl
        self.traces = traces
        match_this = guess_nsl_template(self.reference_nsl)
        self.references = filter(lambda x: util.match_nslc(match_this, x.nslc_id), self.traces)
        self.minmax = {}
        self.scaling_factors = {}

    def set_scaling_factors(self, fband, window=None):
        for tr in self.references:
            tr = tr.copy()
            tr.bandpass(**fband )
            if window:
                tr.chop(*window.get_tmin_tmax())
            #self.minmax[tr.channel] = trace.minmax([tr]).values()
            self.minmax[tr.channel] = num.max(num.abs(tr.ydata))
            
        for tr in self.traces:
            tr = tr.copy()
            tr.bandpass(**fband )
            if window:
                tr.chop(*window.get_tmin_tmax())
            #mm = trace.minmax([tr])
            tr.ydata -= tr.ydata.mean()
            mm = num.max(num.abs(tr.ydata))
            self.scaling_factors[tr.nslc_id] = self.minmax[tr.channel]/mm

    def get_scaled(self):
        if self.scaling_factors == {}:
            raise Exception('run set_scaling_factors first')
            
        scaled = []
        for tr in self.traces:
            tr = tr.copy()
            tr.ydata *= self.scaling_factors[tr.nslc_id]
            scaled.append(tr) 
        
        self.scaled = scaled
        return scaled
        
    def save_traces(self):
        io.save(self.scaled, "autogained.mseed")

    def __str__(self):
        s = ''
        for k,v in self.scaling_factors.items():
            s+= "%s   %1.3f\n" %(k, v)
        return s
     

def candidates():
    
    return candidates

if __name__ == '__main__':

    #fbands = []
    #fbands.apppend([1.0, 2.0])
    #fbands.apppend([2.0, 6.0])
    #fbands.apppend([4.0, 10.])
    methods = ['minmax', 'match']

    #filenames = glob.glob('data/*.mseed')
    filenames = glob.glob('/data/webnet/waveform_R/2008/*')
    stations = model.load_stations('data/stations.pf')
    traces = []
    for fn in filenames:
        for tr in io.iload(fn):
            traces.append(tr)
    traces = filter(lambda tr: tr.channel[-1]=='Z', traces)
    fband = {'order':4, 'corner_hp':1.0, 'corner_lp':4.}
    #phases = PhasePie()
    #window = Window(phases)
    #ChopperConfig('P', stations)
    window = StaticWindow(tmin=util.str_to_time('2013-03-11 15:02:41.000'), 
                          static_length=30.)
    ag = AutoGain('NKC', traces)
    ag.set_scaling_factors(fband, window)
    scaled_traces = ag.get_scaled()
    ag.save_traces()
    print ag
    trace.snuffle(scaled_traces)
