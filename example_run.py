#!/use/bin/env python
import logging
import numpy as num

from pyrocko import trace
from pyrocko import model
from pyrocko import pile
from pyrocko import gui_util
from pyrocko.gf import LocalEngine
from autogain import autogain, util_optic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
km = 1000.

if __name__ == '__main__':

    # Where is your data stored?:
    datapath = '/media/usb/webnet/pole_zero/restituted_displacement/2013Mar'
    data_pile = pile.make_pile(datapath)

    # And the stations:
    stations = model.load_stations('/media/usb/webnet/meta/stations.pf')

    # Station code of the trace you want to scale agains:
    reference_id ='NKC'

    # Frequency band to use:
    fband = {'order':4, 'corner_hp':1.0, 'corner_lp':4.}

    # And a taper to avoid filtering artefacts.
    taper = trace.CosFader(xfrac=0.25)

    # Define a window to chop traces. In this case a static length of 20
    # seconds will be used and the synthetic phase arrival will be in the 
    # center. The relative position can be changed between 0 (phase at t=0) and
    # 1 (phase at t=tmax).
    window = autogain.StaticLengthWindow(static_length=20.,
                                         phase_position=0.5)


    #candidate_fn = 'candidates2013new.pf'
    #candidates = [m.get_event() for m in gui_util.Marker.load_markers(candidate_fn)]

    # If you have a catalog of events that you want to use, load them and make
    # and event EventCollection from those events:
    #candidates = model.load_events('events.pf')

    #event_selector = autogain.EventCollection(events=candidates)

    # I selected events that I wanted to use using snuffler and wrote the
    # selected event markers into a file which is loaded here:
    markers = gui_util.load_markers('candidates2013new.pf')

    # Thus, the EventCollection accepts 'markers' or 'events' as kwargs:
    event_selector = autogain.EventCollection(markers=markers)

    # If you do not have a catalog, yet, you might want to automatically
    # download events either from Geofon or Iris.
    event_selector = autogain.EventSelectorCatalog(distmin=1000*km,
                                                   distmax=20000*km,
                                                   depthmin=2*km,
                                                   depthmax=600*km,
                                                   magmin=4.9)

    # Setup the processing:
    ag = autogain.AutoGain(data_pile, stations=stations,
                           event_selector=event_selector,
                           component='Z',
                           reference_nsl=reference_id,
                           phase_selection='first(P|p)',
                           scale_one=False)

    # Use interpolated travel times from the fomosto store 'globalttt'
    # Either you can download that store from kinherd.org but it's very large.
    # What is needed is only the ttt. So you can generate it yourself using fomosto.
    #phases = LocalEngine(store_superdirs=['/data/stores'],
    #                     default_store_id='globalttt').get_store()

    # ALTERNATIVELY, you can use the PhasePie which calles pyrocko's cake and
    # calculates travel times on demand. Might take longer:
    phases = autogain.PhasePie()
    ag.set_phaser(phases)

    ag.set_window(window)

    # start processing:
    ag.process(fband, taper)

    # Store results in YAML format:
    ag.save_mean('gains.txt')

    optics = util_optic.Optics(ag)
    optics.plot()
    optics.show(file_name='gains.pdf')
    optics.make_waveform_compare()

    # Used events can be dumped to a file:
    #event_selector.save_events('auto_events.pf')
