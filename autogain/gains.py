from pyrocko.guts import *


class Gains(Object):
    trace_gains = Dict.T(String.T(), Float.T())
