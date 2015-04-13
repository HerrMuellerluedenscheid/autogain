from pyrocko.guts import *


class Gains(Object):
    trace_gains = Dict.T(Tuple.T(4, String.T()), Float.T())
