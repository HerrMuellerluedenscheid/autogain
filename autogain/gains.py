from pyrocko.guts import *


class Gains(Object):
    gains = Dict.T(Tuple.T(4, String.T()), Float.T())
