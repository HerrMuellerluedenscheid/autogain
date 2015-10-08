# Get relative scaling factors

Install it using *setup.py* if you prefer not to operate from within this
directory.

An example how to invoke these skripts is given in *example_run.py* including
some remarks.

The entire procedure assumes that stations are "relatively close together", like
in a local network or array. However, this might still work also for stations
further apart as long as the epicentral distance is large compared to station
distances.

# What this script does
is load a waveform dataset, a file containing station information and an event
file (if the auto-download option is not used). One of the stations is taken as
a reference station against which scaling factors are calculated based on the
absolute maximum amplitude within a user-defined window. This window is
positioned in time based on synthetic phase arrivals. These can either be
extraced from a [pyrocko store](http://emolch.github.io/pyrocko/current/fomosto.html),
which encorporates interpolated travel times, or (more time consuming but maybe
more flexible) using [pyrocko's cake](http://emolch.github.io/pyrocko/current/cake_doc.html)
module which calculates phase arrivals on demand.

Results can be dumped in YAML format.


## Things one might want to add:
* SNR checker
* option to make purely synthetic comparison
* verbosity
