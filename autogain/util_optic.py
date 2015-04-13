import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as num
from collections import OrderedDict

class Optics():
    def __init__(self, autogain):
        self.autogain = autogain
        self.set_color()

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        event_count = 0
        sections = sorted(self.autogain.get_results(), key=lambda x:
                              x.event.magnitude)
        for isection, section in enumerate(sections):
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
        ax.set_xlim([-1, len(sections)])
        self.add_event_labels(ax, sections)
        self.add_mean_lines(ax)
    
    def show(self):
        plt.show()

    def add_event_labels(self, ax, sections):
        ymin, ymax = ax.get_ylim()
        for i_s, s in enumerate(sections):
            ax.text(i_s, ymax, s.event.name, rotation='vertical')

    def add_mean_lines(self, ax):
        #ids, _means = self.autogain.mean
        _means = self.autogain.mean
        for k,v in _means.items():
            ax.axhline(y=v, c=self.color(k), label=k)

    def set_color(self):
        want = self.autogain.all_nslc_ids
        self._color_dict = dict(zip(want, num.linspace(0, 1, len(want))))

    def color(self, nslc_id):
        try:
            return cm.gist_rainbow(self._color_dict[nslc_id])
        except AttributeError: 
            self.set_color()
            return cm.gist_rainbow(self._color_dict[nslc_id])

