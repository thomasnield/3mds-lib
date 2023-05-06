from manim import *
import numpy as np
import scipy as sp

class DottedNumberLine(VGroup):
    def __init__(self, data: np.ndarray):
        super().__init__()

        nl = NumberLine(x_range=[np.min(data) - 1, np.max(data) + 1],
                        unit_size=1,
                        include_numbers=True,
                        ).scale_to_fit_width(12)
        dots = VGroup(*[Dot(color=BLUE).move_to(nl.n2p(d)) for d in data])

        self.add(nl, dots)

class Histogram(VGroup):
    def __init__(self, data: np.ndarray,
                 bin_count: int,
                 show_points=True,
                 show_normal_dist=False
                 ):
        super().__init__()

        # bin up points
        binned = np.histogram(data, bins=bin_count)
        # binned[0] : array of counts in each bucket
        # binned[1] : array of each boundary

        # create barchart
        self.barchart = BarChart(
            values=[float(d) for d in binned[0]],
            y_range=[0, np.max(binned[0]) + 1, 2],
            bar_width=1.0,
            bar_colors=[BLUE, RED]
        )
        self.add(self.barchart)


        # identify plot region
        min_x_raw = binned[1][0]
        max_x_raw = binned[1][-1]
        min_y_raw = 0
        max_y_raw = np.max(binned[0]) + 1
        x_len_raw = max_x_raw - min_x_raw
        y_len_raw = max_y_raw - min_y_raw
        chart_horiz_len = self.barchart.get_right()[0] - self.barchart.get_origin()[0]
        chart_vert_len = self.barchart.get_top()[1] - self.barchart.get_origin()[1]

        self.plot_region = Polygon(
             self.barchart.get_origin(),
             self.barchart.get_origin() + [chart_horiz_len, 0, 0],
             self.barchart.get_origin() + [chart_horiz_len, chart_vert_len, 0],
             self.barchart.get_origin() + [0, chart_vert_len, 0]
        )
        # create c2p function
        self.c2p = lambda c: self.barchart.get_origin() + \
                             np.array([(c[0] - min_x_raw) / x_len_raw * chart_horiz_len, 0, 0]) + \
                             np.array([0, (c[1] - min_y_raw) / y_len_raw * chart_vert_len,0])

        # create points
        self.dots = VGroup()

        for d in data:
            self.dots.add(
                Dot(color=BLUE).move_to(self.c2p([d,0]))
            )

        if show_points:
            self.add(self.dots)

        # create normal distribution plot
        axes = Axes(
            x_range=[min_x_raw, max_x_raw],
            y_range=[0, max_y_raw],
            x_length=8,
            y_length=6
        )
        self.normal_dist_plot = axes.plot(lambda x: sp.stats.norm.pdf(x, data.mean(), data.std()),
                                          use_smoothing=True,
                                          color=YELLOW) \
            .stretch_to_fit_width(chart_horiz_len) \
            .stretch_to_fit_height(chart_vert_len) \
            .align_to(self.barchart.get_origin(), LEFT) \
            .align_to(self.barchart.get_origin(), DOWN)

        if show_normal_dist:
            self.add(self.normal_dist_plot)
