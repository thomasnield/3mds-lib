from manim import *
import numpy as np
import scipy
import sympy

class NormalPDF(VGroup):

    def __init__(self, mean: float = 0,
                 std: float = 0,
                 show_mean_line = False,
                 show_area_range_label = False,
                 show_area_sigma_label = False
                 ):
        super().__init__()

        # area trackers
        self.area_lower = ValueTracker(0)
        self.area_upper = ValueTracker(0)

        # PDF function
        self.f_pdf = lambda x: scipy.stats.norm.pdf(x, mean, std)
        self.z_score = lambda x: (x-mean) / std

        # declare axes with default sizing
        self.axes = Axes(
            x_range=[mean-4*std, mean+4*std, std],
            y_range=[0, self.f_pdf(mean)+.1, (self.f_pdf(mean)+.1) / 4],
            axis_config={"include_numbers": True,
                         "numbers_to_exclude" : [mean-4*std],
                         "decimal_number_config" : {
                             "num_decimal_places" : 2
                         }
            }
        )

        # PDF plot
        self.pdf_plot = self.axes.plot(self.f_pdf, color=BLUE)

        # area plot
        self.area_plot = always_redraw(lambda: self.axes.get_area(self.pdf_plot,
                                                           (self.area_lower.get_value(),
                                                            self.area_upper.get_value()),
                                                           color=[BLUE, RED])
                                       )

        self.get_area = lambda: round(
                scipy.stats.norm.cdf(self.area_upper.get_value(), mean, std) -
                    scipy.stats.norm.cdf(self.area_lower.get_value(), mean, std),
            4)

        # mean line
        self.mean_line = always_redraw(lambda: DashedLine(start=self.axes.c2p(mean,0),
                                                     end=self.axes.c2p(mean, self.f_pdf(mean)),
                                                     color=YELLOW
                                                )
                                       )
        if show_mean_line:
            self.add(self.mean_line)

        # area range label
        self.area_range_label = always_redraw(
            lambda: MathTex("P(", round(self.area_lower.get_value()), r"\leq X \leq",
                            round(self.area_upper.get_value()), ") = ",
                            format(self.get_area(), ".4f")
                            ).next_to(self.axes.c2p(mean, self.f_pdf(mean)), UP)
        )
        if show_area_range_label:
            self.add(self.area_range_label)

        self.sigma_range_label = always_redraw(
            lambda: MathTex(r"\pm", self.z_score(self.area_lower.get_value()), r"\sigma", "=", round(self.area_lower.get_value()), r"\leq X \leq",
                            round(self.area_upper.get_value()), ") = ",
                            format(self.get_area(), ".4f")
                            ).next_to(self.axes.c2p(mean, self.f_pdf(mean)), UP)
        )
        # area sigmoid label


        self.add(self.axes, self.pdf_plot, self.area_plot)

