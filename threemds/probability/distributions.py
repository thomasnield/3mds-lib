from manim import *
import numpy as np
import scipy
import sympy

class NormalPDF(VGroup):

    def __init__(self,
                 mean: float = 0,
                 std: float = 0,
                 area_lower_range = ValueTracker(0),
                 area_upper_range = ValueTracker(0),
                 ):
        super().__init__()

        # area trackers
        self.area_lower_range = area_lower_range
        self.area_upper_range = area_upper_range

        # PDF function
        self.f_pdf = lambda x: scipy.stats.norm.pdf(x, mean, std)
        self.z_score = lambda x: (x-mean) / std

        # declare axes with default sizing
        self.axes = Axes(
            x_range=[mean-4*std, mean+4*std, std],
            y_range=[0, self.f_pdf(mean)+.1, (self.f_pdf(mean)+.1) / 4],
            x_axis_config={"include_numbers": False,
                         "numbers_to_exclude" : [mean-4*std]
            },
            y_axis_config={"include_numbers": True,
                           "decimal_number_config": {
                               "num_decimal_places": 2
                           }
            }
        )
        # create both sets of labels
        self.x_labels = VGroup(*[MathTex(round(mean+i*std,2)) \
                                   .scale(.7) \
                                   .next_to(self.axes.c2p(mean+i*std, 0), DOWN)
                               for i in range(-4,5)
                               ])

        self.x_sigma_labels = VGroup(*[MathTex(round(self.z_score(mean+i*std)), r"\sigma") \
                                    .scale(.7) \
                                    .next_to(self.axes.c2p(mean+i*std, 0), DOWN)
                               for i in range(-4,5)
                               ])

        # PDF plot
        self.pdf_plot = self.axes.plot(self.f_pdf, color=BLUE)


        # area plot
        self.area_plot = always_redraw(lambda: self.axes.get_area(self.pdf_plot,
                                                                  (self.area_lower_range.get_value(),
                                                            self.area_upper_range.get_value()),
                                                                  color=[BLUE, RED])
                                       )

        self.get_area = lambda: round(
            scipy.stats.norm.cdf(self.area_upper_range.get_value(), mean, std) -
            scipy.stats.norm.cdf(self.area_lower_range.get_value(), mean, std),
            4)

        self.add(self.axes, self.pdf_plot, self.area_plot)

        # mean line
        self.mean_line = always_redraw(lambda: DashedLine(start=self.axes.c2p(mean,0),
                                                     end=self.axes.c2p(mean, self.f_pdf(mean)),
                                                     color=YELLOW
                                                )
                                       )

        # area range label
        self.area_range_label = always_redraw(
            lambda: MathTex("P(", round(self.area_lower_range.get_value()), r"\leq X \leq",
                            round(self.area_upper_range.get_value()), ") = ",
                            format(self.get_area(), ".3f")
                            ).next_to(self.axes.c2p(mean, self.f_pdf(mean)), UP)
        )

        # sigma area range label
        self.sigma_area_range_label = always_redraw(
            lambda: MathTex("P(", round(self.z_score(self.area_lower_range.get_value())), r"\sigma \leq Z \leq",
                            round(self.z_score(self.area_upper_range.get_value())), r"\sigma) = ",
                            format(self.get_area(), ".3f")
                            ).next_to(self.axes.c2p(mean, self.f_pdf(mean)), UP)
        )


